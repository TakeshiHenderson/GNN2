import os
import json
import torch
import numpy as np
from tqdm import tqdm
import argparse

# --- Modules ---
import config
from source_graph import parse_inkml_and_process, build_gnn_graph
from target_graph import GroundTruthProcessor
from gnn_model import GraphToGraphModel
from train import CRHOMEDataset, g2g_collate_fn 

# Try importing visualization, fallback if missing
try:
    from source_graph import visualize_gnn_graph
except ImportError:
    from visualizations import visualize_gnn_graph

def load_best_model(vocab_len, relation_len):
    """Loads the best trained checkpoint."""
    model_conf = config.get_model_config(vocab_len, relation_len)
    model = GraphToGraphModel(model_conf).to(config.DEVICE)
    
    # Check for best model first, then fallback to latest epoch
    best_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_ep48.pth")
    latest_path = os.path.join(config.CHECKPOINT_DIR, f"checkpoint_ep{config.EPOCHS}.pth")
    
    if os.path.exists(best_path):
        checkpoint_path = best_path
    elif os.path.exists(latest_path):
        checkpoint_path = latest_path
        print(f"Warning: 'best_model.pth' not found. Using latest epoch: {latest_path}")
    else:
        raise FileNotFoundError(f"No checkpoints found in {config.CHECKPOINT_DIR}")
        
    print(f"Loading weights from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()
    return model

def tokens_to_latex(latex_tokens):
    """
    Process generated tokens to actual LaTeX syntax using relations.
    
    Converts tokens like ['y(-)', 'i(Sub)', '<EOS>(-)', '=(Right)'] 
    to clean LaTeX: 'y_{i} = ...'
    
    Relations:
        - Sub: subscript (_)
        - Sup/Above: superscript (^)
        - Right/-: horizontal (space or nothing)
        - Inside: sqrt contents
        - Below: under (for fractions, etc.)
    
    Args:
        latex_tokens: List of tokens in format 'symbol(relation)'
    
    Returns:
        str: Clean LaTeX string
    """
    result = []
    i = 0
    
    while i < len(latex_tokens):
        token = latex_tokens[i]
        
        # Parse symbol and relation
        if '(' in token:
            idx = token.rfind('(')
            symbol = token[:idx]
            relation = token[idx+1:-1]  # Remove ( and )
        else:
            symbol = token
            relation = '-'
        
        # Skip EOS, SOS, PAD tokens
        if symbol in ('<EOS>', '<SOS>', '<PAD>', ''):
            i += 1
            continue
        
        # Build LaTeX based on relation
        if relation == 'Sub':
            result.append(f'_{{{symbol}}}')
        elif relation in ('Sup', 'Above'):
            result.append(f'^{{{symbol}}}')
        elif relation == 'Inside':
            result.append(f'\\sqrt{{{symbol}}}')
        elif relation == 'Below':
            # For fractions - combine with previous if possible
            if result and not result[-1].startswith('\\frac'):
                prev = result.pop()
                result.append(f'\\frac{{{prev}}}{{{symbol}}}')
            else:
                result.append(f'_{{{symbol}}}')  # Fallback to subscript
        else:
            # Right, -, or unknown - just append the symbol
            result.append(symbol)
        
        i += 1
    
    return ' '.join(result)


def greedy_decode(model, strokes, vocab_inv, relation_inv, max_len=50):
    """
    Autoregressive greedy decoding for Graph-to-Graph model.
    
    Improved version with proper tree-structured adjacency:
    - Uses parent stack to track open nodes that can have children
    - Properly handles EOS (end-of-children) to pop back up the tree
    - Builds adjacency with parent/brother/grandparent edges like training
    """
    # Token indices from config (matches vocab.json)
    SOS_TOKEN = config.SOS_TOKEN  # <SOS> = 1
    EOS_TOKEN = config.EOS_TOKEN  # <EOS> = 2
    
    # 1. Encode Source Graph
    nodes_pts, edge_index, edge_attr = build_gnn_graph(strokes)
    num_strokes = len(nodes_pts)

    # Dense conversion for edge features
    edge_dim = config.ENC_EDGE_DIM
    dense_edge_attr = torch.zeros((1, num_strokes, num_strokes, edge_dim)).to(config.DEVICE)
    adj_mask = torch.zeros((1, num_strokes, num_strokes)).to(config.DEVICE)
    
    if edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        dense_edge_attr[0, src, dst] = edge_attr.to(config.DEVICE)
        adj_mask[0, src, dst] = 1.0
    
    # Add self-loops to ensure all strokes are visible in attention
    for i in range(num_strokes):
        adj_mask[0, i, i] = 1.0
        
    strokes_pts_list = [torch.tensor(pts, dtype=torch.float).to(config.DEVICE) for pts in nodes_pts]
    
    with torch.no_grad():
        # Encode source strokes
        stroke_node_feats = model.feature_extractor(strokes_pts_list)
        stroke_feats = stroke_node_feats.unsqueeze(0)  # (1, N, H)
        enc_nodes, _, _, _ = model.encoder(stroke_feats, dense_edge_attr, adj_mask)
        
        # Source mask: ALL strokes are valid
        source_mask = torch.ones((1, num_strokes), dtype=torch.float).to(config.DEVICE)
        
        # 2. Initialize Decoder State
        # Token sequence and tree structure tracking
        curr_tokens = [SOS_TOKEN]  # Start with SOS
        generated_syms = []
        
        # Tree structure: parent_stack tracks which nodes can have children
        # Each entry: (node_index, depth)
        parent_stack = [(0, 0)]  # SOS is root at depth 0
        
        # Track parent index for each node (for adjacency)
        node_parents = [-1]  # SOS has no parent
        node_left_brothers = [-1]  # SOS has no left brother
        
        # Initialize adjacency: just self-loop for SOS
        curr_adj = torch.zeros((1, 1, 1), dtype=torch.long).to(config.DEVICE)
        curr_adj[0, 0, 0] = 1  # Self-loop
        
        for step in range(max_len):
            # A. Feed all tokens generated so far
            tgt_seq = torch.tensor([curr_tokens], dtype=torch.long).to(config.DEVICE)
            
            # B. Forward Decoder
            node_logits, edge_logits, attention_weights = model.decoder(
                tgt_seq, 
                curr_adj, 
                enc_nodes, 
                source_mask
            )
            
            # C. Predict next symbol based on LAST position
            next_step_logits = node_logits[:, -1, :]  # (1, vocab_size)
            next_token_id = torch.argmax(next_step_logits, dim=-1).item()

            # Predict Relation
            next_edge_logits = edge_logits[:, -1, :] # (1, num_edge_classes)
            next_edge_idx = torch.argmax(next_edge_logits, dim=-1).item()
            relation_name = relation_inv.get(next_edge_idx, '?')
            
            # Debug: Print attention info
            last_layer_attn = attention_weights[-1][0]  # (T, N)
            current_step_attn = last_layer_attn[-1]  # Attention for last token
            max_attn = current_step_attn.max().item()
            attended_stroke = current_step_attn.argmax().item()
            
            token_name = vocab_inv.get(next_token_id, '?')
            print(f"Step {step}: Token: {token_name:10s} | Rel: {relation_name:8s} | Max Attn: {max_attn:.4f} | Stroke: {attended_stroke} | Stack: {len(parent_stack)}")
            
            # E. Add new node to sequence (ALWAYS add, even if EOS)
            new_node_idx = len(curr_tokens)
            curr_tokens.append(next_token_id)
            generated_syms.append(f"{token_name}({relation_name})")

            # D. Check for EOS - this means "end children of current parent"
            should_pop = False
            if next_token_id == EOS_TOKEN:
                should_pop = True
            
            # F. Determine parent and brother for this new node
            current_parent_idx, current_depth = parent_stack[-1] if parent_stack else (0, 0)
            
            # Find left brother (previous child of same parent)
            left_brother_idx = -1
            for i in range(new_node_idx - 1, 0, -1):
                if node_parents[i] == current_parent_idx:
                    left_brother_idx = i
                    break
            
            node_parents.append(current_parent_idx)
            node_left_brothers.append(left_brother_idx)
            
            # G. Build new adjacency matrix
            T_new = len(curr_tokens)
            new_adj = torch.zeros((1, T_new, T_new), dtype=torch.long).to(config.DEVICE)
            
            # Copy existing structure
            new_adj[:, :T_new-1, :T_new-1] = curr_adj
            
            # Self-loop for new node (edge type 1)
            new_adj[0, new_node_idx, new_node_idx] = 1
            
            # Parent edge: new_node -> parent (edge type 2)
            if current_parent_idx >= 0:
                new_adj[0, new_node_idx, current_parent_idx] = 2
            
            # Left brother edge: brother -> new_node (edge type 3)
            if left_brother_idx >= 0:
                new_adj[0, new_node_idx, left_brother_idx] = 3
                
            # Grandparent edge: new_node -> grandparent (edge type 4)
            if current_parent_idx > 0 and node_parents[current_parent_idx] >= 0:
                grandparent_idx = node_parents[current_parent_idx]
                new_adj[0, new_node_idx, grandparent_idx] = 4
            
            curr_adj = new_adj
            
            # H. Push this node onto stack (it can have children), UNLESS it is EOS
            if not should_pop:
                parent_stack.append((new_node_idx, current_depth + 1))
            else:
                # If it IS EOS, we are done with the Current Parent
                if len(parent_stack) > 1:
                     parent_stack.pop()
                     print(f"  -> EOS: popped stack, now at depth {len(parent_stack)}")
                else:
                     print("EOS at root - generation complete.")
                     break
            
            # Safety: prevent infinite loops
            if len(generated_syms) >= max_len:
                print("Max length reached.")
                break
            
    return generated_syms


def evaluate_test_set(num_samples=None, split='test'):
    # 1. Setup
    with open(config.VOCAB_FILE, "r") as f:
        vocabs = json.load(f)
    
    symbol_vocab = vocabs['symbol_vocab']
    relation_vocab = vocabs['relation_vocab']
    
    vocab_inv = {v: k for k, v in symbol_vocab.items()}
    relation_inv = {v: k for k, v in relation_vocab.items()}
    
    processor = GroundTruthProcessor(symbol_vocab, relation_vocab)
    
    TEST_INKML = os.path.join(config.DATA_ROOT, split, "inkml") 
    TEST_LG = os.path.join(config.DATA_ROOT, split, "lg_new_1")  # Use lg_new_1 for consistent format
    
    if split == 'train':
        TEST_INKML = config.INKML_DIR
        TEST_LG = config.LG_DIR
    elif split == 'valid':
        TEST_INKML = config.VAL_INKML_DIR
        TEST_LG = config.VAL_LG_DIR
    
    if not os.path.exists(TEST_INKML):
        print(f"{split} directory not found. Using Validation set.")
        TEST_INKML = config.VAL_INKML_DIR
        TEST_LG = config.VAL_LG_DIR

    dataset = CRHOMEDataset(TEST_INKML, TEST_LG, processor)
    
    if num_samples is not None:
        dataset.inkml_files = dataset.inkml_files[:num_samples]
        print(f"Limiting {split} set: {len(dataset)} samples")

    loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, collate_fn=g2g_collate_fn)
    
    try:
        model = load_best_model(len(symbol_vocab), len(relation_vocab))
    except FileNotFoundError as e:
        print(e)
        return

    # 4. Quantitative Evaluation
    print(f"\n--- Quantitative Evaluation [{split}] ---")
    total_loss = 0
    correct_nodes = 0
    total_nodes = 0
    correct_edges = 0
    total_edges = 0
    exprate_numerator = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            if batch is None: continue
            
            batch_gpu = {
                'batch_strokes': [ [s.to(config.DEVICE) for s in sample] for sample in batch['strokes'] ],
                'batch_edge_attr': batch['edge_attr'].to(config.DEVICE),
                'batch_adj_mask': batch['adj_mask'].to(config.DEVICE),
                'target_nodes': batch['target_nodes'].to(config.DEVICE),
                'target_adj': batch['target_adj'].to(config.DEVICE),
                'gt_alignment_mask': batch['gt_alignment'].to(config.DEVICE),
                'enc_node_targets': batch['enc_node_targets'].to(config.DEVICE),
                'enc_edge_targets': batch['enc_edge_targets'].to(config.DEVICE),
                'dec_edge_targets': batch['dec_edge_targets'].to(config.DEVICE)
            }
            
            # Forward
            # Shifted Input: 0 to T-1
            forward_args = {
                'batch_strokes': batch_gpu['batch_strokes'],
                'batch_edge_attr': batch_gpu['batch_edge_attr'],
                'batch_adj_mask': batch_gpu['batch_adj_mask'],
                'target_nodes': batch_gpu['target_nodes'][:, :-1],
                'target_adj': batch_gpu['target_adj'][:, :-1, :-1],
                'gt_alignment_mask': batch_gpu['gt_alignment_mask'][:, :-1, :]
            }
            out = model.forward_train(**forward_args)
            
            # Loss Targets
             # Shifted Target: 1 to T
            loss_targets = {
                 'enc_node_targets': batch_gpu['enc_node_targets'],
                 'enc_edge_targets': batch_gpu['enc_edge_targets'],
                 'adj_mask': batch_gpu['batch_adj_mask'],
                 'target_nodes': batch_gpu['target_nodes'][:, 1:],
                 'dec_edge_targets': batch_gpu['dec_edge_targets'][:, 1:],
                 'gt_alignment': batch_gpu['gt_alignment_mask'][:, 1:, :]
            }
            
            loss, _ = model.compute_loss(out, loss_targets)
            total_loss += loss.item()
            
            # Symbol Accuracy
            logits = out['dec_node_logits'] 
            preds = torch.argmax(logits, dim=-1)
            targets = batch_gpu['target_nodes'][:, 1:] # Shifted target
            
            # Mask: exclude PAD (0) tokens for accuracy calculation
            mask = targets != config.PAD_TOKEN
            correct = (preds == targets) & mask
            correct_nodes += correct.sum().item()
            total_nodes += mask.sum().item()
            
            # Relationship/Edge Accuracy
            edge_logits = out['dec_edge_logits']
            edge_preds = torch.argmax(edge_logits, dim=-1)
            edge_targets = batch_gpu['dec_edge_targets'][:, 1:]  # Shifted target
            
            edge_correct = (edge_preds == edge_targets) & mask
            correct_edges += edge_correct.sum().item()
            total_edges += mask.sum().item()

            if batch_idx == 0: # Debug first batch
                 print(f"DEBUG TARGET: {targets[0][:10].tolist()}")
                 print(f"DEBUG PREDS:  {preds[0][:10].tolist()}")

            # ExpRate and Error Metrics
            batch_size = preds.size(0)
            for i in range(batch_size):
                # Get target sequence (excluding PAD)
                t_seq = targets[i]
                valid_mask = t_seq != config.PAD_TOKEN
                t_seq = t_seq[valid_mask].tolist()
                
                # Get prediction sequence (same length as valid target)
                p_seq = preds[i][valid_mask].tolist()
                
                # Find first EOS in both sequences for fair comparison
                t_eos_idx = len(t_seq)
                p_eos_idx = len(p_seq)
                for idx, tok in enumerate(t_seq):
                    if tok == config.EOS_TOKEN:
                        t_eos_idx = idx + 1  # Include EOS
                        break
                for idx, tok in enumerate(p_seq):
                    if tok == config.EOS_TOKEN:
                        p_eos_idx = idx + 1  # Include EOS
                        break
                
                # Trim to first EOS (compare meaningful content only)
                t_trimmed = t_seq[:t_eos_idx]
                p_trimmed = p_seq[:p_eos_idx]
                
                # Debug: Print first 3 samples to see what's happening
                if total_samples + i < 3:
                    print(f"\n  Sample {total_samples + i}:")
                    print(f"    Target ({len(t_trimmed)} tokens): {t_trimmed[:15]}...")
                    print(f"    Pred   ({len(p_trimmed)} tokens): {p_trimmed[:15]}...")
                    print(f"    Match: {t_trimmed == p_trimmed}")
                
                # Count errors (for detailed metrics)
                t_symbols = [tok for tok in t_trimmed if tok != config.EOS_TOKEN]
                p_symbols = [tok for tok in p_trimmed if tok != config.EOS_TOKEN]
                num_errors = sum(1 for a, b in zip(t_symbols, p_symbols) if a != b)
                num_errors += abs(len(t_symbols) - len(p_symbols))
                
                # ExpRate: Exact match
                if t_trimmed == p_trimmed:
                    exprate_numerator += 1
            
            total_samples += batch_size

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    symbol_acc = correct_nodes / total_nodes * 100 if total_nodes > 0 else 0
    edge_acc = correct_edges / total_edges * 100 if total_edges > 0 else 0
    exprate = exprate_numerator / total_samples * 100 if total_samples > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"TEACHER-FORCING RESULTS [{split}]")
    print(f"{'='*50}")
    print(f"Avg Loss:           {avg_loss:.4f}")
    print(f"Symbol Accuracy:    {symbol_acc:.2f}%")
    print(f"Relation Accuracy:  {edge_acc:.2f}%")
    print(f"ExpRate (TF):       {exprate:.2f}%")
    print(f"Total Samples:      {total_samples}")
    print(f"{'='*50}")


    # 5. Qualitative Inference
    print("\n--- Generation Example ---")
    if len(dataset) > 0:
        example_idx = 1
        inkml_path = dataset.inkml_files[example_idx]
        print(f"Input: {os.path.basename(inkml_path)}")
        
        strokes = parse_inkml_and_process(inkml_path)
        
        if strokes:
            # Greedy decode
            latex_tokens = greedy_decode(model, strokes, vocab_inv, relation_inv)
            print(f"Predicted LaTeX: {tokens_to_latex(latex_tokens)}")
            print(f"Raw tokens: {latex_tokens}")
            
            lg_path = os.path.join(TEST_LG, os.path.basename(inkml_path).replace('.inkml', '.lg'))
            if os.path.exists(lg_path):
                with open(lg_path, 'r') as f:
                    gt_lines = [l.split(',')[2].strip() for l in f.readlines() if l.startswith('O,')]
                print("Ground Truth Symbols:", gt_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=None, help="Number of test files")
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'], help="Dataset split")
    args = parser.parse_args()
    
    evaluate_test_set(num_samples=args.num_samples, split=args.split)