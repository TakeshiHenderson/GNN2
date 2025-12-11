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
    best_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
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

def greedy_decode(model, strokes, vocab_inv, relation_inv, max_len=50):
    """
    Corrected greedy generation using Placeholder Strategy.
    """
    # 1. Encode Source
    nodes_pts, edge_index, edge_attr = build_gnn_graph(strokes)
    num_strokes = len(nodes_pts)

    # Dense conversion
    edge_dim = config.ENC_EDGE_DIM
    dense_edge_attr = torch.zeros((1, num_strokes, num_strokes, edge_dim)).to(config.DEVICE)
    adj_mask = torch.zeros((1, num_strokes, num_strokes)).to(config.DEVICE)
    
    if edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        dense_edge_attr[0, src, dst] = edge_attr.to(config.DEVICE)
        adj_mask[0, src, dst] = 1.0
        
    strokes_pts_list = [torch.tensor(pts, dtype=torch.float).to(config.DEVICE) for pts in nodes_pts]
    
    with torch.no_grad():
        # Encode
        stroke_node_feats = model.feature_extractor(strokes_pts_list)
        stroke_feats = stroke_node_feats.unsqueeze(0)
        enc_nodes, _, _, _ = model.encoder(stroke_feats, dense_edge_attr, adj_mask)
        
        # 2. Decode Loop
        # Start with EMPTY confirmed sequence
        curr_tokens = [] 
        generated_syms = []
        
        # Start Adjacency is empty
        curr_adj = torch.zeros((1, 0, 0), dtype=torch.long).to(config.DEVICE)
        
        for step in range(max_len):
            # A. Prepare Input: Add a PLACEHOLDER (0) for the node we want to predict
            # If curr_tokens is [A, B], we feed [A, B, 0]
            temp_tokens = curr_tokens + [0]
            tgt_seq = torch.tensor([temp_tokens], dtype=torch.long).to(config.DEVICE)
            
            # B. Prepare Adjacency: Add row/col for the placeholder
            T_prev = len(curr_tokens)
            T_new = T_prev + 1
            new_adj = torch.zeros((1, T_new, T_new), dtype=torch.long).to(config.DEVICE)
            
            if T_prev > 0:
                new_adj[:, :T_prev, :T_prev] = curr_adj # Copy existing graph
                
                # Heuristic: Connect new node to immediate previous as Parent (2)
                # This ensures the GNN has a path to flow context to the new node.
                new_adj[:, T_prev-1, T_prev] = 2 
            
            # Always add Self-Loop (1) for the new node (required for GNN)
            new_adj[:, T_prev, T_prev] = 1
            
            # C. Forward Decoder
            source_mask = (adj_mask.sum(dim=-1) > 0).float()
            
            node_logits, edge_logits, attention_weights = model.decoder(
                tgt_seq, 
                new_adj, 
                enc_nodes, 
                source_mask
            )

            
            # D. Predict Symbol for the LAST node (the placeholder)
            next_step_logits = node_logits[:, -1, :] 
            next_token_id = torch.argmax(next_step_logits, dim=-1).item()
            
            last_layer_attn = attention_weights[-1][0] # (T, N)
            # Look at the attention distribution for the current step (last row)
            current_step_attn = last_layer_attn[-1]
            max_attn = current_step_attn.max().item()
            print(f"Step {len(curr_tokens)}: Max Attn: {max_attn:.4f} | Token: {vocab_inv.get(next_token_id)}")
            # Check EOS (Assuming EOS index is 1, check your vocab.json)
            if next_token_id == 1: 
                break
            
            # E. Commit the prediction
            curr_tokens.append(next_token_id)
            
            # Update Adjacency permanently for the next step
            curr_adj = new_adj 
            
            generated_syms.append(vocab_inv.get(next_token_id, '?'))
            
    return generated_syms

def evaluate_test_set(num_samples=None):
    # 1. Setup
    with open(config.VOCAB_FILE, "r") as f:
        vocabs = json.load(f)
    
    symbol_vocab = vocabs['symbol_vocab']
    relation_vocab = vocabs['relation_vocab']
    
    vocab_inv = {v: k for k, v in symbol_vocab.items()}
    relation_inv = {v: k for k, v in relation_vocab.items()}
    
    processor = GroundTruthProcessor(symbol_vocab, relation_vocab)
    
    TEST_INKML = os.path.join(config.DATA_ROOT, "test", "inkml") 
    TEST_LG = os.path.join(config.DATA_ROOT, "test", "lg")
    
    if not os.path.exists(TEST_INKML):
        print(f"Test directory not found. Using Validation set.")
        TEST_INKML = config.VAL_INKML_DIR
        TEST_LG = config.VAL_LG_DIR

    dataset = CRHOMEDataset(TEST_INKML, TEST_LG, processor)
    
    if num_samples is not None:
        dataset.inkml_files = dataset.inkml_files[:num_samples]
        print(f"Limiting test set: {len(dataset)} samples")

    loader = torch.utils.data.DataLoader(dataset, batch_size=config.BATCH_SIZE, collate_fn=g2g_collate_fn)
    
    try:
        model = load_best_model(len(symbol_vocab), len(relation_vocab))
    except FileNotFoundError as e:
        print(e)
        return

    # 4. Quantitative Evaluation
    print("\n--- Quantitative Evaluation ---")
    total_loss = 0
    correct_nodes = 0
    total_nodes = 0
    
    with torch.no_grad():
        for batch in tqdm(loader):
            if batch is None: continue
            
            batch_gpu = {
                'batch_strokes': [ [s.to(config.DEVICE) for s in sample] for sample in batch['strokes'] ],
                'batch_edge_attr': batch['edge_attr'].to(config.DEVICE),
                'batch_adj_mask': batch['adj_mask'].to(config.DEVICE),
                'target_nodes': batch['target_nodes'].to(config.DEVICE),
                'target_adj': batch['target_adj'].to(config.DEVICE),
                'gt_alignment_mask': batch['gt_alignment'].to(config.DEVICE)
            }
            
            # Forward
            out = model.forward_train(**batch_gpu)
            
            # Loss Targets
            loss_targets = {
                 'enc_node_targets': batch['enc_node_targets'].to(config.DEVICE),
                 'enc_edge_targets': batch['enc_edge_targets'].to(config.DEVICE),
                 'adj_mask': batch['adj_mask'].to(config.DEVICE),
                 'target_nodes': batch['target_nodes'].to(config.DEVICE),
                 'dec_edge_targets': batch['dec_edge_targets'].to(config.DEVICE),
                 'gt_alignment': batch['gt_alignment'].to(config.DEVICE)
            }
            
            loss, _ = model.compute_loss(out, loss_targets)
            total_loss += loss.item()
            
            # Accuracy
            logits = out['dec_node_logits'] 
            preds = torch.argmax(logits, dim=-1)
            targets = batch['target_nodes'].to(config.DEVICE)
            
            mask = targets != 0 
            correct = (preds == targets) & mask
            correct_nodes += correct.sum().item()
            total_nodes += mask.sum().item()

    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0
    acc = correct_nodes / total_nodes * 100 if total_nodes > 0 else 0
    
    print(f"Avg Loss: {avg_loss:.4f}")
    print(f"Symbol Accuracy: {acc:.2f}%")

    # 5. Qualitative Inference
    print("\n--- Generation Example ---")
    if len(dataset) > 0:
        example_idx = 0 
        inkml_path = dataset.inkml_files[example_idx]
        print(f"Input: {os.path.basename(inkml_path)}")
        
        strokes = parse_inkml_and_process(inkml_path)
        
        if strokes:
            latex_tokens = greedy_decode(model, strokes, vocab_inv, relation_inv)
            print(f"Predicted LaTeX: {' '.join(latex_tokens)}")
            
            lg_path = os.path.join(TEST_LG, os.path.basename(inkml_path).replace('.inkml', '.lg'))
            if os.path.exists(lg_path):
                with open(lg_path, 'r') as f:
                    gt_lines = [l.split(',')[2].strip() for l in f.readlines() if l.startswith('O,')]
                print("Ground Truth Symbols:", gt_lines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_samples', type=int, default=None, help="Number of test files")
    args = parser.parse_args()
    
    evaluate_test_set(num_samples=args.num_samples)