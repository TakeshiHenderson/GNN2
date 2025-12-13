"""
Inference wrapper for the GNN Handwritten Equation Solver.
Loads the trained model and provides prediction functionality.
"""
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import config
from source_graph.gnn_input import Stroke, build_gnn_graph
from gnn_model import GraphToGraphModel


def tokens_to_latex(latex_tokens):
    """
    Process generated tokens to actual LaTeX syntax using relations.
    (Copied from test.py for self-contained inference)
    """
    result = []
    i = 0
    
    while i < len(latex_tokens):
        token = latex_tokens[i]
        
        # Parse symbol and relation
        if '(' in token:
            idx = token.rfind('(')
            symbol = token[:idx]
            relation = token[idx+1:-1]
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
            if result and not result[-1].startswith('\\frac'):
                prev = result.pop()
                result.append(f'\\frac{{{prev}}}{{{symbol}}}')
            else:
                result.append(f'_{{{symbol}}}')
        else:
            result.append(symbol)
        
        i += 1
    
    return ' '.join(result)


def greedy_decode(model, strokes, vocab_inv, relation_inv, max_len=50):
    """
    Autoregressive greedy decoding for Graph-to-Graph model.
    (Adapted from test.py for API usage)
    """
    SOS_TOKEN = config.SOS_TOKEN
    EOS_TOKEN = config.EOS_TOKEN
    
    # 1. Encode Source Graph
    nodes_pts, edge_index, edge_attr = build_gnn_graph(strokes)
    num_strokes = len(nodes_pts)
    
    if num_strokes == 0:
        return []

    # Dense conversion for edge features
    edge_dim = config.ENC_EDGE_DIM
    dense_edge_attr = torch.zeros((1, num_strokes, num_strokes, edge_dim)).to(config.DEVICE)
    adj_mask = torch.zeros((1, num_strokes, num_strokes)).to(config.DEVICE)
    
    if edge_index.size(1) > 0:
        src, dst = edge_index[0], edge_index[1]
        dense_edge_attr[0, src, dst] = edge_attr.to(config.DEVICE)
        adj_mask[0, src, dst] = 1.0
    
    for i in range(num_strokes):
        adj_mask[0, i, i] = 1.0
        
    strokes_pts_list = [torch.tensor(pts, dtype=torch.float).to(config.DEVICE) for pts in nodes_pts]
    
    with torch.no_grad():
        stroke_node_feats = model.feature_extractor(strokes_pts_list)
        stroke_feats = stroke_node_feats.unsqueeze(0)
        enc_nodes, _, _, _ = model.encoder(stroke_feats, dense_edge_attr, adj_mask)
        
        source_mask = torch.ones((1, num_strokes), dtype=torch.float).to(config.DEVICE)
        
        curr_tokens = [SOS_TOKEN]
        generated_syms = []
        parent_stack = [(0, 0)]
        node_parents = [-1]
        node_left_brothers = [-1]
        curr_adj = torch.zeros((1, 1, 1), dtype=torch.long).to(config.DEVICE)
        curr_adj[0, 0, 0] = 1
        
        for step in range(max_len):
            tgt_seq = torch.tensor([curr_tokens], dtype=torch.long).to(config.DEVICE)
            
            node_logits, edge_logits, attention_weights = model.decoder(
                tgt_seq, curr_adj, enc_nodes, source_mask
            )
            
            next_step_logits = node_logits[:, -1, :]
            next_token_id = torch.argmax(next_step_logits, dim=-1).item()
            next_edge_logits = edge_logits[:, -1, :]
            next_edge_idx = torch.argmax(next_edge_logits, dim=-1).item()
            relation_name = relation_inv.get(next_edge_idx, '?')
            
            token_name = vocab_inv.get(next_token_id, '?')
            
            new_node_idx = len(curr_tokens)
            curr_tokens.append(next_token_id)
            generated_syms.append(f"{token_name}({relation_name})")

            should_pop = next_token_id == EOS_TOKEN
            current_parent_idx, current_depth = parent_stack[-1] if parent_stack else (0, 0)
            
            left_brother_idx = -1
            for i in range(new_node_idx - 1, 0, -1):
                if node_parents[i] == current_parent_idx:
                    left_brother_idx = i
                    break
            
            node_parents.append(current_parent_idx)
            node_left_brothers.append(left_brother_idx)
            
            T_new = len(curr_tokens)
            new_adj = torch.zeros((1, T_new, T_new), dtype=torch.long).to(config.DEVICE)
            new_adj[:, :T_new-1, :T_new-1] = curr_adj
            new_adj[0, new_node_idx, new_node_idx] = 1
            
            if current_parent_idx >= 0:
                new_adj[0, new_node_idx, current_parent_idx] = 2
            if left_brother_idx >= 0:
                new_adj[0, new_node_idx, left_brother_idx] = 3
            if current_parent_idx > 0 and node_parents[current_parent_idx] >= 0:
                grandparent_idx = node_parents[current_parent_idx]
                new_adj[0, new_node_idx, grandparent_idx] = 4
            
            curr_adj = new_adj
            
            if not should_pop:
                parent_stack.append((new_node_idx, current_depth + 1))
            else:
                if len(parent_stack) > 1:
                    parent_stack.pop()
                else:
                    break
            
            if len(generated_syms) >= max_len:
                break
            
    return generated_syms


class EquationRecognizer:
    """Main class for equation recognition from strokes."""
    
    def __init__(self):
        self.model = None
        self.vocab_inv = None
        self.relation_inv = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained model and vocabularies."""
        # Load vocabularies
        with open(config.VOCAB_FILE, "r") as f:
            vocabs = json.load(f)
        
        symbol_vocab = vocabs['symbol_vocab']
        relation_vocab = vocabs['relation_vocab']
        
        self.vocab_inv = {v: k for k, v in symbol_vocab.items()}
        self.relation_inv = {v: k for k, v in relation_vocab.items()}
        
        # Load model
        model_conf = config.get_model_config(len(symbol_vocab), len(relation_vocab))
        self.model = GraphToGraphModel(model_conf).to(config.DEVICE)
        
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "checkpoint_ep35.pth")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from: {checkpoint_path}")
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
        self.model.eval()
        print("Model loaded successfully!")
    
    def predict(self, stroke_data: list) -> dict:
        """
        Predict LaTeX from stroke data.
        
        Args:
            stroke_data: List of dicts with 'id' and 'points' keys
                         Each 'points' is a list of [x, y] coordinates
        
        Returns:
            dict with 'latex', 'tokens', and 'tex_file' keys
        """
        # Convert to Stroke objects
        strokes = self._convert_strokes(stroke_data)
        
        if not strokes:
            return {"latex": "", "tokens": [], "tex_file": None}
        
        # Run inference
        latex_tokens = greedy_decode(
            self.model, strokes, 
            self.vocab_inv, self.relation_inv
        )
        
        # Convert to LaTeX
        latex_str = tokens_to_latex(latex_tokens)
        
        # Save to .tex file
        tex_file = self._save_tex_file(latex_str)
        
        return {
            "latex": latex_str,
            "tokens": latex_tokens,
            "tex_file": tex_file
        }
    
    def _convert_strokes(self, stroke_data: list) -> list:
        """Convert frontend stroke data to Stroke objects."""
        if not stroke_data:
            return []
        
        raw_strokes = []
        for s in stroke_data:
            points = np.array(s['points'], dtype=np.float64)
            if len(points) > 0:
                raw_strokes.append((str(s['id']), points))
        
        if not raw_strokes:
            return []
        
        # Normalize coordinates (same as parse_inkml_and_process)
        all_pts = np.vstack([p for _, p in raw_strokes])
        min_y, max_y = np.min(all_pts[:, 1]), np.max(all_pts[:, 1])
        min_x = np.min(all_pts[:, 0])
        height = max_y - min_y if max_y != min_y else 1.0
        
        strokes = []
        for s_id, pts in raw_strokes:
            norm_x = (pts[:, 0] - min_x) / height
            norm_y = (pts[:, 1] - min_y) / height
            strokes.append(Stroke(s_id, np.column_stack((norm_x, norm_y))))
        
        return strokes
    
    def _save_tex_file(self, latex_str: str) -> str:
        """Save LaTeX to a .tex file."""
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"equation_{timestamp}.tex"
        filepath = os.path.join(output_dir, filename)
        
        tex_content = f"""\\documentclass{{article}}
\\usepackage{{amsmath}}
\\begin{{document}}
$$ {latex_str} $$
\\end{{document}}
"""
        
        with open(filepath, 'w') as f:
            f.write(tex_content)
        
        return filepath
