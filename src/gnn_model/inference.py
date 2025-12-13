import torch
import torch.nn.functional as F
import numpy as np

class G2GInference:
    def __init__(self, model, vocab, relation_vocab, device):
        self.model = model
        self.model.eval()
        self.device = device
        
        self.vocab = vocab
        self.idx_to_sym = {v: k for k, v in vocab.items()}
        self.idx_to_rel = {v: k for k, v in relation_vocab.items()}
        
        self.sos_token = vocab.get('<SOS>', 1)  # Updated: SOS is now index 1
        self.eos_token = vocab.get('<EOS>', 2)   # Updated: EOS is now index 2
        
        # For Masking logic
        self.parent_relation_idx = 2

    def predict(self, strokes, edge_attr, adj_mask, max_steps=50):
        """
        strokes: (1, N, 2)
        edge_attr: (1, N, N, D)
        adj_mask: (1, N, N)
        """
        with torch.no_grad():
            # Encode Source Graph
            source_nodes = self.model.feature_extractor([strokes[0]]) # List wrapping
            source_nodes = source_nodes.unsqueeze(0) # (1, N, H)
            
            enc_nodes, _, _, _ = self.model.encoder(source_nodes, edge_attr, adj_mask)
                        
            # List of generated node indices
            generated_syms = []
            
            # Adjacency Matrix: (1, Max, Max) - we expand it dynamically or mask it
            target_adj = torch.zeros((1, 1, 1), dtype=torch.long, device=self.device)
            target_adj[0, 0, 0] = 1 # Self-loop for first node
            
            # Input to decoder: Just the current known nodes
            curr_target_idx = torch.tensor([[self.sos_token]], device=self.device) # (1, 1)
            
            # Tracking Sub-graphs for masking
            # node_idx -> set of stroke indices (int)
            node_stroke_map = {}
            
            # Loop
            for t in range(max_steps):
                source_mask = torch.ones((1, enc_nodes.size(1)), device=self.device)
                
                # Apply Masking if we are not at root
                # "Mask all sub-graphs that are not adjacent to the corresponding sub-graph of its parent node" 
                masked_source_mask = source_mask.clone()
                if t > 0:
                    pass

                # Forward Decoder
                node_logits, edge_logits, attn_weights = self.model.decoder(
                    curr_target_idx, target_adj, enc_nodes, masked_source_mask
                )
                
                # 3. Prediction for step t (the last node in the sequence)
                last_node_logits = node_logits[:, -1, :] # (1, Vocab)
                last_edge_logits = edge_logits[:, -1, :] # (1, Rel_Vocab)
                last_attn = attn_weights[-1][:, -1, :]   # (1, N)
                
                # Greedy choice
                pred_sym = torch.argmax(last_node_logits, dim=-1).item()
                pred_rel = torch.argmax(last_edge_logits, dim=-1).item()
                
                generated_syms.append(pred_sym)
                
                # Assign strokes to this node (Max attention)
                # "The sub-graph to which the source node aligned with largest coeff belongs"
                max_stroke_idx = torch.argmax(last_attn).item()
                node_stroke_map[t] = {max_stroke_idx} # In real impl, use connected components
                
                if pred_sym == self.eos_token:
                    break
                    

                # Add new node to inputs
                new_idx = torch.tensor([[pred_sym]], device=self.device)
                curr_target_idx = torch.cat([curr_target_idx, new_idx], dim=1)
                
                # Expand Adjacency
                T_curr = target_adj.size(1)
                new_adj = torch.zeros((1, T_curr+1, T_curr+1), device=self.device)
                new_adj[:, :T_curr, :T_curr] = target_adj
                new_adj[:, T_curr, T_curr] = 1 # Self

                # Placeholder: Assume previous node is parent (Linear chain)
                parent_idx = T_curr - 1
                new_adj[:, T_curr, parent_idx] = 2 # Child -> Parent
                
                target_adj = new_adj
                
        return [self.idx_to_sym.get(s, '?') for s in generated_syms]