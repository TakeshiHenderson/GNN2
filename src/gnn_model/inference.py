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
            # 1. Encode Source Graph
            source_nodes = self.model.feature_extractor([strokes[0]]) # List wrapping
            source_nodes = source_nodes.unsqueeze(0) # (1, N, H)
            
            enc_nodes, _, _, _ = self.model.encoder(source_nodes, edge_attr, adj_mask)
            
            # 2. Initialize Decoder State
            # Start with just the Root node (or SOS if your vocab requires)
            # We treat the first generated symbol as Root.
            
            # List of generated node indices
            generated_syms = []
            
            # Adjacency Matrix: (1, Max, Max) - we expand it dynamically or mask it
            # We'll build it incrementally
            target_adj = torch.zeros((1, 1, 1), dtype=torch.long, device=self.device)
            target_adj[0, 0, 0] = 1 # Self-loop for first node
            
            # Input to decoder: Just the current known nodes
            curr_target_idx = torch.tensor([[self.sos_token]], device=self.device) # (1, 1)
            
            # Tracking Sub-graphs for masking
            # node_idx -> set of stroke indices (int)
            node_stroke_map = {}
            
            # Loop
            for t in range(max_steps):
                # CRITICAL FIX: Ensure all strokes are visible (not just connected ones)
                source_mask = torch.ones((1, enc_nodes.size(1)), device=self.device)
                
                # Apply Masking if we are not at root
                # "Mask all sub-graphs that are not adjacent to the corresponding sub-graph of its parent node" 
                masked_source_mask = source_mask.clone()
                if t > 0:
                    # Find parent of current node t
                    # In our adj matrix [child, parent] = 2.
                    # But we haven't generated t yet! We are predicting t.
                    # We are actually predicting the attributes of node t-1? 
                    # No, usually we input t-1 tokens to predict t-th token.
                    # But GNN decoder inputs *all generated nodes so far* to predict attributes of the *next* node?
                    # Actually, the paper generates (Node, Edge) at step t.
                    pass

                # Forward Decoder
                # We feed ALL generated nodes so far. 
                # The decoder outputs logits for these nodes. 
                # We check the output of the LAST node to determine what to do next?
                # Actually, standard Seq2Seq feeds [0...t] to predict [t+1].
                # But this is a graph. We assume we are predicting the properties of node 't'.
                
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
                # "The sub-graph to which the source node aligned with largest coeff belongs" [cite: 286]
                # Here we simplify: take the stroke with max attention
                max_stroke_idx = torch.argmax(last_attn).item()
                node_stroke_map[t] = {max_stroke_idx} # In real impl, use connected components
                
                if pred_sym == self.eos_token:
                    break
                    
                # 4. Prepare for next step
                # We need to determine where this new node connects.
                # The 'edge_logits' tells us the relation to the PARENT.
                # But WHICH node is the parent? 
                # The paper implies a Tree structure traversal (DFS).
                # In DFS, the parent is usually the previous node, or an ancestor if we backtracked.
                # This is complex. For now, we assume simple DFS linear expansion 
                # where we rely on the SLT structure logic defined in ground_truth.py
                
                # Add new node to inputs
                new_idx = torch.tensor([[pred_sym]], device=self.device)
                curr_target_idx = torch.cat([curr_target_idx, new_idx], dim=1)
                
                # Expand Adjacency
                T_curr = target_adj.size(1)
                new_adj = torch.zeros((1, T_curr+1, T_curr+1), device=self.device)
                new_adj[:, :T_curr, :T_curr] = target_adj
                new_adj[:, T_curr, T_curr] = 1 # Self
                
                # We need logic to determine the parent index based on `pred_rel`.
                # This usually requires a stack-based parser logic (SR parser).
                # Since this is a pure GNN approach, the paper uses "Decoder GNN" to infer structure.
                # For simplified inference, we assume the previous node is parent 
                # UNLESS the relation indicates otherwise (e.g. Right/Above/Below usually implied parent is prev).
                
                # Placeholder: Assume previous node is parent (Linear chain)
                parent_idx = T_curr - 1
                new_adj[:, T_curr, parent_idx] = 2 # Child -> Parent
                
                target_adj = new_adj
                
        return [self.idx_to_sym.get(s, '?') for s in generated_syms]