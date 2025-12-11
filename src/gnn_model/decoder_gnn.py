import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderGNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(DecoderGNNLayer, self).__init__()
        self.hidden_dim = hidden_dim

        # Eq 8 : GCN Weights for different edge types
        # Grandparent (gg), Parent (pc), Brother (bb), Self (ce)
        # We use 2*C' because the output is fed into GLU (which halves dimension)
        self.W_gg = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.W_pc = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.W_bb = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)
        self.W_ce = nn.Linear(hidden_dim, 2 * hidden_dim, bias=False)

        self.glu = nn.GLU(dim=-1)

        # Eq 11 & 12 : Attention Weights
        # W_h: Projects GCN output h_m^t (Eq 11)
        # W_K: Projects source nodes f^n (Key) (Eq 12)
        # W_V: Projects source nodes f^n (Value) (Eq 10)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Eq 9: Update Weights
        # W_z: Combines GCN output and Context vector (Eq 9)
        self.W_z = nn. Linear(hidden_dim, hidden_dim, bias=True)

        # OPTIONAL : Layer Normalization for training stability
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, target_nodes, target_adj, source_nodes, source_mask, initial_embeddings, is_first_layer=False):
        B, T, H = target_nodes.size()

        # Eq 8: GCN Aggregation based on edge types
        def aggregate_type(edge_type_idx, linear_layer, input_nodes):
            mask = (target_adj == edge_type_idx).float()
            # bmm(B, T, T) x (B, T, H) -> (B, T, H)
            # This sums the features of the neighbors
            neighbors = torch.bmm(mask, input_nodes)
            return linear_layer(neighbors)
        
        if is_first_layer:
            zeros = torch.zeros_like(target_nodes)
            agg_self = aggregate_type(1, self.W_ce, zeros)
        else:
            agg_self = aggregate_type(1, self.W_ce, target_nodes)
        
        agg_parent = aggregate_type(2, self.W_pc, target_nodes)
        agg_brother = aggregate_type(3, self.W_bb, target_nodes)
        agg_grand = aggregate_type(4, self.W_gg, target_nodes)

        # Eq 8 : h_m^t = glu(sum(w * neighbors))
        h_t = self.glu(agg_self + agg_parent + agg_brother + agg_grand)


        # Eq 11 : Attention
        z0_parent = torch.bmm((target_adj == 2).float(), initial_embeddings)
        z0_brother = torch.bmm((target_adj == 3).float(), initial_embeddings)
        
        query = self.W_h(h_t) + z0_parent + z0_brother  # (B, T, H)
        keys = self.W_k(source_nodes)                     # (B, N, H)
        values = self.W_v(source_nodes)                   # (B, N, H)

        attn_scores = torch.bmm(query, keys.transpose(1, 2))

        if source_mask is not None:
            attn_scores = attn_scores.masked_fill(source_mask.unsqueeze(1) == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        context = torch.bmm(attn_weights, values)         # (B, T, H)
        output = self.W_z(context + h_t)                # (B, T, H)

        # OPTIONAL : Layer Norm on resnet
        output = self.layer_norm(output + target_nodes)

        return output, attn_weights
    
class DecoderGNN(nn.Module):
    def __init__(self, num_symbols, hidden_dim, num_layers, num_edge_classes):
        super(DecoderGNN, self).__init__()
        self.hidden_dim = hidden_dim

        # z_0 : Word Embedding
        self.embedding = nn.Embedding(num_symbols, hidden_dim, padding_idx=0)

        self.layers = nn.ModuleList([
            DecoderGNNLayer(hidden_dim) for _ in range(num_layers)
        ])

        # Supervision Heads
        self.node_classifier = nn.Linear(hidden_dim, num_symbols)
        self.edge_classifier = nn.Linear(2 * hidden_dim, num_edge_classes)
        # Eq 24: W_{s,o} for Sub-graph Symbol Prediction
        self.subgraph_classifier = nn.Linear(hidden_dim, num_symbols)

    def forward(self, target_idx, target_adj, source_nodes, source_mask):
        """
        target_idx: Tensor of shape (batch_size, num_target_nodes) with node indices
        target_adj: Tensor of shape (batch_size, num_target_nodes, num_target_nodes) with edge type indices
        source_nodes: Tensor of shape (batch_size, num_source_nodes, hidden_dim)
        source_mask: Tensor of shape (batch_size, num_source_nodes) with 1s for valid nodes and 0s for padding
        """
        z_0 = self.embedding(target_idx)  # Initial node embeddings
        curr_nodes = z_0
        all_attn_weights = []

        for i, layer in enumerate(self.layers):
            is_first = (i == 0)
            curr_nodes, attn_weights = layer(curr_nodes, target_adj, source_nodes, source_mask, z_0, is_first_layer=is_first)
            all_attn_weights.append(attn_weights)

        node_logits = self.node_classifier(curr_nodes)

        z_parent = torch.bmm((target_adj == 2).float(), curr_nodes)
        edge_inputs = torch.cat([z_parent, curr_nodes], dim=-1)
        edge_logits = self.edge_classifier(edge_inputs)

        return node_logits, edge_logits, all_attn_weights
    
    def compute_loss(self, node_logits, edge_logits, all_attn_weights, source_nodes, 
                     targets, target_edge_labels, gt_stroke_alignment, padding_idx=0):
        """
        Computes the total loss inside the class, accessing self.subgraph_classifier.
        
        Args:
            node_logits: (B, T, Num_Symbols)
            edge_logits: (B, T, Num_Edge_Classes)
            all_attn_weights: List of (B, T, N)
            source_nodes: (B, N, Hidden_Dim) - needed for L_sg
            targets: (B, T) - Ground truth symbols
            target_edge_labels: (B, T) - Ground truth spatial relations
            gt_stroke_alignment: (B, T, N) - Binary mask (1 if stroke n belongs to symbol t)
            padding_idx: Index to ignore
        """
        #  1. Node Symbol Loss (L_nz) 
        vocab_size = node_logits.size(-1)
        loss_nz = F.cross_entropy(
            node_logits.view(-1, vocab_size), 
            targets.view(-1), 
            ignore_index=padding_idx
        )

        #  2. Edge Geometric Loss (L_eg) 
        num_edge_classes = edge_logits.size(-1)
        loss_eg = F.cross_entropy(
            edge_logits.view(-1, num_edge_classes), 
            target_edge_labels.view(-1), 
            ignore_index=padding_idx
        )

        #  3. Sub-graph Attention Loss (L_sa) 
        # Average attention across layers
        avg_attn = torch.stack(all_attn_weights, dim=0).mean(dim=0) # (B, T, N)
        
        # Calculate target distribution p_t(n) (Uniform over relevant strokes)
        stroke_counts = gt_stroke_alignment.sum(dim=-1, keepdim=True) # (B, T, 1)
        p_t = gt_stroke_alignment / (stroke_counts + 1e-9)
        
        # Mask for valid targets (ignore padding or symbols with no strokes)
        valid_mask = (targets != padding_idx) & (stroke_counts.squeeze(-1) > 0)
        
        # Cross Entropy: - sum( p_t * log(attn) )
        loss_sa_per_token = -torch.sum(p_t * torch.log(avg_attn + 1e-9), dim=-1)
        loss_sa = loss_sa_per_token[valid_mask].mean()

        #  4. Sub-graph Symbol Loss (L_sg) 
        # Aggregate source features based on Ground Truth alignment
        # (B, T, N) bmm (B, N, H) -> (B, T, H)
        sum_subgraph_features = torch.bmm(gt_stroke_alignment, source_nodes)
        avg_subgraph_features = sum_subgraph_features / (stroke_counts + 1e-9)
        
        subgraph_logits = self.subgraph_classifier(avg_subgraph_features)
        
        raw_loss_sg = F.cross_entropy(
            subgraph_logits.view(-1, vocab_size),
            targets.view(-1),
            reduction='none',
            ignore_index=padding_idx
        )

        loss_sg = raw_loss_sg[valid_mask.view(-1)].mean()

        # TODO: Move weights into config.py
        #  5. Weighted Sum (Eq 26) 
        # Coefficients from paper implementation details
        lambda_nz = 1.0
        lambda_eg = 1.0
        lambda_sa = 0.3
        lambda_sg = 0.5 

        total_loss = (lambda_nz * loss_nz) + \
                     (lambda_eg * loss_eg) + \
                     (lambda_sa * loss_sa) + \
                     (lambda_sg * loss_sg)

        return total_loss, {
            "loss_nz": loss_nz.item(),
            "loss_eg": loss_eg.item(),
            "loss_sa": loss_sa.item(),
            "loss_sg": loss_sg.item()
        }