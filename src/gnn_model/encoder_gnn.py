import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, hidden_dim, alpha=0.2):
        super(GATLayer, self).__init__()
        self.hiddn_dim = hidden_dim
        
        # Eq 4: Edge update weight
        self.W_b = nn.Linear(3 * hidden_dim, hidden_dim, bias=False)

        # Eq 7: Attention calc weight
        self.W_b_prime = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_f_prime = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Attention vector u_b (Eq 7)
        self.u_b = nn.Linear(hidden_dim, 1, bias=False)

        # Eq 5: Node Update Weights
        self.W_f = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.leaky_relu = nn.LeakyReLU(alpha)

        # OPTIONAL : Add Layer Normalization for training stability
        self.node_layer_norm = nn.LayerNorm(hidden_dim)
        self.edge_layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, nodes, edges, adj_mask):
        """
        nodes: Tensor of shape (batch_size, num_nodes, hidden_dim)
        edges: Tensor of shape (batch_size, num_nodes, num_nodes, hidden_dim)
        adj_mask: Tensor of shape (batch_size, num_nodes, num_nodes) with 1s for valid edges and 0s for no edges
        """
        batch_size, num_nodes, C = nodes.size()

        # Eq 4:  Update Edge Features
        f_i = nodes.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        f_j = nodes.unsqueeze(1).expand(-1, num_nodes, -1, -1)

        edge_concat = torch.cat([f_i, edges, f_j], dim=-1)
        new_edges = self.leaky_relu(self.W_b(edge_concat))
        new_edges = new_edges * adj_mask.unsqueeze(-1)

        # OPTIONAL : Layer Norm on resnet
        new_edges = self.edge_layer_norm(new_edges + edges)

        # For node i, we want to aggregate neighbors j. We need edge j->i.
        # So we transpose the edge matrix: edges_transposed[b, i, j] = edge j->i
        edges_in = new_edges.permute(0, 2, 1, 3)

        # Eq 6 & 7: Calculate Attention Coefficients
        term_edge = self.W_b_prime(edges_in)
        term_node_i = self.W_f_prime(f_i)
        term_node_j = self.W_f_prime(f_j)

        sum_terms = term_edge + term_node_i + term_node_j
        r_ij_unnorm = self.leaky_relu(self.u_b(sum_terms)).squeeze(-1)
        r_ij_masked = r_ij_unnorm.masked_fill(adj_mask == 0, float('-inf'))

        attention_coeffs = F.softmax(r_ij_masked, dim=2)
        # If a row was all -inf, softmax returns NaN. Replace with 0.
        attention_coeffs = torch.nan_to_num(attention_coeffs, nan=0.0)

        # Eq 5: Update Node Features
        nodes_projected = self.W_f(nodes)
        aggregated_features = torch.bmm(attention_coeffs, nodes_projected)
        new_nodes = self.leaky_relu(aggregated_features)

        # OPTIONAL : Layer Norm on resnet
        new_nodes = self.node_layer_norm(new_nodes + nodes)

        return new_nodes, new_edges
    

class EncoderGNN(nn.Module):
    def __init__(self, node_input_dim, edge_input_dim, hidden_dim, num_layers, num_node_classes, num_edge_classes):
        super(EncoderGNN, self).__init__()
        
        self.input_proj_nodes = nn.Linear(node_input_dim, hidden_dim) if node_input_dim != hidden_dim else nn.Identity()
        self.input_proj_edges = nn.Linear(edge_input_dim, hidden_dim)

        self.layers = nn.ModuleList([
            GATLayer(hidden_dim) for _ in range(num_layers)
        ])

        self.node_classifier = nn.Linear(hidden_dim, num_node_classes)
        self.edge_classifier = nn.Linear(hidden_dim, num_edge_classes)
    
    def forward(self, nodes, edges, adj_mask):
        """
        nodes: Tensor of shape (batch_size, num_nodes, input_dim)
        edges: Tensor of shape (batch_size, num_nodes, num_nodes, input_dim)
        adj_mask: Tensor of shape (batch_size, num_nodes, num_nodes) with 1s for valid edges and 0s for no edges
        """
        nodes = self.input_proj_nodes(nodes)
        edges = self.input_proj_edges(edges)

        for layer in self.layers:
            nodes, edges = layer(nodes, edges, adj_mask)

        node_logits = self.node_classifier(nodes)
        edge_logits = self.edge_classifier(edges)

        return nodes, edges, node_logits, edge_logits
    
    def compute_loss(self, node_logits, edge_logits, node_targets, edge_targets, adj_mask):
        """
        Computes the supervised loss for the Encoder GNN as per Eq 24 and 25.
        
        Args:
            node_logits: (B, N, num_symbol_classes) - Predicted symbol scores for each stroke
            edge_logits: (B, N, N, num_edge_classes) - Predicted relation scores for edges
            node_targets: (B, N) - Ground truth symbol indices (0 to num_symbol_classes-1)
            edge_targets: (B, N, N) - Ground truth edge relation indices
            adj_mask: (B, N, N) - 1.0 for valid edges, 0.0 for masked/non-existent edges
            
        Returns:
            total_encoder_loss: Weighted sum of node and edge losses
            metrics: Dictionary containing individual loss values
        """
        # TODO: Move this into config.py
        # Hyperparameters (Section 4.1 Implementation Details)
        lambda_1 = 0.5  # Weight for Node Loss 
        lambda_2 = 0.5  # Weight for Edge Loss
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

        # L_nf : Node Classification Loss
        # Flatten: (B * N, Classes) and (B * N)
        node_logits_flat = node_logits.view(-1, node_logits.size(-1))
        node_targets_flat = node_targets.view(-1)

        valid_node_mask = (node_targets_flat >= 0)
        node_loss_raw = criterion(node_logits_flat, node_targets_flat)

        if valid_node_mask.sum() > 0:
            loss_nf = (node_loss_raw * valid_node_mask.float()).sum() / valid_node_mask.sum()
        else:
            loss_nf = torch.tensor(0.0, device=node_logits.device)

        # L_ef : Edge Classification Loss
        # Flatten: (B * N * N, Classes) and (B * N * N)
        edge_logits_flat = edge_logits.reshape(-1, edge_logits.size(-1))
        edge_targets_flat = edge_targets.reshape(-1)

        valid_edge_mask = adj_mask.reshape(-1) > 0

        edge_loss_raw = criterion(edge_logits_flat, edge_targets_flat)

        if valid_edge_mask.sum() > 0:
            loss_eb = (edge_loss_raw * valid_edge_mask).sum() / valid_edge_mask.sum()
        else:
            loss_eb = torch.tensor(0.0, device=edge_logits.device)
        
        # Total Encoder Loss
        total_encoder_loss = lambda_1 * loss_nf + lambda_2 * loss_eb

        metrics = {
            "loss_nf": loss_nf.item(),
            "loss_eb": loss_eb.item()
        }

        return total_encoder_loss, metrics