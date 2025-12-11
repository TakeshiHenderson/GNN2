import torch
import torch.nn as nn
from source_graph import StrokeNodeEncoder
from gnn_model.encoder_gnn import EncoderGNN
from gnn_model.decoder_gnn import DecoderGNN


class GraphToGraphModel(nn.Module):
    """
    Main Graph-to-Graph Model.
    
    This model accepts pre-constructed graphs (nodes as stroke points, edges as geometric features)
    extracted via `gnn_input.py`.
    
    Architecture:
    1. Node Feature Extraction: Input Points -> StrokeNodeEncoder -> Initial Node Embeddings (F0)
    2. Encoder GNN: (Nodes, Edges) -> Refined Embeddings & Classification
    3. Decoder GNN: Refined Embeddings -> Target SLT Generation
    """
    def __init__(self, config):
        super(GraphToGraphModel, self).__init__()
        
        self.feature_extractor = StrokeNodeEncoder(
            input_dim=config.get('input_dim', 2),
            hidden_dim=config['hidden_dim'],
            n_blocks=config.get('n_blocks', 4)
        )

        self.encoder = EncoderGNN(
            node_input_dim=config['hidden_dim'],
            edge_input_dim=config.get('edge_input_dim', 21),
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_encoder_layers', 3),
            num_node_classes=config['num_source_node_classes'],
            num_edge_classes=config['num_source_edge_classes']
        )

        self.decoder = DecoderGNN(
            num_symbols=config['num_source_node_classes'],
            hidden_dim=config['hidden_dim'],
            num_layers=config.get('num_decoder_layers', 3),
            num_edge_classes=config.get('num_target_edge_classes')
        )

    def forward_train(self, 
                      batch_strokes, 
                      batch_edge_attr, 
                      batch_adj_mask, 
                      target_nodes, 
                      target_adj, 
                      gt_alignment_mask=None):
        """
        Forward pass for Training.
        
        Args:
            batch_strokes (List[List[Tensor]]): 
                A batch of inputs where each item is a list of stroke point tensors.
                This corresponds to the `node_points_list` from `gnn_input.py`.
                
            batch_edge_attr (Tensor): Shape (B, N, N, Edge_Dim)
                The geometric edge features calculated in `gnn_input.py`.
                
            batch_adj_mask (Tensor): Shape (B, N, N)
                Adjacency matrix derived from `edge_index` in `gnn_input.py`.
                
            target_nodes (Tensor): Shape (B, T)
                Ground Truth target symbols (for teacher forcing).
                
            target_adj (Tensor): Shape (B, T, T)
                Ground Truth SLT structure (Parent/Child/Brother edge types).
                
            gt_alignment_mask (Tensor): Shape (B, T, N)
                Ground Truth alignment between target symbols and source strokes.
                Required for Sub-graph Attention Loss (L_sa) and Sub-graph Symbol Loss (L_sg).
                
        Returns:
            dict: Raw logits and attentions needed for loss calculation.
        """
        source_node_list = []
        for strokes in batch_strokes:
            nodes = self.feature_extractor(strokes)  # (Num_Strokes, Hidden_Dim)
            source_node_list.append(nodes)
        
        source_nodes = torch.nn.utils.rnn.pad_sequence(source_node_list, batch_first=True)

        # Encoder GNN
        enc_nodes, enc_edges, enc_node_logits, enc_edge_logits = self.encoder(
            source_nodes,
            batch_edge_attr,
            batch_adj_mask
        )

        # Decoder GNN
        source_mask = (batch_adj_mask.sum(dim=-1) > 0).float()

        dec_node_logits, dec_edge_logits, attn_weights = self.decoder(
            target_nodes,
            target_adj,
            enc_nodes,
            source_mask
        )

        return {
            # Encoder outputs (for L_nf, L_eb)
            "enc_node_logits": enc_node_logits,
            "enc_edge_logits": enc_edge_logits,
            
            # Decoder outputs (for L_nz, L_eg)
            "dec_node_logits": dec_node_logits,
            "dec_edge_logits": dec_edge_logits,
            
            # Sub-graph outputs (for L_sa, L_sg)
            "attn_weights": attn_weights,
            "enc_nodes_final": enc_nodes 
        }

    def compute_loss(self, model_outputs, batch_targets):
        """
        Computes total loss for the Graph-to-Graph model.
        
        Args:
            model_outputs (dict): The output from `forward_train`.
            batch_targets (dict): Dictionary of ground truth targets.
        """
        # 1. Encoder Supervision
        loss_encoder, metrics_enc = self.encoder.compute_loss(
            model_outputs["enc_node_logits"],
            model_outputs["enc_edge_logits"],
            batch_targets['enc_node_targets'],  # GT symbol class for input strokes
            batch_targets['enc_edge_targets'],  # GT relationship for input edges
            batch_targets['adj_mask']
        )
        
        # 2. Decoder & Sub-graph Supervision
        loss_decoder, metrics_dec = self.decoder.compute_loss(
            model_outputs["dec_node_logits"],
            model_outputs["dec_edge_logits"],
            model_outputs["attn_weights"],
            model_outputs["enc_nodes_final"], 
            batch_targets['target_nodes'],      # GT target symbols
            batch_targets['dec_edge_targets'],  # GT spatial relations
            batch_targets['gt_alignment'],      # GT stroke-symbol alignment
            padding_idx=0
        )
        
        total_loss = loss_encoder + loss_decoder
        
        # Combine metrics for logging
        all_metrics = {**metrics_enc, **metrics_dec}
        all_metrics['total_loss'] = total_loss.item()

        return total_loss, all_metrics