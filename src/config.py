import torch
import os
import sys

# ==========================================
# 1. Encoder Configuration
# ==========================================
ENC_HIDDEN_DIM = 256           # Hidden dimension (C in paper)
ENC_EDGE_DIM = 21              # Geometric edge features (from edge_features.py)
ENC_LAYERS = 3                 # Number of encoder GNN blocks
ENC_NUM_NODE_CLASSES = 101     # Number of symbol classes (CROHME 2014)
ENC_NUM_EDGE_CLASSES = 2       # Binary classification: "Same Symbol" vs "Different"

# ==========================================
# 2. Feature Extractor Configuration
# ==========================================
FEAT_INPUT_DIM = 2             # (x, y) coordinates of points
FEAT_N_BLOCKS = 4              # Number of BiGRU-TCN blocks

# ==========================================
# 3. Decoder Configuration
# ==========================================
DEC_HIDDEN_DIM = 256           # Decoder embedding dimension (C' in paper)
DEC_LAYERS = 3                 # Number of decoder GNN blocks
DEC_ATTN_DIM = 400             # Dimension for Sub-graph Attention
DEC_NUM_EDGE_CLASSES = 8       # 6 Spatial Relations + Self + End-Node ('-')
                               # (Right, Above, Below, Sup, Sub, Inside, -, Self)

# ==========================================
# 4. Training Hyperparameters
# ==========================================
BATCH_SIZE = 8
LEARNING_RATE = 5e-4           # Adam Optimizer learning rate
WEIGHT_DECAY = 1e-4
EPOCHS = 50
CLIP_GRAD = 5.0                # Gradient clipping threshold
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 5. Loss Weights (Eq 26)
# ==========================================
# Encoder Supervision
LAMBDA_NF = 0.5                # L_nf: Node Classification Loss
LAMBDA_EB = 0.5                # L_eb: Edge Classification Loss

# Decoder & Sub-graph Supervision
LAMBDA_NZ = 1.0                # L_nz: Target Node Symbol Loss
LAMBDA_EG = 1.0                # L_eg: Spatial Relation Loss
LAMBDA_SA = 0.3                # L_sa: Sub-graph Attention Regularization
LAMBDA_SG = 0.5                # L_sg: Sub-graph Symbol Prediction (Eq 24)

# ==========================================
# 6. Data & Paths
# ==========================================
DATA_ROOT = "./crohme_dataset"
INKML_DIR = os.path.join(DATA_ROOT, "train", "inkml")
LG_DIR = os.path.join(DATA_ROOT, "train", "lg_new_1")
VAL_INKML_DIR = os.path.join(DATA_ROOT, "valid", "inkml")
VAL_LG_DIR = os.path.join(DATA_ROOT, "valid", "lg_new_1")

VOCAB_FILE = "./src/vocab.json"
CHECKPOINT_DIR = "./checkpoints_1"

# ==========================================
# 7. Helper Function to Export as Dict
# ==========================================
def get_model_config(vocab_len, relation_vocab_len):
    """
    Returns a dictionary with keys matching GraphToGraphModel.__init__
    """
    return {
        # Feature Extractor
        'input_dim': FEAT_INPUT_DIM,
        'hidden_dim': ENC_HIDDEN_DIM,
        'n_blocks': FEAT_N_BLOCKS,  # Matches config.get('n_blocks')

        # Encoder GNN
        'edge_input_dim': ENC_EDGE_DIM,       # Matches config.get('edge_input_dim')
        'num_encoder_layers': ENC_LAYERS,     # Matches config.get('num_encoder_layers')
        'num_source_node_classes': vocab_len, # Matches config['num_source_node_classes']
        'num_source_edge_classes': ENC_NUM_EDGE_CLASSES, # Matches config['num_source_edge_classes']

        # Decoder GNN
        'num_decoder_layers': DEC_LAYERS,     # Matches config.get('num_decoder_layers')
        'num_target_edge_classes': relation_vocab_len,   # Matches config.get('num_target_edge_classes')
    }