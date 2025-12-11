# import os
# import json
# import csv

# import torch
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm

# import config
# from source_graph import parse_inkml_and_process, build_gnn_graph
# from target_graph import GroundTruthProcessor
# from gnn_model import GraphToGraphModel


# class CRHOMEDataset(Dataset):
#     def __init__(self, inkml_dir, lg_dir, processor, max_seq_len=100):
#         self.inkml_files = [os.path.join(inkml_dir, f) for f in os.listdir(inkml_dir) if f.endswith('.inkml')]
#         self.lg_dir = lg_dir
#         self.processor = processor
#         self.max_seq_len = max_seq_len
#         print(f"CRHOMEDataset initialized with {len(self.inkml_files)} samples.")

#     def __len__(self):
#         return len(self.inkml_files)
    
#     def __getitem__(self, idx):
#         inkml_path = self.inkml_files[idx]
#         file_name = os.path.basename(inkml_path).replace('.inkml', '')
#         lg_path = os.path.join(self.lg_dir, file_name + ".lg")

#         strokes = parse_inkml_and_process(inkml_path)
#         if not strokes:
#             return None
        
#         nodes_pts, edge_index, edge_attr = build_gnn_graph(strokes)
#         num_strokes = len(nodes_pts)
#         edge_dim = edge_attr.size(-1) if edge_attr.dim() > 1 else 21

#         dense_edge_attr = torch.zeros((num_strokes, num_strokes, edge_dim), dtype=torch.float)
#         adj_mask = torch.zeros((num_strokes, num_strokes), dtype=torch.float)

#         if edge_index.size(1) > 0:
#             src_indices = edge_index[0]
#             dst_indices = edge_index[1]
#             dense_edge_attr[src_indices, dst_indices] = edge_attr
#             adj_mask[src_indices, dst_indices] = 1.0
        
#         if not os.path.exists(lg_path):
#             return None
        
#         try:
#             target_data = self.processor.process(lg_path)
#             if target_data is None: return None
#         except Exception as e:
#             print(f"Error processing {lg_path}: {e}")
#             return None

#         id_to_idx = {str(s.id):i for i, s in enumerate(strokes)}

#         num_targets = len(target_data["target_nodes"])
#         gt_alignment_mask = torch.zeros((num_targets, num_strokes), dtype=torch.float)

#         raw_alignment_ids = target_data['alignment_stroke_ids']

#         for t in range(num_targets):
#             raw_ids = raw_alignment_ids[t]
#             for raw_id in raw_ids:
#                 s_id = str(raw_id)
#                 if s_id in id_to_idx:
#                     idx_in_tensor = id_to_idx[s_id]
#                     gt_alignment_mask[t, idx_in_tensor] = 1.0
        
#         enc_node_targets = torch.full((num_strokes, ), -1, dtype=torch.long)

#         for t in range(num_targets):
#             sym_class = target_data['target_nodes'][t]
#             mask = gt_alignment_mask[t] > 0
#             enc_node_targets[mask] = sym_class
        
#         enc_edge_targets = torch.zeros((num_strokes, num_strokes), dtype=torch.long)

#         for t in range(num_targets):
#             indices = torch.nonzero(gt_alignment_mask[t]).squeeze(-1)
#             if len(indices) > 1:
#                 for i in indices:
#                     for j in indices:
#                         if i != j:
#                             enc_edge_targets[i, j] = 1

#         return {
#             "strokes_pts": [torch.tensor(pts, dtype=torch.float) for pts in nodes_pts], # List of (L, 2)
#             "dense_edge_attr": dense_edge_attr, # (N, N, D)
#             "adj_mask": adj_mask,               # (N, N)
#             "enc_node_targets": enc_node_targets, # (N,)
#             "enc_edge_targets": enc_edge_targets, # (N, N)
            
#             "target_nodes": target_data['target_nodes'], # (T,)
#             "target_adj": target_data['target_adj'],     # (T, T)
#             "dec_edge_targets": target_data['dec_edge_targets'], # (T,)
#             "gt_alignment": gt_alignment_mask           # (T, N)
#         }
    
# def g2g_collate_fn(batch):
#     batch = [b for b in batch if b is not None]
#     if not batch: return None

#     # Stroke points: flattened list of lists
#     batch_strokes = [b['strokes_pts'] for b in batch]
#     batch_size = len(batch)

#     # Dense tensors (edges, masks): 2D padding
#     max_n = max([b['adj_mask'].size(0) for b in batch])
#     edge_dim = batch[0]['dense_edge_attr'].size(-1)

#     # Init padded tensors
#     batched_edge_attr = torch.zeros((batch_size, max_n, max_n, edge_dim))
#     batched_adj_mask = torch.zeros((batch_size, max_n, max_n))
#     batched_enc_node_targets = torch.full((batch_size, max_n), -1, dtype=torch.long)
#     batched_enc_edge_targets = torch.zeros((batch_size, max_n, max_n), dtype=torch.long)

#     for i, b in enumerate(batch):
#         n = b['adj_mask'].size(0)
#         batched_edge_attr[i, :n, :n] = b['dense_edge_attr']
#         batched_adj_mask[i, :n, :n] = b['adj_mask']
#         batched_enc_node_targets[i, :n] = b['enc_node_targets']
#         batched_enc_edge_targets[i, :n, :n] = b['enc_edge_targets']

#     # Target data: pad sequences
#     batched_target_nodes =  pad_sequence([b['target_nodes'] for b in batch], 
#                                                 batch_first=True, padding_value=0)
#     batched_dec_edge_targets = pad_sequence([b['dec_edge_targets'] for b in batch], 
#                                                 batch_first=True, padding_value=0)
    

#     max_t = max([b['target_adj'].size(0) for b in batch])
#     batched_target_adj = torch.zeros((batch_size, max_t, max_t), dtype=torch.long)
#     batched_gt_alignment = torch.zeros((batch_size, max_t, max_n), dtype=torch.float)

#     for i, b in enumerate(batch):
#         t = b['target_adj'].size(0)
#         n = b['adj_mask'].size(0)
#         batched_target_adj[i, :t, :t] = b['target_adj']
#         batched_gt_alignment[i, :t, :n] = b['gt_alignment']

#     return {
#         "strokes": batch_strokes,
#         "edge_attr": batched_edge_attr,
#         "adj_mask": batched_adj_mask,
#         "enc_node_targets": batched_enc_node_targets,
#         "enc_edge_targets": batched_enc_edge_targets,
#         "target_nodes": batched_target_nodes,
#         "target_adj": batched_target_adj,
#         "dec_edge_targets": batched_dec_edge_targets,
#         "gt_alignment": batched_gt_alignment
#     }


# def validate(model, val_loader, device):
#     """
#     Runs evaluation on the validation set.
#     Returns average total loss and individual metrics.
#     """
#     model.eval()
#     total_loss = 0
#     total_metrics = {}
#     count = 0
    
#     with torch.no_grad():
#         for batch in val_loader:
#             if batch is None: continue
            
#             # Move batch to GPU
#             batch_gpu = {
#                 'batch_strokes': [ [s.to(device) for s in sample] for sample in batch['strokes'] ],
#                 'batch_edge_attr': batch['edge_attr'].to(device),
#                 'batch_adj_mask': batch['adj_mask'].to(device),
                
#                 'target_nodes': batch['target_nodes'].to(device),
#                 'target_adj': batch['target_adj'].to(device),
#                 'gt_alignment_mask': batch['gt_alignment'].to(device), 

#                 'enc_node_targets': batch['enc_node_targets'].to(device),
#                 'enc_edge_targets': batch['enc_edge_targets'].to(device),
#                 'dec_edge_targets': batch['dec_edge_targets'].to(device)
#             }

#             forward_args = {
#                 'batch_strokes': batch_gpu['batch_strokes'],
#                 'batch_edge_attr': batch_gpu['batch_edge_attr'],
#                 'batch_adj_mask': batch_gpu['batch_adj_mask'],
#                 'target_nodes': batch_gpu['target_nodes'],
#                 'target_adj': batch_gpu['target_adj'],
#                 'gt_alignment_mask': batch_gpu['gt_alignment_mask']
#             }
            
#             # Run Forward Pass
#             model_out = model.forward_train(**forward_args)
            
#             # Prepare targets for loss computation
#             loss_targets = {
#                  'enc_node_targets': batch_gpu['enc_node_targets'],
#                  'enc_edge_targets': batch_gpu['enc_edge_targets'],
#                  'adj_mask': batch_gpu['batch_adj_mask'],
#                  'target_nodes': batch_gpu['target_nodes'],
#                  'dec_edge_targets': batch_gpu['dec_edge_targets'],
#                  'gt_alignment': batch_gpu['gt_alignment_mask']
#             }
            
#             # Compute Loss
#             loss, metrics = model.compute_loss(model_out, loss_targets)
            
#             total_loss += loss.item()
#             for k, v in metrics.items():
#                 total_metrics[k] = total_metrics.get(k, 0) + v
#             count += 1
            
#     avg_loss = total_loss / count if count > 0 else 0
#     avg_metrics = {k: v/count for k,v in total_metrics.items()}
#     return avg_loss, avg_metrics
    

# def train():
#     os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
#     log_csv_path = os.path.join(config.CHECKPOINT_DIR, "training_log.csv")
#     print(f"Training log will be saved to: {log_csv_path}")

#     with open(config.VOCAB_FILE, "r") as f:
#         vocabs = json.load(f)

#     symbol_vocab = vocabs.get("symbol_vocab", {})
#     relation_vocab = vocabs.get("relation_vocab", {})
#     print(f"Loaded symbol vocab of size {len(symbol_vocab)}, relation vocab of size {len(relation_vocab)}")

#     processor = GroundTruthProcessor(symbol_vocab, relation_vocab)

#     train_dataset = CRHOMEDataset(
#         inkml_dir = config.INKML_DIR,
#         lg_dir = config.LG_DIR,
#         processor = processor
#     )
#     print(f"len(train_dataset): {len(train_dataset)}")

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size = config.BATCH_SIZE,
#         shuffle = True,
#         collate_fn = g2g_collate_fn,
#         num_workers=4
#     )

#     val_dataset = CRHOMEDataset(
#         inkml_dir = config.VAL_INKML_DIR,
#         lg_dir = config.VAL_LG_DIR,
#         processor = processor
#     )

#     val_loader = DataLoader(
#         val_dataset,
#         batch_size = config.BATCH_SIZE,
#         shuffle = False,
#         collate_fn = g2g_collate_fn,
#         num_workers=2
#     )

#     print(f"Training samples: {len(train_dataset)}, \nValidation samples: {len(val_dataset)}")


#     model_config = config.get_model_config(len(symbol_vocab), len(relation_vocab))
#     model = GraphToGraphModel(model_config).to(config.DEVICE)
#     print(f"Model initialized on {config.DEVICE}")

#     optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

#     best_val_loss = float('inf')
#     metric_keys = ['loss_nf', 'loss_eb', 'loss_nz', 'loss_eg', 'loss_sa', 'loss_sg']
    
#     for epoch in range(config.EPOCHS):
#         model.train()
#         total_train_loss = 0
#         train_metrics_sum = {}
        
#         # Progress bar for training
#         pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS} [Train]")
        
#         for batch in pbar:
#             if batch is None: continue
            
#             # Move Batch to GPU
#             batch_gpu = {
#                 'batch_strokes': [ [s.to(config.DEVICE) for s in sample] for sample in batch['strokes'] ],
#                 'batch_edge_attr': batch['edge_attr'].to(config.DEVICE),
#                 'batch_adj_mask': batch['adj_mask'].to(config.DEVICE),
#                 'target_nodes': batch['target_nodes'].to(config.DEVICE),
#                 'target_adj': batch['target_adj'].to(config.DEVICE),
#                 'gt_alignment_mask': batch['gt_alignment'].to(config.DEVICE),
                
#                 # Targets for loss calculation (passed to compute_loss, not forward)
#                 'enc_node_targets': batch['enc_node_targets'].to(config.DEVICE),
#                 'enc_edge_targets': batch['enc_edge_targets'].to(config.DEVICE),
#                 'dec_edge_targets': batch['dec_edge_targets'].to(config.DEVICE)
#             }
            
#             optimizer.zero_grad()
            
#             # Forward Pass: Returns raw logits and attention weights
#             model_out = model.forward_train(**{k:v for k,v in batch_gpu.items() if k in 
#                                              ['batch_strokes', 'batch_edge_attr', 'batch_adj_mask', 
#                                               'target_nodes', 'target_adj', 'gt_alignment_mask']})
            
#             # Loss Calculation: Pass all targets for Eq 26 computation
#             # Note: We need to combine batch_gpu keys for compute_loss expected format
#             loss_targets = {
#                  'enc_node_targets': batch_gpu['enc_node_targets'],
#                  'enc_edge_targets': batch_gpu['enc_edge_targets'],
#                  'adj_mask': batch_gpu['batch_adj_mask'],
#                  'target_nodes': batch_gpu['target_nodes'],
#                  'dec_edge_targets': batch_gpu['dec_edge_targets'],
#                  'gt_alignment': batch_gpu['gt_alignment_mask']
#             }
            
#             loss, metrics = model.compute_loss(model_out, loss_targets)
            
#             # Backward
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
#             optimizer.step()
            
#             total_train_loss += loss.item()
            
#             # Accumulate training metrics
#             for k, v in metrics.items():
#                 train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v

#             # Update Pbar
#             pbar.set_postfix({
#                 'Loss': f"{loss.item():.6f}",
#                 'L_nz': f"{metrics.get('loss_nz', 0):.6f}"
#             })
            
#         # Calculate Training Averages
#         avg_train_loss = total_train_loss / len(train_loader)
#         avg_train_metrics = {k: v / len(train_loader) for k, v in train_metrics_sum.items()}
        
#         print("Running Validation...")
#         avg_val_loss, val_metrics = validate(model, val_loader, config.DEVICE)
        
#         def fmt_metrics(m):
#             return f"\nL_NF:{m.get('loss_nf',0):.6f} \n L_EB:{m.get('loss_eb',0):.6f} \n L_NZ:{m.get('loss_nz',0):.6f} \n L_EG:{m.get('loss_eg',0):.6f} \n L_SA:{m.get('loss_sa',0):.6f} \n L_SG:{m.get('loss_sg',0):.6f}"

#         print(f"Epoch {epoch+1} Summary:")
#         print(f"  Train Loss: {avg_train_loss:.6f} | {fmt_metrics(avg_train_metrics)}")
#         print(f"  Val Loss:   {avg_val_loss:.6f} | {fmt_metrics(val_metrics)}")

#         # --- CSV LOGGING ---
#         file_exists = os.path.isfile(log_csv_path)
#         with open(log_csv_path, mode='a', newline='') as csv_file:
#             # Define Headers: Epoch, Train_Loss, Val_Loss, + Train_Metrics + Val_Metrics
#             headers = ['Epoch', 'Train_Loss', 'Val_Loss'] + \
#                       [f'Train_{k}' for k in metric_keys] + \
#                       [f'Val_{k}' for k in metric_keys]
            
#             writer = csv.DictWriter(csv_file, fieldnames=headers)
            
#             if not file_exists:
#                 writer.writeheader()
            
#             # Construct Row Data
#             row_data = {
#                 'Epoch': epoch + 1,
#                 'Train_Loss': f"{avg_train_loss:.6f}",
#                 'Val_Loss': f"{avg_val_loss:.6f}"
#             }
#             # Add specific metrics
#             for k in metric_keys:
#                 row_data[f'Train_{k}'] = f"{avg_train_metrics.get(k, 0):.6f}"
#                 row_data[f'Val_{k}'] = f"{val_metrics.get(k, 0):.6f}"
            
#             writer.writerow(row_data)
        
#         # Checkpoint
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
#             print(f"  -> Best model saved! (Val Loss: {best_val_loss:.4f})")
            
#         torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, f"checkpoint_ep{epoch+1}.pth"))
    

# if __name__ == "__main__":
#     train()    

import os
import json
import csv
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import config
# --- FIX IMPORTS to match your uploaded filenames ---
from source_graph import parse_inkml_and_process, build_gnn_graph  # Was source_graph
from target_graph import GroundTruthProcessor                   # Was target_graph
from gnn_model import GraphToGraphModel

class CRHOMEDataset(Dataset):
    def __init__(self, inkml_dir, lg_dir, processor):
        # Filter for valid pairs only
        self.files = []
        ink_files = sorted([f for f in os.listdir(inkml_dir) if f.endswith('.inkml')])
        for f in ink_files:
            name = f.replace('.inkml', '')
            lg_path = os.path.join(lg_dir, name + ".lg")
            if os.path.exists(lg_path):
                self.files.append((os.path.join(inkml_dir, f), lg_path))
        
        self.processor = processor
        print(f"Dataset initialized with {len(self.files)} pairs.")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        inkml_path, lg_path = self.files[idx]
        
        # 1. Process Input (Source Graph)
        strokes = parse_inkml_and_process(inkml_path)
        if not strokes: return None
        
        nodes_pts, edge_index, edge_attr = build_gnn_graph(strokes)
        num_strokes = len(nodes_pts)
        
        # Create Dense Adjacency & Edge Attrs
        edge_dim = edge_attr.size(-1) if edge_attr.dim() > 1 else 21
        dense_edge_attr = torch.zeros((num_strokes, num_strokes, edge_dim), dtype=torch.float)
        adj_mask = torch.zeros((num_strokes, num_strokes), dtype=torch.float)

        if edge_index.size(1) > 0:
            src, dst = edge_index[0], edge_index[1]
            dense_edge_attr[src, dst] = edge_attr
            adj_mask[src, dst] = 1.0
        
        # 2. Process Target (Target Graph)
        try:
            target_data = self.processor.process(lg_path)
            if target_data is None: return None
        except Exception as e:
            print(f"Error processing {lg_path}: {e}")
            return None

        # 3. Alignments & Targets
        # Map stroke IDs (str) to tensor indices (int)
        id_to_idx = {str(s.id): i for i, s in enumerate(strokes)}
        
        num_targets = len(target_data["target_nodes"])
        gt_alignment_mask = torch.zeros((num_targets, num_strokes), dtype=torch.float)
        raw_alignment_ids = target_data['alignment_stroke_ids']

        for t in range(num_targets):
            for raw_id in raw_alignment_ids[t]:
                s_id = str(raw_id)
                if s_id in id_to_idx:
                    gt_alignment_mask[t, id_to_idx[s_id]] = 1.0
        
        # Encoder Supervision Targets
        enc_node_targets = torch.full((num_strokes, ), -1, dtype=torch.long)
        enc_edge_targets = torch.zeros((num_strokes, num_strokes), dtype=torch.long)

        # Assign symbol class to strokes
        for t in range(num_targets):
            sym_class = target_data['target_nodes'][t]
            mask = gt_alignment_mask[t] > 0
            enc_node_targets[mask] = sym_class
            
            # Assign "Same Symbol" edges
            indices = torch.nonzero(mask).squeeze(-1)
            if len(indices) > 1:
                # Create meshgrid of indices to set all pairs to 1
                grid_x, grid_y = torch.meshgrid(indices, indices, indexing='ij')
                # Exclude self-loops from "Same Symbol" relation if desired, 
                # but usually relation=1 means connected. 
                # We simply set all pairs within the symbol to 1.
                enc_edge_targets[grid_x, grid_y] = 1
                # Optional: Zero out diagonal if strictly needed, but usually fine.
                enc_edge_targets.fill_diagonal_(0)

        return {
            "strokes_pts": [torch.tensor(pts, dtype=torch.float) for pts in nodes_pts],
            "dense_edge_attr": dense_edge_attr,
            "adj_mask": adj_mask,
            "enc_node_targets": enc_node_targets,
            "enc_edge_targets": enc_edge_targets,
            "target_nodes": target_data['target_nodes'],
            "target_adj": target_data['target_adj'],
            "dec_edge_targets": target_data['dec_edge_targets'],
            "gt_alignment": gt_alignment_mask
        }

def g2g_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if not batch: return None

    batch_size = len(batch)
    batch_strokes = [b['strokes_pts'] for b in batch]

    # Find max dimensions for padding
    max_n = max([b['adj_mask'].size(0) for b in batch])
    max_t = max([b['target_adj'].size(0) for b in batch])
    edge_dim = batch[0]['dense_edge_attr'].size(-1)

    # Initialize Tensors
    batched_edge_attr = torch.zeros((batch_size, max_n, max_n, edge_dim))
    batched_adj_mask = torch.zeros((batch_size, max_n, max_n))
    batched_enc_node_targets = torch.full((batch_size, max_n), -1, dtype=torch.long)
    batched_enc_edge_targets = torch.zeros((batch_size, max_n, max_n), dtype=torch.long)
    
    batched_target_adj = torch.zeros((batch_size, max_t, max_t), dtype=torch.long)
    batched_gt_alignment = torch.zeros((batch_size, max_t, max_n), dtype=torch.float)

    for i, b in enumerate(batch):
        n = b['adj_mask'].size(0)
        t = b['target_adj'].size(0)
        
        batched_edge_attr[i, :n, :n] = b['dense_edge_attr']
        batched_adj_mask[i, :n, :n] = b['adj_mask']
        batched_enc_node_targets[i, :n] = b['enc_node_targets']
        batched_enc_edge_targets[i, :n, :n] = b['enc_edge_targets']
        
        batched_target_adj[i, :t, :t] = b['target_adj']
        batched_gt_alignment[i, :t, :n] = b['gt_alignment']

    batched_target_nodes = pad_sequence([b['target_nodes'] for b in batch], batch_first=True, padding_value=0)
    batched_dec_edge_targets = pad_sequence([b['dec_edge_targets'] for b in batch], batch_first=True, padding_value=0)

    return {
        "strokes": batch_strokes,
        "edge_attr": batched_edge_attr,
        "adj_mask": batched_adj_mask,
        "enc_node_targets": batched_enc_node_targets,
        "enc_edge_targets": batched_enc_edge_targets,
        "target_nodes": batched_target_nodes,
        "target_adj": batched_target_adj,
        "dec_edge_targets": batched_dec_edge_targets,
        "gt_alignment": batched_gt_alignment
    }

def validate(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_metrics = {}
    count = 0
    
    with torch.no_grad():
        for batch in val_loader:
            if batch is None: continue
            
            # Map batch to GPU
            batch_gpu = {
                'batch_strokes': [[s.to(device) for s in sample] for sample in batch['strokes']],
                'batch_edge_attr': batch['edge_attr'].to(device),
                'batch_adj_mask': batch['adj_mask'].to(device),
                'target_nodes': batch['target_nodes'].to(device),
                'target_adj': batch['target_adj'].to(device),
                'gt_alignment_mask': batch['gt_alignment'].to(device),
                'enc_node_targets': batch['enc_node_targets'].to(device),
                'enc_edge_targets': batch['enc_edge_targets'].to(device),
                'dec_edge_targets': batch['dec_edge_targets'].to(device)
            }

            # Forward
            model_out = model.forward_train(**{k:v for k,v in batch_gpu.items() if k in 
                                             ['batch_strokes', 'batch_edge_attr', 'batch_adj_mask', 
                                              'target_nodes', 'target_adj', 'gt_alignment_mask']})
            
            # Loss targets
            loss_targets = {
                 'enc_node_targets': batch_gpu['enc_node_targets'],
                 'enc_edge_targets': batch_gpu['enc_edge_targets'],
                 'adj_mask': batch_gpu['batch_adj_mask'],
                 'target_nodes': batch_gpu['target_nodes'],
                 'dec_edge_targets': batch_gpu['dec_edge_targets'],
                 'gt_alignment': batch_gpu['gt_alignment_mask']
            }
            
            loss, metrics = model.compute_loss(model_out, loss_targets)
            
            total_loss += loss.item()
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v
            count += 1
            
    avg_loss = total_loss / count if count > 0 else 0
    avg_metrics = {k: v/count for k,v in total_metrics.items()}
    return avg_loss, avg_metrics

def train():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    log_csv_path = os.path.join(config.CHECKPOINT_DIR, "training_log.csv")

    with open(config.VOCAB_FILE, "r") as f:
        vocabs = json.load(f)

    symbol_vocab = vocabs.get("symbol_vocab", {})
    relation_vocab = vocabs.get("relation_vocab", {})
    
    processor = GroundTruthProcessor(symbol_vocab, relation_vocab)

    train_dataset = CRHOMEDataset(config.INKML_DIR, config.LG_DIR, processor)
    val_dataset = CRHOMEDataset(config.VAL_INKML_DIR, config.VAL_LG_DIR, processor)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=g2g_collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=g2g_collate_fn, num_workers=2)

    model_config = config.get_model_config(len(symbol_vocab), len(relation_vocab))
    model = GraphToGraphModel(model_config).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)

    best_val_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        model.train()
        total_train_loss = 0
        train_metrics_sum = {}
        count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for batch in pbar:
            if batch is None: continue
            
            # Move to GPU
            batch_gpu = {
                'batch_strokes': [[s.to(config.DEVICE) for s in sample] for sample in batch['strokes']],
                'batch_edge_attr': batch['edge_attr'].to(config.DEVICE),
                'batch_adj_mask': batch['adj_mask'].to(config.DEVICE),
                'target_nodes': batch['target_nodes'].to(config.DEVICE),
                'target_adj': batch['target_adj'].to(config.DEVICE),
                'gt_alignment_mask': batch['gt_alignment'].to(config.DEVICE),
                'enc_node_targets': batch['enc_node_targets'].to(config.DEVICE),
                'enc_edge_targets': batch['enc_edge_targets'].to(config.DEVICE),
                'dec_edge_targets': batch['dec_edge_targets'].to(config.DEVICE)
            }
            
            optimizer.zero_grad()
            
            # Forward
            forward_inputs = {k:v for k,v in batch_gpu.items() if k in 
                             ['batch_strokes', 'batch_edge_attr', 'batch_adj_mask', 
                              'target_nodes', 'target_adj', 'gt_alignment_mask']}
            
            model_out = model.forward_train(**forward_inputs)
            
            # Loss Targets
            loss_targets = {
                 'enc_node_targets': batch_gpu['enc_node_targets'],
                 'enc_edge_targets': batch_gpu['enc_edge_targets'],
                 'adj_mask': batch_gpu['batch_adj_mask'],
                 'target_nodes': batch_gpu['target_nodes'],
                 'dec_edge_targets': batch_gpu['dec_edge_targets'],
                 'gt_alignment': batch_gpu['gt_alignment_mask']
            }
            
            loss, metrics = model.compute_loss(model_out, loss_targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD)
            optimizer.step()
            
            total_train_loss += loss.item()
            for k, v in metrics.items():
                train_metrics_sum[k] = train_metrics_sum.get(k, 0) + v
            count += 1
            
            pbar.set_postfix({'Loss': f"{loss.item():.4f}"})

        # Epoch End
        avg_train_loss = total_train_loss / count if count > 0 else 0
        avg_train_metrics = {k: v/count for k,v in train_metrics_sum.items()}
        
        avg_val_loss, val_metrics = validate(model, val_loader, config.DEVICE)
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Save Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))

        # Log CSV
        file_exists = os.path.isfile(log_csv_path)
        with open(log_csv_path, 'a', newline='') as f:
            headers = ['Epoch', 'Train_Loss', 'Val_Loss'] + list(avg_train_metrics.keys()) + list(val_metrics.keys())
            writer = csv.writer(f)
            if not file_exists: writer.writerow(headers)
            writer.writerow([epoch+1, avg_train_loss, avg_val_loss] + list(avg_train_metrics.values()) + list(val_metrics.values()))

if __name__ == "__main__":
    train()