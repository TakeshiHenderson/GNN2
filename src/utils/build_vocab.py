import os
import glob
import json
from tqdm import tqdm

def build_vocab_from_dataset(data_dir, output_file="vocab.json"):
    """
    Scans all .lg files in data_dir to build symbol and relation vocabularies.
    
    Args:
        data_dir (str): Path to the directory containing .lg files (e.g., 'data/train/')
        output_file (str): Path to save the resulting JSON dictionary.
    """
    
    # 1. Initialize Sets to store unique labels
    unique_symbols = set()
    unique_relations = set()
    
    # Get list of all .lg files
    lg_files = glob.glob(os.path.join(data_dir, "**", "*.lg"), recursive=True)
    
    print(f"Found {len(lg_files)} .lg files. Scanning...")
    
    for filepath in tqdm(lg_files):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = [p.strip() for p in line.split(',')]
                
                # Parse Object Lines for Symbols
                # Format: O, node_id, label, ...
                if parts[0] == 'O':
                    if len(parts) >= 3:
                        label = parts[2]
                        unique_symbols.add(label)
                        
                # Parse Edge Lines for Relations
                # Format: EO, src, dst, label, ...
                elif parts[0] == 'EO':
                    if len(parts) >= 4:
                        label = parts[3]
                        unique_relations.add(label)
                        
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # 2. Build Symbol Vocabulary
    # Standard Special Tokens
    # <SOS> is required for autoregressive decoding (inference)
    symbol_vocab = {
        '<PAD>': 0, 
        '<SOS>': 1,
        '<EOS>': 2, 
        '<UNK>': 3
    }
    
    # Sort symbols for deterministic indices
    sorted_symbols = sorted(list(unique_symbols))
    
    start_idx = len(symbol_vocab)
    for i, sym in enumerate(sorted_symbols):
        symbol_vocab[sym] = start_idx + i
        
    # 3. Build Relation Vocabulary
    # 0 is reserved for padding/no-edge
    relation_vocab = {'<PAD>': 0}
    
    # Important: Ensure the End-Child relation '-' exists
    # [cite_start]The paper uses '-' for the virtual end nodes [cite: 204]
    if '-' not in unique_relations:
        unique_relations.add('-')
        
    sorted_relations = sorted(list(unique_relations))
    
    start_idx_rel = len(relation_vocab)
    for i, rel in enumerate(sorted_relations):
        relation_vocab[rel] = start_idx_rel + i

    # 4. Save to file
    final_output = {
        "symbol_vocab": symbol_vocab,
        "relation_vocab": relation_vocab
    }
    
    with open(output_file, 'w') as f:
        json.dump(final_output, f, indent=4)
        
    print(f"\nSuccess! Vocab saved to {output_file}")
    print(f"Total Symbols: {len(symbol_vocab)}")
    print(f"Total Relations: {len(relation_vocab)}")
    print("-" * 30)
    print("Relations found:", sorted_relations)

if __name__ == "__main__":
    # CONFIGURATION
    # Change this to the folder containing your .lg files
    DATASET_DIR = "../crohme_dataset/train/lg/" 
    
    build_vocab_from_dataset(DATASET_DIR)