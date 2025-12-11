import os
import copy
import torch
import numpy as np
import json


class SLTNode:
    """
    Helper class to build the Symbol Layout Tree (SLT).
    """
    def __init__(self, node_id, label, stroke_ids):
        self.node_id = node_id
        self.label = label
        self.stroke_ids = stroke_ids # List of stroke indices belonging to this symbol
        self.children = [] # List of tuples: (relation_label, SLTNode)
        self.parent = None
        
        # For Graph Generation
        self.dfs_index = -1
        self.left_brother = None

    def add_child(self, relation, node):
        self.children.append((relation, node))
        node.parent = self


class GroundTruthProcessor:
    """
    Parses .lg files and constructs the Target Graph (SLT) 
    according to Graph-to-Graph paper specs.
    """
    def __init__(self, vocab, relation_vocab):
        self.vocab = vocab # Dict: Symbol -> Int
        self.relation_vocab = relation_vocab # Dict: Relation Str -> Int
        
        # Special Tokens
        # self.eos_token = vocab.get('<EOS>', 1) 
        self.end_child_relation = relation_vocab.get('-', 0) # The '*' relation [cite: 185]

    def parse_lg_file(self, filepath):
        """
        Parses 'O' (Object) and 'EO' (Edge Object) lines from the LG file.
        """
        nodes = {} 
        edges = [] 
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = [p.strip() for p in line.split(',')]
            # print(parts)
            # Format: O, node_id, label, weight, stroke_ids...
            if parts[0] == 'O':
                node_id = parts[1]
                label = parts[2]
                stroke_ids = [int(s) for s in parts[4:] if s.isdigit()]
                nodes[node_id] = SLTNode(node_id, label, stroke_ids)
            
            # Format: EO, src_id, dst_id, label, weight
            elif parts[0] == 'EO':
                src_id = parts[1]
                dst_id = parts[2]
                label = parts[3]
                edges.append((src_id, dst_id, label))
        
        # print(f"Parsed {len(nodes)} nodes and {len(edges)} edges from {os.path.basename(filepath)}")
        return nodes, edges
        
    def build_slt(self, nodes, edges):
        """
        Links nodes and dynamically finds the logical Root (Node with no parents).
        REPLACES the old build_slt that relied on 'root_id'.
        """
        child_ids = set()
  
        for src, dst, label in edges:
            if src in nodes and dst in nodes:
                nodes[src].add_child(label, nodes[dst])
                child_ids.add(dst)
        
        all_ids = set(nodes.keys())
        potential_roots = list(all_ids - child_ids)
        
        if not potential_roots:
             if len(nodes) == 1:
                 return list(nodes.values())[0]
             raise ValueError(f"Cycle detected or no root found. Nodes: {len(nodes)}")

        true_root_id = potential_roots[0]
        
        return nodes[true_root_id]
    
    def add_end_nodes(self, node):
        """
        Recursively adds a virtual 'End Child' node to every node 
        as the rightmost child using the '-' relationship.
        """
        for _, child in node.children:
            self.add_end_nodes(child)
            
        end_node = SLTNode(f"{node.node_id}_EOS", '<EOS>', [])
        node.add_child('-', end_node)

    def get_dfs_sequence(self, root):
        """
        Linearizes the tree using DFS.
        """
        sequence = []
        
        def dfs(current_node):
            current_node.dfs_index = len(sequence)
            sequence.append(current_node)
            
            previous_child = None
            
            for rel, child in current_node.children:
                child.left_brother = previous_child 
                previous_child = child
                dfs(child)
                
        dfs(root)
        return sequence
    
    def process(self, filepath):
        """
        Main pipeline.
        """
        # 1. Parse
        nodes_dict, edges_list = self.parse_lg_file(filepath)
        
        if not nodes_dict:
            return None
            
        # 2. Build and Find True Root using the new logic
        try:
            root = self.build_slt(nodes_dict, edges_list)
        except ValueError as e:
            print(f"Skipping {filepath}: {e}")
            return None
        
        # 3. Add End Nodes (Recursive)
        self.add_end_nodes(root)
        
        # 4. Linearize (DFS)
        node_sequence = self.get_dfs_sequence(root)
        num_nodes = len(node_sequence)
        
        target_nodes = torch.zeros(num_nodes, dtype=torch.long)
        target_adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)
        dec_edge_targets = torch.zeros(num_nodes, dtype=torch.long)
        alignment_stroke_ids = []

        for t, node in enumerate(node_sequence):
            sym_idx = self.vocab.get(node.label, self.vocab.get('<UNK>', 0))
            target_nodes[t] = sym_idx
            alignment_stroke_ids.append(node.stroke_ids)
            
            target_adj[t, t] = 1 
            
            if node.parent:
                p_idx = node.parent.dfs_index
                target_adj[t, p_idx] = 2 
                
                for rel, child in node.parent.children:
                    if child == node:
                        rel_idx = self.relation_vocab.get(rel, 0)
                        dec_edge_targets[t] = rel_idx
                        break
            
            if node.left_brother:
                bro_idx = node.left_brother.dfs_index
                target_adj[bro_idx, t] = 3
                
            if node.parent and node.parent.parent:
                gp_idx = node.parent.parent.dfs_index
                target_adj[gp_idx, t] = 4

        return {
            "target_nodes": target_nodes,
            "target_adj": target_adj,
            "dec_edge_targets": dec_edge_targets,
            "alignment_stroke_ids": alignment_stroke_ids
        }
    
if __name__ == "__main__":
    # Load vocab from JSON file
    vocab_path = "./vocab.json"
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    # Relation Vocab MUST include '-' 
    symbol_vocab = vocab.get("symbol_vocab", {})
    relation_vocab = vocab.get("relation_vocab", {})

    processor = GroundTruthProcessor(symbol_vocab, relation_vocab)
    lg_file_path = "../crohme_dataset/train/lg_new_1/0001.lg"

    result = processor.process(lg_file_path)
    print("Result from .lg file processing:")
    print(f"Target Nodes: {result['target_nodes']}")
    print(f"Target Adjacency:\n{result['target_adj']}")
    print(f"Dec Edge Targets: {result['dec_edge_targets']}")
