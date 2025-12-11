import xml.etree.ElementTree as ET
import numpy as np
from scipy. spatial import ConvexHull
from scipy.spatial.distance import cdist
import math
import torch
import os

try:
    from . visualizations import (
        visualize_gnn_graph,
        visualize_gnn_bounding_boxes,
        visualize_gnn_dual,
        visualize_strokes_only,
    )
    from .edge_features import calculate_geometric_features
    from .node_features import StrokeNodeEncoder
except ImportError:
    # Fallback for running as main script
    from visualizations import (
        visualize_gnn_graph,
        visualize_gnn_bounding_boxes,
        visualize_gnn_dual,
        visualize_strokes_only,
    )
    from edge_features import calculate_geometric_features
    from node_features import StrokeNodeEncoder

# ==========================================
# Data Structures
# ==========================================

class Stroke:
    def __init__(self, trace_id, points):
        self.id = trace_id
        self. points = np.array(points, dtype=np.float64)
        
        # Bounding Box Center for LOS
        if len(self.points) > 0:
            min_coords = np.min(self.points, axis=0)
            max_coords = np.max(self.points, axis=0)
            self.center = (min_coords + max_coords) / 2.0
            self.bbox_min = min_coords
            self. bbox_max = max_coords
        else:
            self.center = np.array([0., 0.])
            self. bbox_min = np.array([0., 0.])
            self.bbox_max = np. array([0., 0.])
        
        # Convex Hull for LOS
        if len(self.points) >= 3:
            try:
                hull = ConvexHull(self.points)
                self.hull_points = self.points[hull.vertices]
            except Exception:
                self.hull_points = self.points
        else:
            self.hull_points = self. points


class AngleIntervals:
    """Manages visibility blocking for LOS Algorithm."""
    def __init__(self):
        self.intervals = [(0, 2 * np.pi)]

    def is_visible(self, start, end):
        if start > end: 
            return self.is_visible(start, 2 * np.pi) or self.is_visible(0, end)
        for (u_start, u_end) in self.intervals:
            if not (end < u_start or start > u_end):
                return True
        return False

    def block_range(self, start, end):
        if start > end:
            self.block_range(start, 2 * np.pi)
            self.block_range(0, end)
            return
        new_intervals = []
        for (u_start, u_end) in self.intervals:
            if end < u_start or start > u_end:
                new_intervals.append((u_start, u_end))
            elif start <= u_start and end >= u_end:
                continue
            else:
                if start > u_start:
                    new_intervals.append((u_start, start))
                if end < u_end:
                    new_intervals.append((end, u_end))
        self.intervals = new_intervals

# ==========================================
# LOS Algorithm (Spatial Edges)
# ==========================================

def get_angular_range(observer_center, target_hull_points):
    ox, oy = observer_center
    angles = []
    for (tx, ty) in target_hull_points:
        theta = math.atan2(ty - oy, tx - ox)
        if theta < 0:
            theta += 2 * np.pi
        angles.append(theta)
    
    angles = np.sort(angles)
    max_gap = 0
    gap_index = -1
    for i in range(len(angles)):
        next_i = (i + 1) % len(angles)
        diff = angles[next_i] - angles[i]
        if next_i == 0:
            diff += 2 * np.pi
        if diff > max_gap:
            max_gap = diff
            gap_index = i
            
    if max_gap > np.pi:
        return angles[(gap_index + 1) % len(angles)], angles[gap_index]
    else:
        return angles[0], angles[-1]


def calculate_min_point_distance(stroke_a, stroke_b):
    dists = cdist(stroke_a.points, stroke_b.points)
    return np.min(dists)


def build_los_edges(strokes):
    """
    Constructs Spatial Edges using Line-of-Sight (LOS). 
    """
    edges = set()
    for s in strokes:
        U = AngleIntervals()
        others = [t for t in strokes if t. id != s.id]
        
        # Optimization: Filter empty strokes to avoid crash
        others = [t for t in others if len(t.points) > 0]
        if len(s.points) == 0:
            continue

        others.sort(key=lambda t: calculate_min_point_distance(s, t))
        
        for t in others:
            theta_min, theta_max = get_angular_range(s.center, t.hull_points)
            if U.is_visible(theta_min, theta_max):
                edges.add((s.id, t.id))
                U.block_range(theta_min, theta_max)
    return edges

# ==========================================
# G2G Graph Construction
# ==========================================

def build_gnn_graph(strokes):
    """
    Builds the complete G_x = (V_x, E_x) for the Graph-to-Graph model.
    Combines LOS edges and Temporal edges.
    """
    # 1.  Spatial Edges (LOS)
    los_edges = build_los_edges(strokes)
    final_edges = set(los_edges)
    
    # 2.  Temporal Edges (Bidirectional neighbors)
    sorted_strokes = sorted(strokes, key=lambda s: int(s.id) if str(s.id).isdigit() else s.id)
    stroke_map = {s.id: i for i, s in enumerate(sorted_strokes)}
    
    # add temporal edges if we have > 1 stroke
    if len(sorted_strokes) > 1:
        for i in range(len(sorted_strokes) - 1):
            id_curr = sorted_strokes[i].id
            id_next = sorted_strokes[i+1].id

            final_edges.add((id_curr, id_next))
            final_edges.add((id_next, id_curr))
    
    for s in strokes:
        final_edges.add((s.id, s.id))  # Self-loop

    # Construct PyTorch Tensors
    src_indices = []
    dst_indices = []
    edge_features = []
    
    for (id_u, id_v) in final_edges:
        u_idx = stroke_map[id_u]
        v_idx = stroke_map[id_v]
        
        stroke_u = sorted_strokes[u_idx]
        stroke_v = sorted_strokes[v_idx]
        
        # Calculate features for this specific direction u->v
        # Paper: "...for each directed edge e_{x}^{i,j}... we extract a geometric feature"
        feats_uv = calculate_geometric_features(stroke_u, stroke_v, u_idx, v_idx)
        
        src_indices.append(u_idx)
        dst_indices.append(v_idx)
        edge_features.append(feats_uv)
        
    # Handle Single Stroke
    if len(edge_features) > 0:
        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        edge_attr = torch.stack(edge_features)
    else:
        # Create empty tensors with correct dimensions
        # Shape: (2, 0) for index, (0, 21) for attr (assuming 21 feats from config)
        edge_index = torch.tensor([[], []], dtype=torch.long)
        edge_attr = torch.zeros((0, 21), dtype=torch.float)
    
    node_points_list = [s.points for s in sorted_strokes]
    
    return node_points_list, edge_index, edge_attr

# ==========================================
# Parsing & Loading (UPDATED FOR 2D/3D SUPPORT)
# ==========================================

def parse_inkml_and_process(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # 1.  Determine dimensions from <traceFormat>
    # This allows handling both 2D (Standard) and 3D (Time-series) InkML files
    num_channels = 2 
    trace_format = root.find('.//{*}traceFormat')
    if trace_format is not None:
        channels = trace_format.findall('.//{*}channel')
        if len(channels) > 0:
            num_channels = len(channels)
    
    traces = root.findall('.//{*}trace')
    
    raw_strokes = []
    for trace in traces:
        t_id = trace.get('id')
        text = trace.text
        if not text:
            continue
        
        # Parse all numbers into a flat list
        coords = text.replace(',', ' ').split()
        data = [float(x) for x in coords]
        
        # Reshape based on the specific channel count found in header
        points = None
        if len(data) > 0 and len(data) % num_channels == 0:
            # Reshape to (N_points, N_channels)
            points_nd = np.reshape(data, (-1, num_channels))
            
            # We only want the first 2 columns (X, Y), discard T/F/etc. 
            points = points_nd[:, :2]
        else:
            # Data length doesn't match the declared format (corrupted trace)
            continue
        
        # Preprocessing: Filter duplicates & Smooth
        clean_pts = [points[0]]
        for i in range(1, len(points)):
            if not np.array_equal(points[i], points[i-1]):
                clean_pts.append(points[i])
        clean_pts = np.array(clean_pts)
        
        if len(clean_pts) > 2:  # Smoothing
            smoothed = [clean_pts[0]]
            for i in range(1, len(clean_pts)-1):
                smoothed.append((clean_pts[i-1] + clean_pts[i] + clean_pts[i+1]) / 3.0)
            smoothed. append(clean_pts[-1])
            clean_pts = np.array(smoothed)

        if len(clean_pts) > 0:
            raw_strokes. append((t_id, clean_pts))

    if len(raw_strokes) == 0:
        # Instead of crashing, print a warning and return empty list
        # The Dataset class in train.py should handle this via dummy_item
        print(f"Warning: No valid strokes found in {file_path}")
        return []

    # Normalize
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


if __name__ == "__main__":
    # Test block - Visualize source graph construction
    file_path = "../crohme_dataset/test/inkml/0102.inkml"
    if os.path.exists(file_path):
        print(f"Processing: {file_path}")
        
        # Parse and build graph
        strokes = parse_inkml_and_process(file_path)
        print(f"Loaded {len(strokes)} strokes.")
        
        if len(strokes) > 0:
            node_points, edge_index, edge_attr = build_gnn_graph(strokes)
            print(f"Built graph with {len(node_points)} nodes and {edge_index.size(1)} edges")
            
            # Visualize the graph
            try:
                print("\nGenerating visualizations...")
                
                # 0. Strokes only (no edges)
                print("\n[Figure 0] Strokes Only (close window to continue)")
                visualize_strokes_only(
                    strokes,
                    title='Raw Strokes',
                    show_ids=True,
                    show_centers=False
                )
                
                # 1. Basic graph visualization
                print("\n[Figure 1] Basic Graph (close window to continue)")
                visualize_gnn_graph(
                    strokes, 
                    edge_index, 
                    title='Source Graph: Strokes + LOS + Temporal Edges',
                    show_strokes=True
                )
            except Exception as e:
                print(f"Visualization error: {e}")
    else:
        print(f"File not found: {file_path}")