import numpy as np
import torch
from scipy.spatial.distance import cdist
import math


def get_stroke_properties(points):
    """
    Calculates advanced properties for a single stroke:
    - Path Length (sum of segment distances)
    - Curvature (sum of turning angles)
    - Duration (number of points)
    """
    if len(points) < 2:
        return 0.0, 0.0, 1.0

    # 1. Segment Vectors
    # (P1-P0), (P2-P1), ...
    vecs = points[1:] - points[:-1] 
    
    # 2. Segment Lengths
    seg_lens = np.linalg.norm(vecs, axis=1)
    path_length = np.sum(seg_lens)
    
    # 3. Curvature (Sum of absolute angle changes)
    # Calculate angles of each segment
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    
    # Calculate difference between consecutive angles
    # Handle wrapping (e.g. pi to -pi is a small change, not huge)
    angle_diffs = angles[1:] - angles[:-1]
    # Normalize to [-pi, pi]
    angle_diffs = (angle_diffs + np.pi) % (2 * np.pi) - np.pi
    
    total_curvature = np.sum(np.abs(angle_diffs))
    
    # 4. Duration (Number of points)
    duration = float(len(points))
    
    return path_length, total_curvature, duration


def calculate_geometric_features(stroke_a, stroke_b, idx_a, idx_b):
    """
    Extracts the specific 19 geometric features required by G2G (Ye et al. 2019 set).
    """
    # 1. Extract Data
    pa, pb = stroke_a.points, stroke_b.points
    
    # Bounding Box Centers
    bbc_a = stroke_a.center
    bbc_b = stroke_b.center
    
    # Centroids
    cent_a = np.mean(pa, axis=0)
    cent_b = np.mean(pb, axis=0)
    
    # Endpoints
    start_a, end_a = pa[0], pa[-1]
    start_b, end_b = pb[0], pb[-1]
    
    # Advanced Properties
    len_a, curv_a, dur_a = get_stroke_properties(pa)
    len_b, curv_b, dur_b = get_stroke_properties(pb)
    
    # epsilon for division
    eps = 1e-6

    # 1. Minimum distance between 2 strokes
    dists = cdist(pa, pb)
    feat_1 = np.min(dists)

    # 2. Minimum distance between the endpoints of 2 strokes
    # Combinations: (StartA, StartB), (StartA, EndB), (EndA, StartB), (EndA, EndB)
    endpoints_a = np.array([start_a, end_a])
    endpoints_b = np.array([start_b, end_b])
    end_dists = cdist(endpoints_a, endpoints_b)
    feat_2 = np.min(end_dists)

    # 3. Maximum distance between the endpoints of 2 strokes
    feat_3 = np.max(end_dists)

    # 4. Distance between the centers of the 2 bounding boxes (BBC)
    feat_4 = np.linalg.norm(bbc_a - bbc_b)

    # 5. Horizontal distances between the centroids of 2 strokes
    feat_5 = abs(cent_a[0] - cent_b[0])

    # 6. Vertical distances between the centroids of 2 strokes
    feat_6 = abs(cent_a[1] - cent_b[1])

    # 7. Off-stroke distance between 2 strokes
    # Defined as distance from End of A -> Start of B
    off_stroke_vec = start_b - end_a
    feat_7 = np.linalg.norm(off_stroke_vec)

    # 8. Off-stroke distance projected on X- and Y-axes
    # Note: This is usually 2 features, but your list implies one line item. 
    # Usually implemented as two separate inputs to the feature vector.
    feat_8_x = off_stroke_vec[0] # Signed or Unsigned? Usually signed offset.
    feat_8_y = off_stroke_vec[1]

    # 9. Temporal distance between 2 strokes
    t_dist = float(idx_b - idx_a)
    feat_9 = abs(t_dist)

    # 10. Ratio of off-stroke distance to temporal distance
    feat_10 = feat_7 / (feat_9 + eps)

    # 11. Ratio of off-stroke distance on X-, Y-axes to temporal distance
    feat_11_x = feat_8_x / (feat_9 + eps)
    feat_11_y = feat_8_y / (feat_9 + eps)

    # --- Bounding Box Ratios ---
    w_a = stroke_a.bbox_max[0] - stroke_a.bbox_min[0]
    h_a = stroke_a.bbox_max[1] - stroke_a.bbox_min[1]
    area_a = w_a * h_a
    
    w_b = stroke_b.bbox_max[0] - stroke_b.bbox_min[0]
    h_b = stroke_b.bbox_max[1] - stroke_b.bbox_min[1]
    area_b = w_b * h_b

    # Union Box
    union_min_x = min(stroke_a.bbox_min[0], stroke_b.bbox_min[0])
    union_min_y = min(stroke_a.bbox_min[1], stroke_b.bbox_min[1])
    union_max_x = max(stroke_a.bbox_max[0], stroke_b.bbox_max[0])
    union_max_y = max(stroke_a.bbox_max[1], stroke_b.bbox_max[1])
    area_union = (union_max_x - union_min_x) * (union_max_y - union_min_y)

    # 12. Ratio of area of the largest bounding box of 2 strokes to their union
    feat_12 = max(area_a, area_b) / (area_union + eps)

    # 13. Ratio of widths of the bounding boxes of 2 strokes
    # Usually min/max or a/b. We use a/(b+eps) to allow directionality, or min/max for stability.
    # Standard practice is often min/max to bound it 0-1. Let's use a/b.
    feat_13 = w_a / (w_b + eps)

    # 14. Ratio of heights of the bounding boxes of 2 strokes
    feat_14 = h_a / (h_b + eps)

    # 15. Ratio of diagonals of the bounding boxes of 2 strokes
    diag_a = np.sqrt(w_a**2 + h_a**2)
    diag_b = np.sqrt(w_b**2 + h_b**2)
    feat_15 = diag_a / (diag_b + eps)

    # 16. Ratio of areas of the bounding boxes of 2 strokes
    feat_16 = area_a / (area_b + eps)

    # 17. Ratio of lengths of 2 strokes
    feat_17 = len_a / (len_b + eps)

    # 18. Ratio of durations of 2 strokes
    feat_18 = dur_a / (dur_b + eps)

    # 19. Ratio of curvatures of 2 strokes
    feat_19 = curv_a / (curv_b + eps)

    # Assemble Tensor
    # Note: Feature 8 and 11 are actually X and Y components.
    # If strictly mapping to 19 indices, we treat pairs as sequential.
    # Count:
    # 1, 2, 3, 4, 5, 6, 7 (1 each) = 7
    # 8 (X, Y) = 2
    # 9 (1) = 1
    # 10 (1) = 1
    # 11 (X, Y) = 2
    # 12, 13, 14, 15, 16, 17, 18, 19 (1 each) = 8
    # Total vector size = 7 + 2 + 1 + 1 + 2 + 8 = 21 dimensions
    
    features = [
        feat_1, feat_2, feat_3, feat_4, feat_5, feat_6, 
        feat_7, 
        feat_8_x, feat_8_y, # Feature 8 split
        feat_9, 
        feat_10, 
        feat_11_x, feat_11_y, # Feature 11 split
        feat_12, feat_13, feat_14, feat_15, feat_16, feat_17, feat_18, feat_19
    ]

    return torch.tensor(features, dtype=torch.float)