import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle


def _sort_strokes(strokes):
    return sorted(strokes, key=lambda s: int(s.id) if str(s.id).isdigit() else s.id)


def _get_undirected_edges(edge_index):
    edges = set()
    if edge_index is None or edge_index.numel() == 0:
        return edges
    for src, dst in edge_index.t().tolist():
        if src == dst:
            continue
        edges.add(tuple(sorted((int(src), int(dst)))))
    return edges


def visualize_segmentation(strokes, los_edges, symbols):
    plt.figure(figsize=(10, 5))
    stroke_map = {s.id: s for s in strokes}

    for (id1, id2) in los_edges:
        s1 = stroke_map[id1]
        s2 = stroke_map[id2]
        plt.plot([s1.center[0], s2.center[0]],
                 [s1.center[1], s2.center[1]], 'k-', alpha=0.1)

    colors = cm.rainbow(np.linspace(0, 1, len(symbols)))
    for i, symbol_group in enumerate(symbols):
        color = colors[i]
        for s_id in symbol_group:
            s = stroke_map[s_id]
            plt.plot(s.points[:, 0], s.points[:, 1], color=color, linewidth=2)
            plt.text(s.center[0], s.center[1], f"Sym{i}", color=color,
                     fontsize=10, fontweight='bold')

    plt.gca().invert_yaxis()
    plt.title(f"Symbol Segmentation (Mock Classifier) - Found {len(symbols)} Symbols")
    plt.axis('equal')
    plt.show()


def visualize_gnn_graph(strokes, edge_index, title="Stroke Graph (LOS + Temporal)", show_strokes=True):
    if not strokes:
        print("No strokes to visualize.")
        return

    sorted_strokes = _sort_strokes(strokes)
    centers = np.array([s.center for s in sorted_strokes])
    plt.figure(figsize=(8, 8))

    if show_strokes:
        for stroke in sorted_strokes:
            plt.plot(stroke.points[:, 0], stroke.points[:, 1], color='black',
                     linewidth=3.0, alpha=0.8, zorder=0)

    for src, dst in _get_undirected_edges(edge_index):
        xs = [centers[src, 0], centers[dst, 0]]
        ys = [centers[src, 1], centers[dst, 1]]
        plt.plot(xs, ys, 'k-', alpha=0.4, linewidth=1.2, zorder=1)

    plt.scatter(centers[:, 0], centers[:, 1], c='royalblue', s=40, zorder=2)
    for stroke in sorted_strokes:
        plt.text(stroke.center[0], stroke.center[1], str(stroke.id), color='darkred',
                 fontsize=8, ha='center', va='center')

    plt.title(title)
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_gnn_bounding_boxes(strokes, edge_index, title="Stroke Graph with Bounding Boxes", show_strokes=True):
    if not strokes:
        print("No strokes to visualize.")
        return

    sorted_strokes = _sort_strokes(strokes)
    centers = np.array([s.center for s in sorted_strokes])
    fig, ax = plt.subplots(figsize=(8, 8))

    if show_strokes:
        for stroke in sorted_strokes:
            ax.plot(stroke.points[:, 0], stroke.points[:, 1], color='black', linewidth=3, alpha=0.7, zorder=0)

    for stroke in sorted_strokes:
        min_x, min_y = stroke.bbox_min
        max_x, max_y = stroke.bbox_max
        rect = Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                         linewidth=1.2, edgecolor='orange', facecolor='none', alpha=0.9, zorder=0)
        ax.add_patch(rect)

    for src, dst in _get_undirected_edges(edge_index):
        xs = [centers[src, 0], centers[dst, 0]]
        ys = [centers[src, 1], centers[dst, 1]]
        ax.plot(xs, ys, 'k--', alpha=0.4, linewidth=1.0, zorder=1)

    ax.scatter(centers[:, 0], centers[:, 1], c='royalblue', s=40, zorder=2)
    for stroke in sorted_strokes:
        ax.text(stroke.center[0], stroke.center[1], str(stroke.id), color='darkred', fontsize=8,
                ha='center', va='center')

    ax.set_title(title)
    ax.set_xlabel('Normalized X')
    ax.set_ylabel('Normalized Y')
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='datalim')
    fig.tight_layout()
    plt.show()


def visualize_strokes_only(strokes, title="Strokes Visualization", show_ids=True, show_centers=False):
    """Visualize strokes without edges."""
    if not strokes:
        print("No strokes to visualize.")
        return

    sorted_strokes = _sort_strokes(strokes)
    plt.figure(figsize=(8, 8))

    for stroke in sorted_strokes:
        plt.plot(stroke.points[:, 0], stroke.points[:, 1], color='black',
                 linewidth=2.5, alpha=0.8, zorder=0)

    if show_centers:
        centers = np.array([s.center for s in sorted_strokes])
        plt.scatter(centers[:, 0], centers[:, 1], c='royalblue', s=40, zorder=2)

    if show_ids:
        for stroke in sorted_strokes:
            plt.text(stroke.center[0], stroke.center[1], str(stroke.id), color='darkred',
                     fontsize=8, ha='center', va='center', zorder=3)

    plt.title(title)
    plt.xlabel('Normalized X')
    plt.ylabel('Normalized Y')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_gnn_dual(strokes, edge_index,
                       titles=("Graph Overlay", "Graph + Bounding Boxes")):
    if not strokes:
        print("No strokes to visualize.")
        return

    sorted_strokes = _sort_strokes(strokes)
    centers = np.array([s.center for s in sorted_strokes])
    edges = _get_undirected_edges(edge_index)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_graph, ax_bbox = axes

    for stroke in sorted_strokes:
        ax_graph.plot(stroke.points[:, 0], stroke.points[:, 1], color='black', linewidth=2, alpha=0.8, zorder=0)
        ax_bbox.plot(stroke.points[:, 0], stroke.points[:, 1], color='dimgray', linewidth=1.5, alpha=0.7, zorder=0)
        rect = Rectangle((stroke.bbox_min[0], stroke.bbox_min[1]),
                         stroke.bbox_max[0] - stroke.bbox_min[0],
                         stroke.bbox_max[1] - stroke.bbox_min[1],
                         linewidth=1.1, edgecolor='orange', facecolor='none', alpha=0.9, zorder=0)
        ax_bbox.add_patch(rect)

    for src, dst in edges:
        xs = [centers[src, 0], centers[dst, 0]]
        ys = [centers[src, 1], centers[dst, 1]]
        ax_graph.plot(xs, ys, 'k-', alpha=0.4, linewidth=1.2, zorder=1)
        ax_bbox.plot(xs, ys, 'k--', alpha=0.4, linewidth=1.0, zorder=1)

    for axis, title in zip(axes, titles):
        axis.scatter(centers[:, 0], centers[:, 1], c='royalblue', s=40, zorder=2)
        for stroke in sorted_strokes:
            axis.text(stroke.center[0], stroke.center[1], str(stroke.id), color='darkred', fontsize=8,
                      ha='center', va='center')
        axis.invert_yaxis()
        axis.set_aspect('equal', adjustable='datalim')
        axis.set_xlabel('Normalized X')
        axis.set_ylabel('Normalized Y')
        axis.set_title(title)

    fig.tight_layout()
    plt.show()
