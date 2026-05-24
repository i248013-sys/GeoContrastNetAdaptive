"""
Build baseline KNN and adaptive graph variants for comparison.
Runs stage-1 (graph construction + feature building) for both topologies.
Outputs graphs as .bin files for stage-2 training.

Usage:
  python build_baseline_vs_adaptive.py
"""

import os
import sys
import json
import pickle
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.doc2_graph.data.dataloader import Document2Graph
from src.data.doc2_graph.paths import FUNSD_TRAIN, FUNSD_TEST
from src.data.doc2_graph.utils import get_config
import dgl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def build_graphs_for_mode(edge_type, output_dir):
    """Build graphs (stage-1) for a given edge construction mode.

    Args:
        edge_type (str): 'knn' or 'adaptive'
        output_dir (Path): where to save .bin and stats
    """

    print(f"\n{'='*60}")
    print(f"Building graphs with edge_type = {edge_type}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}\n")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily modify config
    cfg = get_config('preprocessing')
    original_edge_type = cfg.GRAPHS.edge_type
    cfg.GRAPHS.edge_type = edge_type

    try:
        # Load training data (creates graphs with specified edge_type)
        print(f"Loading FUNSD TRAIN with edge_type={edge_type}...")
        train_data = Document2Graph(
            name=f'FUNSD TRAIN ({edge_type})',
            src_path=str(FUNSD_TRAIN),
            device=device,
            output_dir=output_dir
        )

        print(f"Loading FUNSD TEST with edge_type={edge_type}...")
        test_data = Document2Graph(
            name=f'FUNSD TEST ({edge_type})',
            src_path=str(FUNSD_TEST),
            device=device,
            output_dir=output_dir
        )

        # Split train into train/val
        from sklearn.model_selection import train_test_split
        train_graphs, val_graphs, _, _ = train_test_split(
            train_data.graphs,
            torch.ones(len(train_data.graphs), 1),
            test_size=0.1,
            random_state=42
        )

        # Collect statistics
        stats = {
            'edge_type': edge_type,
            'num_train_graphs': len(train_graphs),
            'num_val_graphs': len(val_graphs),
            'num_test_graphs': len(test_data.graphs),
            'config': {
                'candidate_pool_size': cfg.ADAPTIVE.candidate_pool_size if hasattr(cfg, 'ADAPTIVE') else None,
                'adaptive_k': cfg.ADAPTIVE.adaptive_k if hasattr(cfg, 'ADAPTIVE') else None,
                'score_threshold': cfg.ADAPTIVE.score_threshold if hasattr(cfg, 'ADAPTIVE') else None,
                'weights': dict(cfg.ADAPTIVE.weights) if hasattr(cfg, 'ADAPTIVE') else None,
            },
            'per_doc_stats': []
        }

        # Compute per-document statistics
        all_graphs = train_graphs + val_graphs + test_data.graphs
        for i, g in enumerate(all_graphs):
            n_nodes = g.number_of_nodes()
            n_edges = g.number_of_edges()
            avg_degree = n_edges / max(n_nodes, 1)
            stats['per_doc_stats'].append({
                'doc_id': i,
                'num_nodes': n_nodes,
                'num_edges': n_edges,
                'avg_degree': avg_degree
            })

        # Print aggregate stats
        all_edge_counts = [s['num_edges'] for s in stats['per_doc_stats']]
        print(f"\nAggregate statistics:")
        print(f"  Total documents: {len(all_graphs)}")
        print(f"  Total edges across all graphs: {sum(all_edge_counts)}")
        print(f"  Average edges per document: {sum(all_edge_counts) / len(all_graphs):.2f}")
        print(f"  Min edges: {min(all_edge_counts)}, Max edges: {max(all_edge_counts)}")

        # Save graphs
        bin_path_train = output_dir / 'train_contrastive.bin'
        bin_path_val = output_dir / 'val_contrastive.bin'
        bin_path_test = output_dir / 'test_contrastive.bin'

        print(f"\nSaving graphs to .bin files...")
        dgl.save_graphs(str(bin_path_train), train_graphs)
        dgl.save_graphs(str(bin_path_val), val_graphs)
        dgl.save_graphs(str(bin_path_test), test_data.graphs)

        # Save statistics
        stats_path = output_dir / 'graph_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved statistics to {stats_path}")

        return stats

    finally:
        # Restore original config
        cfg.GRAPHS.edge_type = original_edge_type


def main():
    project_root = Path(__file__).parent.parent
    output_base = project_root / 'graphs_stage1' / 'adaptive_experiment'

    results = {}

    # Build KNN baseline
    knn_stats = build_graphs_for_mode('knn', output_base / 'knn')
    results['knn'] = knn_stats

    # Build adaptive
    adaptive_stats = build_graphs_for_mode('adaptive', output_base / 'adaptive')
    results['adaptive'] = adaptive_stats

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for mode in ['knn', 'adaptive']:
        stats = results[mode]
        total_edges = sum(s['num_edges'] for s in stats['per_doc_stats'])
        avg_edges = total_edges / len(stats['per_doc_stats'])
        print(f"\n{mode.upper()}:")
        print(f"  Train graphs: {stats['num_train_graphs']}")
        print(f"  Val graphs: {stats['num_val_graphs']}")
        print(f"  Test graphs: {stats['num_test_graphs']}")
        print(f"  Total edges: {total_edges}")
        print(f"  Avg edges/doc: {avg_edges:.2f}")

    # Comparison
    knn_total = sum(s['num_edges'] for s in results['knn']['per_doc_stats'])
    adaptive_total = sum(s['num_edges'] for s in results['adaptive']['per_doc_stats'])
    print(f"\nAdaptive vs KNN edge ratio: {adaptive_total / knn_total:.3f}")

    print(f"\nGraphs saved to: {output_base}")


if __name__ == '__main__':
    main()
