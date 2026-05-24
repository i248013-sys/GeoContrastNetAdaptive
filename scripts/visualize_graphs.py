"""
Visualize KNN vs adaptive graph construction on sample FUNSD documents.
Saves side-by-side comparisons: original bboxes, KNN graph, adaptive graph, ground-truth linking.
"""

import os
import json
from pathlib import Path
from PIL import Image, ImageDraw
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.doc2_graph.data.graph_builder import GraphBuilder
from src.data.doc2_graph.paths import FUNSD_TRAIN
from src.data.doc2_graph.utils import get_config

def visualize_graphs(num_samples=5):
    """
    Load a few FUNSD documents and visualize both KNN and adaptive graph topologies.
    """

    output_dir = Path(__file__).parent.parent / 'outputs' / 'graph_compare'
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = get_config('preprocessing')
    gb = GraphBuilder()

    # Get annotation files
    annot_dir = FUNSD_TRAIN / 'adjusted_annotations'
    annot_files = sorted([f for f in os.listdir(annot_dir) if f.endswith('.json')])[:num_samples]

    print(f"Visualizing {len(annot_files)} samples...")

    for annot_file in annot_files:
        doc_name = annot_file.replace('.json', '')
        print(f"\nProcessing {doc_name}...")

        # Load annotation
        with open(annot_dir / annot_file, 'r') as f:
            data = json.load(f)
        form = data['form']

        # Extract boxes and texts
        boxs = [elem['box'] for elem in form]
        texts = [elem['text'] for elem in form]
        pair_labels = []
        for elem in form:
            pair_labels.extend(elem['linking'])

        # Convert to indices
        ids = [elem['id'] for elem in form]
        pair_indices = []
        for src_id, dst_id in pair_labels:
            try:
                src_idx = ids.index(src_id)
                dst_idx = ids.index(dst_id)
                pair_indices.append([src_idx, dst_idx])
            except ValueError:
                pass

        # Load image
        img_path = FUNSD_TRAIN / 'images' / f'{doc_name}.png'
        img = Image.open(img_path).convert('RGB')
        img_w, img_h = img.size

        # Build KNN graph
        cfg.GRAPHS.edge_type = 'knn'
        u_knn, v_knn = gb._GraphBuilder__knn((img_w, img_h), boxs)

        # Build adaptive graph
        cfg.GRAPHS.edge_type = 'adaptive'
        u_adaptive, v_adaptive = gb._GraphBuilder__adaptive((img_w, img_h), boxs, texts)

        center = lambda rect: ((rect[2]+rect[0])/2.0, (rect[3]+rect[1])/2.0)

        # Draw helper
        def draw_bboxes(img_draw, boxes, color='blue', width=2):
            for box in boxes:
                img_draw.rectangle(box, outline=color, width=width)

        def draw_edges(img_draw, u, v, boxes, color, width=1):
            for src, dst in zip(u, v):
                sc = center(boxes[src])
                ec = center(boxes[dst])
                img_draw.line((sc, ec), fill=color, width=width)

        # 1. Bboxes only
        img_bboxes = img.copy()
        draw = ImageDraw.Draw(img_bboxes)
        draw_bboxes(draw, boxs, color='blue', width=2)
        img_bboxes.save(output_dir / f'{doc_name}_01_bboxes.png')

        # 2. KNN graph
        img_knn = img.copy()
        draw = ImageDraw.Draw(img_knn)
        draw_bboxes(draw, boxs, color='blue', width=1)
        draw_edges(draw, u_knn, v_knn, boxs, color='gray', width=1)
        img_knn.save(output_dir / f'{doc_name}_02_knn.png')

        # 3. Adaptive graph
        img_adaptive = img.copy()
        draw = ImageDraw.Draw(img_adaptive)
        draw_bboxes(draw, boxs, color='blue', width=1)
        draw_edges(draw, u_adaptive, v_adaptive, boxs, color='green', width=1)
        img_adaptive.save(output_dir / f'{doc_name}_03_adaptive.png')

        # 4. Ground truth linking
        img_gt = img.copy()
        draw = ImageDraw.Draw(img_gt)
        draw_bboxes(draw, boxs, color='blue', width=1)
        for src, dst in pair_indices:
            sc = center(boxs[src])
            ec = center(boxs[dst])
            draw.line((sc, ec), fill='red', width=2)
        img_gt.save(output_dir / f'{doc_name}_04_ground_truth.png')

        print(f"  Saved 4 images to {output_dir}/{doc_name}_*.png")
        print(f"    KNN edges: {len(u_knn)}")
        print(f"    Adaptive edges: {len(u_adaptive)}")
        print(f"    Ground truth pairs: {len(pair_indices)}")

    print(f"\nVisualization complete. Saved to: {output_dir}")

if __name__ == '__main__':
    visualize_graphs(num_samples=5)
