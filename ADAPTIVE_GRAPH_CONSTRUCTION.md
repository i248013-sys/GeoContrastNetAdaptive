# Adaptive Graph Construction for GeoContrastNet

## Overview

This implementation adds an **adaptive graph construction** module to the existing GeoContrastNet codebase. Instead of using rigid K-nearest neighbor (KNN) spatial relationships, the adaptive approach scores candidate edges using multiple layout-aware signals (distance, row/column alignment, direction, overlap) and selects the top candidates based on combined scores.

## Key Idea

**Problem:** Fixed KNN graph construction is rigid. It connects each node to the K nearest neighbors by Euclidean distance. However, in real document layouts, the semantically correct relation is not always the nearest node. For example, a key-value pair may be aligned horizontally or vertically, even if far apart by distance.

**Solution:** Adaptive graph construction generates a larger candidate pool by distance, then scores each candidate using multiple signals (alignment, direction, overlap, etc.) to select edges that better capture document relations.

## Architecture

All changes are **isolated to the graph construction stage (Stage 1)**. The downstream model (Stage 2) remains unchanged — only the input graph topology differs.

### Files Modified

#### 1. `src/data/doc2_graph/configs/preprocessing.yaml`
Added `ADAPTIVE` configuration block:
```yaml
ADAPTIVE:
  candidate_pool_size: 15   # Generate 15 nearest candidates per node
  adaptive_k: 10             # Keep top 10 scoring candidates
  score_threshold: 0.0       # Drop edges below this score
  weights:                   # Weights for each signal
    distance: 1.0
    row_align: 0.6
    col_align: 0.6
    direction: 0.4
    overlap: 0.3
    text_sim: 0.0            # Disabled by default
  enabled_signals: [distance, row_align, col_align, direction, overlap, text_sim]
```

To switch between modes, set `GRAPHS.edge_type: knn` or `GRAPHS.edge_type: adaptive` in the same file.

#### 2. `src/data/doc2_graph/data/graph_builder.py`
Added three new methods to the `GraphBuilder` class:

- **`_discrete_position(rect_src, rect_dst)`**: Compute 9-bin directional relationship (up, down, left, right, etc.).
- **`_score_candidate_edge(box_i, box_j, ...)`**: Score an edge candidate using 6 signals:
  1. **Distance**: `1 - (euclidean_dist / max_pool_dist)`
  2. **Row alignment**: `1 - (|Δy_center| / page_h)` — captures horizontal relations
  3. **Column alignment**: `1 - (|Δx_center| / page_w)` — captures vertical relations
  4. **Direction bonus**: Favor right/below directions (natural reading order)
  5. **Overlap**: IoU or projected overlap between boxes
  6. **Text similarity**: (Disabled by default) token-bag cosine if texts provided

  Returns combined score: `Σ w_k * signal_k`

- **`__adaptive(size, bboxs, texts)`**: Main algorithm:
  1. Compute distances to all other nodes
  2. Get `candidate_pool_size` nearest as initial pool
  3. Score each candidate using `_score_candidate_edge`
  4. Keep top `adaptive_k` scoring candidates (subject to `score_threshold`)
  5. Emit bidirectional edges

**Key invariant:** `__knn` is unchanged. All adaptive logic is new code. You can switch between modes by simply changing the config.

#### 3. Three dispatch sites in `__fromFUNSD` (GT), `__fromFUNSD` (YOLO), and `__fromPAU`
Added `elif self.edge_type == 'adaptive':` branches to call `__adaptive` instead of `__knn`.

### Files Created

#### 1. `scripts/build_baseline_vs_adaptive.py`
**Purpose:** Build stage-1 graphs for both KNN and adaptive topologies.

**Usage:**
```bash
python scripts/build_baseline_vs_adaptive.py
```

**Outputs:**
- `graphs_stage1/adaptive_experiment/knn/{train,val,test}_contrastive.bin` — KNN graphs with stage-1 features
- `graphs_stage1/adaptive_experiment/adaptive/{train,val,test}_contrastive.bin` — Adaptive graphs with stage-1 features
- `graphs_stage1/adaptive_experiment/{knn,adaptive}/graph_stats.json` — Per-document graph statistics

Both are embedded with stage-1 contrastive embeddings (node features) for direct use in stage-2.

#### 2. `scripts/visualize_graphs.py`
**Purpose:** Visualize graph construction on sample FUNSD documents.

**Usage:**
```bash
python scripts/visualize_graphs.py
```

**Outputs (one set per sample document):**
- `outputs/graph_compare/<doc_name>_01_bboxes.png` — Bounding boxes only
- `outputs/graph_compare/<doc_name>_02_knn.png` — KNN graph edges (gray)
- `outputs/graph_compare/<doc_name>_03_adaptive.png` — Adaptive graph edges (green)
- `outputs/graph_compare/<doc_name>_04_ground_truth.png` — Ground-truth linking pairs (red)

Allows visual inspection of which edges are being constructed.

#### 3. `scripts/collect_results.py`
**Purpose:** Aggregate metrics from stage-2 training runs.

**Usage:**
```bash
python scripts/collect_results.py
```

**Outputs:**
- `outputs/comparison.csv` — Tabular results (KNN vs adaptive)
- `outputs/comparison.json` — Full metrics in JSON format

#### 4. `setups_stage2/FUNSD/run111_knn.yaml` and `run111_adaptive.yaml`
Clones of the original `run111.yaml` but pointing at different graph sources:
- `run111_knn.yaml` → `graphs_stage1/adaptive_experiment/knn/*.bin`
- `run111_adaptive.yaml` → `graphs_stage1/adaptive_experiment/adaptive/*.bin`

Same hyperparameters, same model architecture. Only input graph topology differs.

## How to Run (Google Colab)

### Prerequisites
Install dependencies:
```bash
pip install torch torchvision dgl scikit-learn pillow pyyaml matplotlib tqdm requests
python -m spacy download en_core_web_lg
```

### Step 1: Build both graph variants (Stage 1)
```bash
python scripts/build_baseline_vs_adaptive.py
```

**Output:**
Two sets of `.bin` files: `graphs_stage1/adaptive_experiment/{knn,adaptive}/`

**Compute:** ~5-10 min on CPU, ~1-2 min on GPU.

### Step 2: Visualize graphs (optional, helps debug)
```bash
python scripts/visualize_graphs.py
```

**Output:** 20 PNG images in `outputs/graph_compare/` showing KNN vs adaptive edges.

### Step 3a: Train Stage 2 on KNN baseline
```bash
python main.py --run-name run111_knn
```

**Compute:** ~2-4 hours on CPU (100 epochs), ~20 min on GPU.

**Output:** `runs/run111_knn/` with weights and metrics.

### Step 3b: Train Stage 2 on adaptive
```bash
python main.py --run-name run111_adaptive
```

**Compute:** Same as step 3a.

**Output:** `runs/run111_adaptive/` with weights and metrics.

### Step 4: Compare results
```bash
python scripts/collect_results.py
```

**Output:**
- `outputs/comparison.csv` — Side-by-side P/R/F1 for KNN vs adaptive.
- `outputs/comparison.json` — Full metrics.

## Interpreting Results

### Graph-level metrics (from `graph_stats.json`)
- **num_edges**: Total edges in graph. Adaptive typically has fewer edges than KNN because row/col alignment is more selective.
- **avg_degree**: Average edges per node. Should be close but not identical.

### Stage-2 metrics (from `comparison.csv`)
- **F1-macro / F1-micro**: Overall edge prediction performance.
- **Precision / Recall**: Per-class metrics.

**Expected outcome:** If layout alignment is informative, adaptive should match or beat KNN on F1 despite having fewer edges (more signal, less noise).

## Ablation Studies (Future Work)

The framework supports ablations by zeroing individual signal weights:

```yaml
# no_row_align
weights:
  distance: 1.0
  row_align: 0.0  # disabled
  col_align: 0.6
  direction: 0.4
  overlap: 0.3
  text_sim: 0.0
```

Create new YAMLs for each ablation (e.g., `run111_adaptive_no_row_align.yaml`), then rebuild graphs and train.

## Parameters and Knobs

### Tuning candidate pool size
Increase `candidate_pool_size` (default 15) to consider more neighbors:
- Pro: Better chance of finding true relations far from center.
- Con: More computation, potentially more noise.

### Adjusting signal weights
The `weights` dictionary scales the contribution of each signal. Increase a weight if that signal should matter more:
- `distance: 1.0` — Euclidean distance (strong baseline)
- `row_align: 0.6` — Horizontal alignment (key contribution for forms)
- `col_align: 0.6` — Vertical alignment (key contribution for tables)
- `direction: 0.4` — Directional bias (soft prior)
- `overlap: 0.3` — Bounding-box overlap (tiebreaker)
- `text_sim: 0.0` — Token similarity (disabled; enable to test text contribution)

### Enabling text similarity
To test whether text contributes:
1. Set `text_sim: 0.1` in config.
2. Rebuild graphs.
3. Compare stage-2 results.

## Thesis Narrative

**Contribution:** Adaptive graph construction is a lightweight improvement to the stage-1 graph builder. By incorporating layout-aware signals beyond Euclidean distance, the method captures document relations that KNN misses (e.g., far-apart but aligned key-value pairs).

**Claim:** The adaptive topology, used as input to the same downstream model, leads to better entity linking performance (F1 on `key_value` edges).

**Scope:** This is *not* a new model, nor a new loss, nor new node features. The contribution is purely in the *graph construction* stage — making the edge topology more semantically meaningful for the document domain.

## Citation / Reference

Original GeoContrastNet: [paper/repo link if applicable]

Adaptive modifications: This work extends GeoContrastNet's graph construction by replacing fixed KNN with learned-free scored candidate selection, improving layout-awareness for document understanding tasks.

---

## Troubleshooting

### Q: Why are adaptive edge counts different from KNN?
**A:** Adaptive uses selective scoring on a larger candidate pool. Edges that are far by distance but well-aligned by direction/row/column survive; many small-distance but poorly-aligned edges are dropped. This is intended.

### Q: Can I revert to KNN?
**A:** Yes. Set `GRAPHS.edge_type: knn` in `preprocessing.yaml` and rebuild graphs.

### Q: What if adaptive performs worse?
**A:** This is informative! It suggests layout alignment is less important than distance for this dataset, or the signal weights need tuning. Try:
1. Increasing `row_align`/`col_align` weights.
2. Increasing `candidate_pool_size` to 20-25.
3. Running ablations to see which signals help/hurt.

### Q: Can I train a new stage-1 model on adaptive graphs?
**A:** Yes, but it requires rerunning stage-1 contrastive training (expensive on CPU). See `src/training/gat_embeddings_contrastive.py`. For a first comparison, reusing the KNN-trained stage-1 weights is faster and still meaningful.
