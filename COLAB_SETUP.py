"""
Google Colab Setup and Execution Guide for Adaptive Graph Construction Experiments

This script sets up the environment and runs the full pipeline on Google Colab.
Run this in Colab cells in order.

Prerequisites: Colab GPU instance (free tier has T4)
"""

# ============================================================================
# CELL 1: Mount Google Drive and clone repository from GitHub
# ============================================================================

# Mount Google Drive
from google.colab import drive
drive.mount('/gdrive')

import os
os.chdir('/content')

# Clone GeoContrastNet repo from GitHub
# IMPORTANT: Replace URL with your actual GitHub repo
GITHUB_REPO = "https://github.com/NilBiescas/GeoContrastNet"

print("=" * 70)
print("STEP 1: Clone GeoContrastNet from GitHub")
print("=" * 70)
print(f"Repository URL: {GITHUB_REPO}")
print("(If repo is private, use: https://<TOKEN>@github.com/<user>/GeoContrastNet.git)")
print()

import subprocess
result = subprocess.run(["git", "clone", GITHUB_REPO], cwd="/content")

# Verify clone
import time
time.sleep(1)
if os.path.exists('/content/GeoContrastNet'):
    print("\n[SUCCESS] GeoContrastNet cloned to /content/")
    os.chdir('/content/GeoContrastNet')
    print("\nRepository contents:")
    !ls -la | head -20
else:
    print("\n[ERROR] Clone failed. Possible reasons:")
    print("  1. Check GitHub repo URL is correct")
    print("  2. For private repos, use Personal Access Token: https://<TOKEN>@github.com/...")
    print("  3. Make sure git is installed")
    raise RuntimeError("Failed to clone repository")

# ============================================================================
# CELL 2: Install dependencies
# ============================================================================

!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -q dgl scikit-learn pillow pyyaml matplotlib tqdm requests pandas seaborn
!python -m spacy download en_core_web_lg

print("Dependencies installed!")

# Verify
import torch
import dgl
import numpy as np
print(f"PyTorch: {torch.__version__}")
print(f"DGL: {dgl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 3: Download FUNSD dataset (if not already present)
# ============================================================================

import os
data_dir = '/content/GeoContrastNet/src/data/datasets/FUNSD'
if not os.path.exists(data_dir):
    print("FUNSD not found, downloading...")
    os.chdir('/content/GeoContrastNet/src/data')
    !python download.py
    os.chdir('/content/GeoContrastNet')
    print("FUNSD download complete!")
else:
    print(f"FUNSD already exists at {data_dir}")

!ls -la src/data/datasets/FUNSD/ | head -20

# ============================================================================
# CELL 4: Build both graph variants (Stage 1)
# ============================================================================

import subprocess
import sys

print("\n" + "="*70)
print("STAGE 1: Building KNN and Adaptive graphs")
print("="*70 + "\n")

result = subprocess.run([sys.executable, 'scripts/build_baseline_vs_adaptive.py'],
                       cwd='/content/GeoContrastNet')

if result.returncode == 0:
    print("\n[SUCCESS] Graph building complete!")
    !ls -la graphs_stage1/adaptive_experiment/
else:
    print("\n[ERROR] Graph building failed!")
    sys.exit(1)

# ============================================================================
# CELL 5: Visualize graphs (optional, helps debug)
# ============================================================================

print("\n" + "="*70)
print("Visualizing KNN vs Adaptive graphs")
print("="*70 + "\n")

result = subprocess.run([sys.executable, 'scripts/visualize_graphs.py'],
                       cwd='/content/GeoContrastNet')

print("\n[SUCCESS] Visualizations saved to outputs/graph_compare/")
!ls -la outputs/graph_compare/ | head -20

# ============================================================================
# CELL 6: Train Stage 2 on KNN baseline
# ============================================================================

print("\n" + "="*70)
print("STAGE 2a: Training on KNN baseline graphs (100 epochs)")
print("Estimated time: ~20-30 min on GPU, ~2-4 hours on CPU")
print("="*70 + "\n")

result = subprocess.run([sys.executable, 'main.py', '--run-name', 'run111_knn'],
                       cwd='/content/GeoContrastNet')

if result.returncode == 0:
    print("\n[SUCCESS] KNN training complete!")
    !ls -la runs/run111_knn/
else:
    print("\n[ERROR] KNN training failed!")

# ============================================================================
# CELL 7: Train Stage 2 on adaptive graphs
# ============================================================================

print("\n" + "="*70)
print("STAGE 2b: Training on adaptive graphs (100 epochs)")
print("Estimated time: ~20-30 min on GPU, ~2-4 hours on CPU")
print("="*70 + "\n")

result = subprocess.run([sys.executable, 'main.py', '--run-name', 'run111_adaptive'],
                       cwd='/content/GeoContrastNet')

if result.returncode == 0:
    print("\n[SUCCESS] Adaptive training complete!")
    !ls -la runs/run111_adaptive/
else:
    print("\n[ERROR] Adaptive training failed!")

# ============================================================================
# CELL 8: Collect and compare results
# ============================================================================

print("\n" + "="*70)
print("Comparing KNN vs Adaptive results")
print("="*70 + "\n")

result = subprocess.run([sys.executable, 'scripts/collect_results.py'],
                       cwd='/content/GeoContrastNet')

print("\n✓ Results aggregated!")

# Display comparison
import pandas as pd
from pathlib import Path

comparison_csv = Path('/content/GeoContrastNet/outputs/comparison.csv')
if comparison_csv.exists():
    df = pd.read_csv(comparison_csv)
    print("\n" + "="*70)
    print("COMPARISON RESULTS")
    print("="*70)
    print(df.to_string())

# ============================================================================
# CELL 9: Save results back to Google Drive (thesis2 folder)
# ============================================================================

print("\n" + "="*70)
print("Saving results to Google Drive")
print("="*70)

import shutil
import os

# Ensure thesis2 folder exists
!mkdir -p /gdrive/'My Drive'/thesis2

# Copy the full GeoContrastNet folder with results back to Drive
print("\nCopying runs/ and outputs/ to Google Drive...")
!cp -r /content/GeoContrastNet/runs "/gdrive/My Drive/thesis2/" 2>/dev/null && echo "OK: Copied runs/" || echo "WARN: Could not copy runs"
!cp -r /content/GeoContrastNet/outputs "/gdrive/My Drive/thesis2/" 2>/dev/null && echo "OK: Copied outputs/" || echo "WARN: Could not copy outputs"
!cp -r /content/GeoContrastNet/graphs_stage1 "/gdrive/My Drive/thesis2/" 2>/dev/null && echo "OK: Copied graphs_stage1/" || echo "WARN: Could not copy graphs"

# Also create a zip for easy download
print("\nCreating backup zip file...")
!cd /content/GeoContrastNet && zip -r "/gdrive/My Drive/thesis2/GeoContrastNet_results_backup.zip" runs outputs graphs_stage1 2>/dev/null

print("\nAll results saved to /gdrive/My Drive/thesis2/")
print("Files available:")
!ls -lh "/gdrive/My Drive/thesis2/" | head -20

# ============================================================================
# Final Summary
# ============================================================================

print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)
print("\nAll results saved to: /gdrive/My Drive/thesis2/")
print("\nFiles generated:")
print("  - runs/run111_knn/ (baseline results)")
print("  - runs/run111_adaptive/ (adaptive results)")
print("  - outputs/comparison.csv (metrics summary)")
print("  - outputs/comparison.json (detailed metrics)")
print("  - outputs/graph_compare/*.png (visualizations)")
print("  - graphs_stage1/adaptive_experiment/ (graph binaries)")
print("\nNext steps:")
print("  1. Download results from Google Drive")
print("  2. Review comparison.csv for metrics")
print("  3. Check visualizations in graph_compare/")
print("  4. Document findings for your thesis")
print("\nFor full documentation, see: ADAPTIVE_GRAPH_CONSTRUCTION.md")
