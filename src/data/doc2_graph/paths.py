"""Patched paths for the vendored doc2graph subset.

Original used `dotenv` to read ROOT from a `root.env` file. We now derive
ROOT from this file's location instead, and align DATA with the existing
GeoContrastNet `src/data/datasets/` layout so FUNSD/PAU resolve correctly.
"""
from pathlib import Path

# This file lives at: GeoContrastNet/src/data/doc2_graph/paths.py
HERE = Path(__file__).resolve().parent           # .../doc2_graph
ROOT = HERE.parents[2]                           # GeoContrastNet/

# PROJECT TREE — DATA points at the existing datasets/ folder
DATA          = ROOT / 'src' / 'data' / 'datasets'
CONFIGS       = HERE / 'configs'
CFGM          = CONFIGS / 'models'
OUTPUTS       = ROOT / 'outputs'
RUNS          = OUTPUTS / 'runs'
RESULTS       = OUTPUTS / 'results'
IMGS          = OUTPUTS / 'images'
TRAIN_SAMPLES = OUTPUTS / 'train_samples'
TEST_SAMPLES  = OUTPUTS / 'test_samples'
TRAINING      = ROOT / 'src' / 'training'
MODELS        = HERE / 'models'
CHECKPOINTS   = MODELS / 'checkpoints'

# FUNSD
FUNSD_TRAIN = DATA / 'FUNSD' / 'training_data'
FUNSD_TEST  = DATA / 'FUNSD' / 'testing_data'

# PAU
PAU_TRAIN = DATA / 'PAU' / 'train'
PAU_TEST  = DATA / 'PAU' / 'test'
