"""Thin MLflow wrapper for GeoContrastNet training runs.

All public functions are no-ops if mlflow is not installed or if the env
var ``MLFLOW_DISABLE=1`` is set, so the training script keeps working
either way.

Usage:
    mlflow_logger.init(config, run_name, repo_root)
    mlflow_logger.log_epoch(epoch, train_loss=..., val_auc=...)
    mlflow_logger.log_test(test_f1=..., test_acc=...)
    mlflow_logger.log_artifact("runs/run111/images/Test Set - Edges.png")
    mlflow_logger.finish()
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

_enabled = False

try:
    import mlflow as _mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _mlflow = None
    _MLFLOW_AVAILABLE = False


def _flatten(prefix: str, d: dict, out: dict) -> None:
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            _flatten(key, v, out)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[key] = v
        elif isinstance(v, Path):
            out[key] = str(v)
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(str(x) for x in v)
        elif callable(v):
            out[key] = getattr(v, "__name__", repr(v))
        else:
            out[key] = str(v)


def init(config: dict, run_name: str, repo_root: Path) -> None:
    """Start an MLflow run, set tracking URI to <repo>/mlruns, log config."""
    global _enabled
    if not _MLFLOW_AVAILABLE or os.getenv("MLFLOW_DISABLE") == "1":
        return

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        tracking_uri = "file:" + (repo_root / "mlruns").as_posix()
    _mlflow.set_tracking_uri(tracking_uri)

    experiment_name = os.getenv("MLFLOW_EXPERIMENT", "GeoContrastNet")
    _mlflow.set_experiment(experiment_name)

    _mlflow.start_run(run_name=run_name)
    _enabled = True

    params: dict[str, Any] = {}
    _flatten("", config, params)
    for k, v in params.items():
        try:
            _mlflow.log_param(k, str(v)[:500])
        except Exception:
            pass

    try:
        _mlflow.set_tag("framework", "pytorch+dgl")
        _mlflow.set_tag("model_name", config.get("model_name", "?"))
        _mlflow.set_tag("train_method", config.get("train_method", "?"))
    except Exception:
        pass


def log_epoch(epoch: int, **metrics: Any) -> None:
    if not _enabled:
        return
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        if v is None:
            continue
        try:
            clean[k] = float(v)
        except (TypeError, ValueError):
            continue
    if clean:
        try:
            _mlflow.log_metrics(clean, step=epoch)
        except Exception:
            pass


def log_test(prefix: str = "test", **metrics: Any) -> None:
    if not _enabled:
        return
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        if v is None:
            continue
        try:
            clean[f"{prefix}_{k}"] = float(v)
        except (TypeError, ValueError):
            continue
    if clean:
        try:
            _mlflow.log_metrics(clean)
        except Exception:
            pass


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    if not _enabled:
        return
    p = Path(path)
    if not p.exists():
        return
    try:
        if p.is_dir():
            _mlflow.log_artifacts(str(p), artifact_path=artifact_path or p.name)
        else:
            _mlflow.log_artifact(str(p), artifact_path=artifact_path)
    except Exception:
        pass


def log_dict(obj: dict, filename: str) -> None:
    if not _enabled:
        return
    try:
        _mlflow.log_dict(obj, filename)
    except Exception:
        pass


def is_enabled() -> bool:
    return _enabled


def finish(status: str = "FINISHED") -> None:
    global _enabled
    if not _enabled:
        return
    try:
        _mlflow.end_run(status=status)
    finally:
        _enabled = False
