"""Experiment tracking integration for VGC AI.

This module provides MLflow integration for tracking
training experiments, hyperparameters, and model artifacts.
"""

from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
import json
from datetime import datetime
import os
from loguru import logger

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logger.warning("MLflow not available. Falling back to local logging.")


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking."""
    
    experiment_name: str = "vgc-ai"
    tracking_uri: str = "mlruns"  # Local directory or MLflow server URI
    artifact_location: Optional[str] = None
    tags: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}


class ExperimentTracker:
    """Unified experiment tracking interface.
    
    Supports MLflow when available, falls back to local JSON logging.
    
    Example:
        tracker = ExperimentTracker(
            experiment_name="vgc-imitation",
        )
        
        with tracker.run("baseline_v1"):
            tracker.log_params({"lr": 0.001, "batch_size": 256})
            
            for epoch in range(epochs):
                loss = train_epoch()
                tracker.log_metrics({"loss": loss}, step=epoch)
            
            tracker.log_model(model, "final_model")
    """
    
    def __init__(
        self,
        experiment_name: str = "vgc-ai",
        tracking_uri: Optional[str] = None,
        save_dir: Optional[Path] = None,
    ):
        """Initialize tracker.
        
        Args:
            experiment_name: Name of experiment
            tracking_uri: MLflow tracking URI
            save_dir: Fallback directory for local logging
        """
        self.experiment_name = experiment_name
        self.save_dir = save_dir or Path("data/experiments")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_mlflow = MLFLOW_AVAILABLE
        self._run_id: Optional[str] = None
        self._run_name: Optional[str] = None
        self._local_log: Dict[str, Any] = {}
        
        if self.use_mlflow:
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
            
            # Create or get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            
            logger.info(f"MLflow tracking enabled: {experiment_name}")
    
    @contextmanager
    def run(
        self,
        run_name: str,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ):
        """Context manager for a training run.
        
        Args:
            run_name: Name for this run
            tags: Optional tags
            description: Optional description
        """
        self._run_name = run_name
        
        if self.use_mlflow:
            with mlflow.start_run(run_name=run_name):
                self._run_id = mlflow.active_run().info.run_id
                
                if tags:
                    mlflow.set_tags(tags)
                if description:
                    mlflow.set_tag("description", description)
                
                try:
                    yield self
                finally:
                    self._run_id = None
        else:
            # Local tracking
            self._local_log = {
                "run_name": run_name,
                "start_time": datetime.utcnow().isoformat(),
                "tags": tags or {},
                "description": description,
                "params": {},
                "metrics": {},
                "artifacts": [],
            }
            
            try:
                yield self
            finally:
                self._local_log["end_time"] = datetime.utcnow().isoformat()
                self._save_local_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters.
        
        Args:
            params: Dict of parameter names to values
        """
        if self.use_mlflow and mlflow.active_run():
            # MLflow requires string values
            mlflow.log_params({k: str(v) for k, v in params.items()})
        else:
            self._local_log["params"].update(params)
        
        logger.debug(f"Logged params: {list(params.keys())}")
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ):
        """Log metrics.
        
        Args:
            metrics: Dict of metric names to values
            step: Optional step number
        """
        if self.use_mlflow and mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
        else:
            for name, value in metrics.items():
                if name not in self._local_log["metrics"]:
                    self._local_log["metrics"][name] = []
                self._local_log["metrics"][name].append({
                    "value": value,
                    "step": step,
                })
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
    ):
        """Log a single metric.
        
        Args:
            name: Metric name
            value: Metric value
            step: Optional step number
        """
        self.log_metrics({name: value}, step=step)
    
    def log_artifact(self, path: Union[str, Path], artifact_path: Optional[str] = None):
        """Log an artifact file.
        
        Args:
            path: Local path to artifact
            artifact_path: Optional path within artifact store
        """
        path = Path(path)
        
        if self.use_mlflow and mlflow.active_run():
            mlflow.log_artifact(str(path), artifact_path)
        else:
            self._local_log["artifacts"].append({
                "local_path": str(path),
                "artifact_path": artifact_path,
            })
    
    def log_model(
        self,
        model: Any,
        artifact_path: str,
        signature: Optional[Any] = None,
    ):
        """Log a PyTorch model.
        
        Args:
            model: PyTorch model
            artifact_path: Path for model artifacts
            signature: Optional model signature
        """
        import torch
        
        if self.use_mlflow and mlflow.active_run():
            try:
                mlflow.pytorch.log_model(model, artifact_path)
            except Exception as e:
                logger.warning(f"Failed to log model to MLflow: {e}")
                # Fallback to artifact
                local_path = self.save_dir / f"{self._run_name}_model.pt"
                torch.save(model.state_dict(), local_path)
                self.log_artifact(local_path, artifact_path)
        else:
            local_path = self.save_dir / f"{self._run_name}_model.pt"
            torch.save(model.state_dict(), local_path)
            self._local_log["artifacts"].append({
                "type": "model",
                "path": str(local_path),
            })
    
    def log_figure(self, figure: Any, artifact_path: str):
        """Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            artifact_path: Path for figure
        """
        if self.use_mlflow and mlflow.active_run():
            mlflow.log_figure(figure, artifact_path)
        else:
            local_path = self.save_dir / artifact_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            figure.savefig(local_path)
            self._local_log["artifacts"].append({
                "type": "figure",
                "path": str(local_path),
            })
    
    def log_dict(self, dictionary: Dict[str, Any], artifact_path: str):
        """Log a dictionary as JSON artifact.
        
        Args:
            dictionary: Dict to log
            artifact_path: Path for JSON file
        """
        if self.use_mlflow and mlflow.active_run():
            mlflow.log_dict(dictionary, artifact_path)
        else:
            local_path = self.save_dir / artifact_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            with open(local_path, "w") as f:
                json.dump(dictionary, f, indent=2)
            self._local_log["artifacts"].append({
                "type": "dict",
                "path": str(local_path),
            })
    
    def set_tag(self, key: str, value: str):
        """Set a tag on the current run.
        
        Args:
            key: Tag key
            value: Tag value
        """
        if self.use_mlflow and mlflow.active_run():
            mlflow.set_tag(key, value)
        else:
            self._local_log["tags"][key] = value
    
    def _save_local_run(self):
        """Save local run log to file."""
        filename = f"{self._run_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.save_dir / "runs" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w") as f:
            json.dump(self._local_log, f, indent=2, default=str)
        
        logger.info(f"Saved local run log to {path}")
    
    @staticmethod
    def get_best_run(
        experiment_name: str,
        metric: str = "val_accuracy",
        ascending: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """Get best run from an experiment.
        
        Args:
            experiment_name: Experiment name
            metric: Metric to optimize
            ascending: Whether lower is better
            
        Returns:
            Best run info or None
        """
        if not MLFLOW_AVAILABLE:
            logger.warning("MLflow not available")
            return None
        
        client = MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            return None
        
        order = "ASC" if ascending else "DESC"
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )
        
        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "artifacts": run.info.artifact_uri,
            }
        
        return None


# Convenience functions

def start_run(
    run_name: str,
    experiment_name: str = "vgc-ai",
    **kwargs,
) -> ExperimentTracker:
    """Start a new tracking run.
    
    Args:
        run_name: Name for the run
        experiment_name: Experiment name
        **kwargs: Additional arguments
        
    Returns:
        ExperimentTracker instance
    """
    tracker = ExperimentTracker(experiment_name=experiment_name)
    return tracker


def log_training_run(
    run_name: str,
    config: Dict[str, Any],
    train_fn: callable,
    experiment_name: str = "vgc-ai",
) -> Dict[str, Any]:
    """Convenience wrapper for logging a training run.
    
    Args:
        run_name: Name for the run
        config: Training configuration
        train_fn: Training function (receives tracker, returns metrics)
        experiment_name: Experiment name
        
    Returns:
        Final metrics from training
    """
    tracker = ExperimentTracker(experiment_name=experiment_name)
    
    with tracker.run(run_name):
        tracker.log_params(config)
        
        try:
            result = train_fn(tracker)
            
            if isinstance(result, dict):
                tracker.log_metrics({f"final_{k}": v for k, v in result.items()})
            
            return result
            
        except Exception as e:
            tracker.set_tag("status", "failed")
            tracker.set_tag("error", str(e))
            raise

