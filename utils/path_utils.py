# coding=utf-8
"""
Path Management Utility Module

Centralized generation and organization of all project output paths.
"""
import os
from datetime import datetime


def get_timestamp():
    """Get current timestamp, format: YYYY-MM-DD-HHMMSS"""
    return datetime.now().strftime("%Y-%m-%d-%H%M%S")


def get_output_paths(dataset_name, model_name, num_runs=1):
    """
    Get standardized output path structure.

    Args:
        dataset_name: Dataset name (e.g., 'FashionMNIST', 'CIFAR10')
        model_name: Model name (e.g., 'classic_tiny_1', 'quantum_tiny_1')
        num_runs: Number of runs

    Returns:
        dict: Dictionary containing various paths
    """
    # Check for unified timestamp from parallel_launcher (passed via environment variable)
    shared_timestamp = os.environ.get('PARALLEL_LAUNCHER_TIMESTAMP')

    # Determine which timestamp to use
    timestamp = shared_timestamp if shared_timestamp else get_timestamp()

    if shared_timestamp:
        # parallel_launcher mode: output/{dataset}/{timestamp}/{model_name}/
        # All runs share the same timestamp folder
        base_output_dir = os.path.join("output", dataset_name, timestamp, model_name)
        base_checkpoint_dir = os.path.join("checkpoint", dataset_name, timestamp, model_name)
        logs_dir = os.path.join("logs", dataset_name, timestamp, model_name)
    else:
        # Standalone serial mode: output/{dataset}/{timestamp}/
        # Auto-generated timestamp, no model_name level
        base_output_dir = os.path.join("output", dataset_name, timestamp)
        base_checkpoint_dir = os.path.join("checkpoint", dataset_name, timestamp)
        logs_dir = os.path.join("logs", dataset_name, timestamp)

    paths = {
        "output_dir": base_output_dir,
        "checkpoint_base": base_checkpoint_dir,
        "logs_dir": logs_dir,
        "timestamp": timestamp,
    }

    # Checkpoint paths for multiple runs
    if num_runs > 1:
        paths["run_checkpoint_dirs"] = []
        for run_idx in range(1, num_runs + 1):
            run_dir = os.path.join(base_checkpoint_dir, f"run_{run_idx}")
            paths["run_checkpoint_dirs"].append(run_dir)

    return paths


def create_output_paths(paths):
    """Create all necessary output directories."""
    dirs_to_create = [
        paths["output_dir"],
        paths["checkpoint_base"],
        paths["logs_dir"],
    ]

    # Checkpoint directories for multiple runs
    if "run_checkpoint_dirs" in paths:
        dirs_to_create.extend(paths["run_checkpoint_dirs"])

    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)


def get_checkpoint_path(paths, run_idx=None, model_name="model", ext="bin"):
    """
    Get checkpoint file path.

    Args:
        paths: Path dictionary returned by get_output_paths()
        run_idx: Run index (1-based), None for single run
        model_name: Model name
        ext: File extension

    Returns:
        Checkpoint file path
    """
    if run_idx is not None and "run_checkpoint_dirs" in paths:
        # Multiple runs
        checkpoint_dir = paths["run_checkpoint_dirs"][run_idx - 1]
    else:
        # Single run
        checkpoint_dir = paths["checkpoint_base"]

    return os.path.join(checkpoint_dir, f"{model_name}_checkpoint.{ext}")


def get_tensorboard_log_path(paths):
    """Get TensorBoard log path."""
    return paths["logs_dir"]


def get_results_path(paths, model_name):
    """Get JSON results file path."""
    return os.path.join(paths["output_dir"], f"{model_name}_results.json")


def get_plot_path(paths, filename):
    """Get plot file path."""
    return os.path.join(paths["output_dir"], filename)


def get_latest_timestamp(dataset_name):
    """Get the latest timestamp directory for a given dataset."""
    base_dir = os.path.join("output", dataset_name)
    if not os.path.exists(base_dir):
        return None
    timestamps = [d for d in os.listdir(base_dir)
                  if os.path.isdir(os.path.join(base_dir, d))]
    if not timestamps:
        return None
    return sorted(timestamps)[-1]
