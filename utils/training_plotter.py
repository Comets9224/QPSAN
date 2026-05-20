# coding=utf-8
"""
Real-time Training Curve Plotter

Provides tools for:
- Real-time plotting of training curves (loss, accuracy)
- Plotting average curves from multiple runs
- Visualizing statistical bands (standard deviation)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt


class TrainingPlotter:
    """
    Real-time training curve plotter.

    Creates and updates training visualizations during training.
    Plots are saved as PNG images.
    """

    def __init__(self, save_dir, model_name="model", run_idx=None, plot_every=10, dpi=150):
        """
        Initialize the plotter.

        Parameters:
            save_dir: Directory to save plots
            model_name: Name of the model (for filename)
            run_idx: Run index (1-based) for multi-run experiments, None for single run
            plot_every: Update plots every N validation steps
            dpi: Image resolution
        """
        self.save_dir = save_dir
        self.model_name = model_name
        self.run_idx = run_idx
        self.plot_every = plot_every
        self.dpi = dpi

        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.steps = []

        # Create output directory
        os.makedirs(save_dir, exist_ok=True)

    def update(self, step, train_loss, val_loss=None, train_acc=None, val_acc=None):
        """
        Update training history and possibly plot.

        Parameters:
            step: Global training step
            train_loss: Training loss
            val_loss: Validation loss (optional)
            train_acc: Training accuracy (optional)
            val_acc: Validation accuracy (optional)
        """
        self.steps.append(step)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_acc is not None:
            self.train_accs.append(train_acc)
        if val_acc is not None:
            self.val_accs.append(val_acc)

        # Plot periodically
        if len(self.steps) % self.plot_every == 0 or step == 1:
            self.plot()

    def plot(self):
        """Plot current training curves."""
        if not self.steps:
            return

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left plot: Training and Validation Loss
        ax1 = axes[0]
        ax1.plot(self.steps, self.train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
        if self.val_losses:
            ax1.plot(self.steps, self.val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.model_name} - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Right plot: Accuracy
        ax2 = axes[1]
        if self.train_accs:
            ax2.plot(self.steps, self.train_accs, 'b-', label='Train Acc', linewidth=2, marker='o', markersize=3)
        if self.val_accs:
            ax2.plot(self.steps, self.val_accs, 'r-', label='Val Acc', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{self.model_name} - Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)

        plt.tight_layout()

        # Save plot
        step = self.steps[-1]
        # Filename includes run info
        run_suffix = f"_run{self.run_idx}" if self.run_idx is not None else ""
        save_path = os.path.join(self.save_dir, f'{self.model_name}_training_step{run_suffix}_{step:04d}.png')
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        # Also save as latest
        latest_path = os.path.join(self.save_dir, f'{self.model_name}_training_latest{run_suffix}.png')
        os.system(f'cp "{save_path}" "{latest_path}"')

    def save_final_plot(self):
        """Save final training curves with more detail."""
        if not self.steps:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Loss curves
        ax1 = axes[0, 0]
        ax1.plot(self.steps, self.train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=3)
        if self.val_losses:
            ax1.plot(self.steps, self.val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=3)
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'{self.model_name} - Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Accuracy curves
        ax2 = axes[0, 1]
        if self.train_accs:
            ax2.plot(self.steps, self.train_accs, 'b-', label='Train Acc', linewidth=2, marker='o', markersize=3)
        if self.val_accs:
            ax2.plot(self.steps, self.val_accs, 'r-', label='Val Acc', linewidth=2, marker='s', markersize=3)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Accuracy')
        ax2.set_title(f'{self.model_name} - Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(min(min(self.train_accs) if self.train_accs else 0.5,
                         min(self.val_accs) if self.val_accs else 0.5) * 0.9, 1.0)

        # 3. Loss zoom (last 50%)
        ax3 = axes[1, 0]
        if len(self.steps) > 1:
            zoom_idx = max(0, len(self.steps) // 2)
            zoom_steps = self.steps[zoom_idx:]
            zoom_train_losses = self.train_losses[zoom_idx:]
            ax3.plot(zoom_steps, zoom_train_losses, 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
            if self.val_losses:
                zoom_val_losses = self.val_losses[zoom_idx:]
                ax3.plot(zoom_steps, zoom_val_losses, 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
            ax3.set_xlabel('Training Steps (last 50%)')
            ax3.set_ylabel('Loss')
            ax3.set_title(f'{self.model_name} - Loss (Zoomed)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Accuracy zoom (last 50%)
        ax4 = axes[1, 1]
        if len(self.steps) > 1 and (self.train_accs or self.val_accs):
            zoom_idx = max(0, len(self.steps) // 2)
            zoom_steps = self.steps[zoom_idx:]
            if self.train_accs:
                zoom_train_accs = self.train_accs[zoom_idx:]
                ax4.plot(zoom_steps, zoom_train_accs, 'b-', label='Train Acc', linewidth=2, marker='o', markersize=4)
            if self.val_accs:
                zoom_val_accs = self.val_accs[zoom_idx:]
                ax4.plot(zoom_steps, zoom_val_accs, 'r-', label='Val Acc', linewidth=2, marker='s', markersize=4)
            ax4.set_xlabel('Training Steps (last 50%)')
            ax4.set_ylabel('Accuracy')
            ax4.set_title(f'{self.model_name} - Accuracy (Zoomed)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            y_min = min(min(self.train_accs[zoom_idx:]) if self.train_accs else 0.5,
                        min(self.val_accs[zoom_idx:]) if self.val_accs else 0.5)
            ax4.set_ylim(y_min * 0.95, 1.0)

        plt.tight_layout()
        # Filename includes run info
        run_suffix = f"_run{self.run_idx}" if self.run_idx is not None else ""
        save_path = os.path.join(self.save_dir, f'{self.model_name}_training_final{run_suffix}.png')
        plt.savefig(save_path, dpi=self.dpi * 2, bbox_inches='tight')
        plt.close()


def plot_average_curves(results, model_name, save_dir, dpi=300):
    """
    Plot average training curves from multiple runs with standard deviation bands.

    Parameters:
        results: List of run result dictionaries, each containing:
            - train_losses: List of training losses
            - val_losses: List of validation losses (optional)
            - val_accs: List of validation accuracies
            - steps: List of step numbers
        model_name: Name of the model
        save_dir: Directory to save plot
        dpi: Image resolution
    """
    if not results:
        return

    # Find max number of steps across all runs
    max_steps = max(len(r.get('steps', [])) for r in results if r.get('steps'))

    # Align data: for each step index, collect values from all runs that have that step
    avg_train_loss = []
    std_train_loss = []
    avg_val_loss = []
    std_val_loss = []
    avg_val_acc = []
    std_val_acc = []
    aligned_steps = []

    for step_idx in range(max_steps):
        step_losses = []
        step_val_losses = []
        step_accs = []

        for r in results:
            steps = r.get('steps', [])
            if step_idx < len(steps):
                # Get the step number for consistency
                if not aligned_steps:
                    aligned_steps.append(steps[step_idx])
                elif step_idx > 0:
                    # Ensure step numbers match across runs
                    if steps[step_idx] != aligned_steps[-1] and step_idx == len(aligned_steps):
                        aligned_steps.append(steps[step_idx])

                if step_idx < len(r.get('train_losses', [])):
                    step_losses.append(r['train_losses'][step_idx])
                if step_idx < len(r.get('val_losses', [])):
                    step_val_losses.append(r['val_losses'][step_idx])
                if step_idx < len(r.get('val_accs', [])):
                    step_accs.append(r['val_accs'][step_idx])

        if step_losses:
            avg_train_loss.append(np.mean(step_losses))
            std_train_loss.append(np.std(step_losses, ddof=1))
        else:
            avg_train_loss.append(np.nan)
            std_train_loss.append(np.nan)

        if step_val_losses:
            avg_val_loss.append(np.mean(step_val_losses))
            std_val_loss.append(np.std(step_val_losses, ddof=1))
        else:
            avg_val_loss.append(np.nan)
            std_val_loss.append(np.nan)

        if step_accs:
            avg_val_acc.append(np.mean(step_accs))
            std_val_acc.append(np.std(step_accs, ddof=1))
        else:
            avg_val_acc.append(np.nan)
            std_val_acc.append(np.nan)

    # Filter out NaN values
    valid_idx = [i for i, (a, s) in enumerate(zip(avg_train_loss, std_train_loss))
                 if not np.isnan(a) and not np.isnan(s)]

    if not valid_idx:
        print(f"Warning: No valid data to plot for {model_name}")
        return

    avg_train_loss = [avg_train_loss[i] for i in valid_idx]
    std_train_loss = [std_train_loss[i] for i in valid_idx]
    avg_val_loss = [avg_val_loss[i] for i in valid_idx if i < len(avg_val_loss) and not np.isnan(avg_val_loss[i])]
    std_val_loss = [std_val_loss[i] for i in valid_idx if i < len(std_val_loss) and not np.isnan(std_val_loss[i])]
    avg_val_acc = [avg_val_acc[i] for i in valid_idx if i < len(avg_val_acc) and not np.isnan(avg_val_acc[i])]
    std_val_acc = [std_val_acc[i] for i in valid_idx if i < len(std_val_acc) and not np.isnan(std_val_acc[i])]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{model_name} - Average Training Curves (n={len(results)} runs)', fontsize=14)

    steps = aligned_steps[:len(valid_idx)]

    # Plot 1: Training Loss with std band
    ax1 = axes[0]
    ax1.plot(steps, avg_train_loss, 'b-', linewidth=2, label='Mean')
    ax1.fill_between(steps,
                     np.array(avg_train_loss) - np.array(std_train_loss),
                     np.array(avg_train_loss) + np.array(std_train_loss),
                     alpha=0.3, color='blue', label='+/- 1 std')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss (Mean +/- Std)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy with std band
    ax2 = axes[1]
    if avg_val_acc:
        ax2.plot(steps[:len(avg_val_acc)], avg_val_acc, 'r-', linewidth=2, label='Mean')
        ax2.fill_between(steps[:len(std_val_acc)],
                         np.array(avg_val_acc) - np.array(std_val_acc),
                         np.array(avg_val_acc) + np.array(std_val_acc),
                         alpha=0.3, color='red', label='+/- 1 std')
        ax2.set_ylabel('Validation Accuracy')
        ax2.set_title('Validation Accuracy (Mean +/- Std)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0.5, 1.0)
    else:
        ax2.text(0.5, 0.5, 'No validation accuracy data',
                 ha='center', va='center', transform=ax2.transAxes)

    ax2.set_xlabel('Training Steps')

    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_name}_average_curves.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved average curves to: {save_path}")


def plot_comparison_curves(classical_results, quantum_results, save_dir, dpi=300):
    """
    Plot comparison between classical and quantum models.

    Parameters:
        classical_results: List of classical model run results
        quantum_results: List of quantum model run results
        save_dir: Directory to save plot
        dpi: Image resolution
    """
    # Compute averages
    classical_avg = _compute_run_averages(classical_results)
    quantum_avg = _compute_run_averages(quantum_results)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Classical vs Quantum ViT Comparison', fontsize=14)

    # Plot 1: Training Loss
    ax1 = axes[0]
    if classical_avg['steps'] and classical_avg['train_loss']:
        ax1.plot(classical_avg['steps'], classical_avg['train_loss'], 'b-',
                linewidth=2, label='Classical', marker='o', markersize=3)
        if classical_avg['train_loss_std']:
            ax1.fill_between(classical_avg['steps'],
                             np.array(classical_avg['train_loss']) - np.array(classical_avg['train_loss_std']),
                             np.array(classical_avg['train_loss']) + np.array(classical_avg['train_loss_std']),
                             alpha=0.2, color='blue')
    if quantum_avg['steps'] and quantum_avg['train_loss']:
        ax1.plot(quantum_avg['steps'], quantum_avg['train_loss'], 'r-',
                linewidth=2, label='Quantum', marker='s', markersize=3)
        if quantum_avg['train_loss_std']:
            ax1.fill_between(quantum_avg['steps'],
                             np.array(quantum_avg['train_loss']) - np.array(quantum_avg['train_loss_std']),
                             np.array(quantum_avg['train_loss']) + np.array(quantum_avg['train_loss_std']),
                             alpha=0.2, color='red')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Validation Accuracy
    ax2 = axes[1]
    if classical_avg['steps'] and classical_avg['val_acc']:
        ax2.plot(classical_avg['steps'], classical_avg['val_acc'], 'b-',
                linewidth=2, label='Classical', marker='o', markersize=3)
        if classical_avg['val_acc_std']:
            ax2.fill_between(classical_avg['steps'],
                             np.array(classical_avg['val_acc']) - np.array(classical_avg['val_acc_std']),
                             np.array(classical_avg['val_acc']) + np.array(classical_avg['val_acc_std']),
                             alpha=0.2, color='blue')
    if quantum_avg['steps'] and quantum_avg['val_acc']:
        ax2.plot(quantum_avg['steps'], quantum_avg['val_acc'], 'r-',
                linewidth=2, label='Quantum', marker='s', markersize=3)
        if quantum_avg['val_acc_std']:
            ax2.fill_between(quantum_avg['steps'],
                             np.array(quantum_avg['val_acc']) - np.array(quantum_avg['val_acc_std']),
                             np.array(quantum_avg['val_acc']) + np.array(quantum_avg['val_acc_std']),
                             alpha=0.2, color='red')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Validation Accuracy')
    ax2.set_title('Validation Accuracy Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0.5, 1.0)

    plt.tight_layout()

    # Save plot
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'classical_vs_quantum_comparison.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to: {save_path}")


def _compute_run_averages(results):
    """
    Helper function to compute average curves from multiple runs.

    Returns:
        Dictionary with 'steps', 'train_loss', 'train_loss_std', 'val_acc', 'val_acc_std'
    """
    if not results:
        return {'steps': [], 'train_loss': [], 'train_loss_std': [], 'val_acc': [], 'val_acc_std': []}

    max_steps = max(len(r.get('steps', [])) for r in results if r.get('steps'))

    avg_train_loss = []
    std_train_loss = []
    avg_val_acc = []
    std_val_acc = []
    aligned_steps = []

    for step_idx in range(max_steps):
        step_losses = []
        step_accs = []

        for r in results:
            steps = r.get('steps', [])
            if step_idx < len(steps):
                if not aligned_steps or (step_idx > 0 and step_idx < len(aligned_steps)):
                    if step_idx >= len(aligned_steps):
                        aligned_steps.append(steps[step_idx])
                    elif aligned_steps[step_idx] != steps[step_idx]:
                        aligned_steps[step_idx] = steps[step_idx]

                if step_idx < len(r.get('train_losses', [])):
                    step_losses.append(r['train_losses'][step_idx])
                if step_idx < len(r.get('val_accs', [])):
                    step_accs.append(r['val_accs'][step_idx])

        if step_losses:
            avg_train_loss.append(np.mean(step_losses))
            std_train_loss.append(np.std(step_losses, ddof=1))
        else:
            avg_train_loss.append(np.nan)
            std_train_loss.append(np.nan)

        if step_accs:
            avg_val_acc.append(np.mean(step_accs))
            std_val_acc.append(np.std(step_accs, ddof=1))
        else:
            avg_val_acc.append(np.nan)
            std_val_acc.append(np.nan)

    # Filter out NaN
    valid_idx = [i for i, (a, s) in enumerate(zip(avg_train_loss, std_train_loss))
                 if not np.isnan(a) and not np.isnan(s)]

    return {
        'steps': [aligned_steps[i] for i in valid_idx] if valid_idx else [],
        'train_loss': [avg_train_loss[i] for i in valid_idx] if valid_idx else [],
        'train_loss_std': [std_train_loss[i] for i in valid_idx] if valid_idx else [],
        'val_acc': [avg_val_acc[i] for i in valid_idx if i < len(avg_val_acc) and not np.isnan(avg_val_acc[i])],
        'val_acc_std': [std_val_acc[i] for i in valid_idx if i < len(std_val_acc) and not np.isnan(std_val_acc[i])]
    }


if __name__ == "__main__":
    # Test the plotter
    import tempfile

    # Test TrainingPlotter
    with tempfile.TemporaryDirectory() as tmpdir:
        plotter = TrainingPlotter(tmpdir, model_name="test_model", plot_every=5)

        # Simulate training
        for step in range(1, 51):
            loss = 1.0 * np.exp(-step / 20) + 0.1 * np.random.randn()
            acc = 1 - 0.5 * np.exp(-step / 15) + 0.05 * np.random.randn()
            val_acc = min(acc, 1 - 0.4 * np.exp(-step / 18) + 0.03 * np.random.randn())
            plotter.update(step, loss, val_loss=loss*1.1, train_acc=acc, val_acc=val_acc)

        plotter.save_final_plot()
        print(f"Test plots saved to {tmpdir}")

    # Test average curves
    with tempfile.TemporaryDirectory() as tmpdir:
        results = []
        for run in range(5):
            steps = list(range(1, 51))
            train_losses = [1.0 * np.exp(-s/20) + 0.1 * np.random.randn() for s in steps]
            val_accs = [1 - 0.5 * np.exp(-s/15) + 0.05 * np.random.randn() for s in steps]
            results.append({
                'steps': steps,
                'train_losses': train_losses,
                'val_accs': val_accs
            })

        plot_average_curves(results, "test_model", tmpdir)
        print(f"Average curves saved to {tmpdir}")
