from superneuroabm.ssn.SpikingConv2dt import Conv2dtNet
import pandas as pd
import unittest
import os
import numpy as np
import math
import matplotlib.pyplot as plt

class TestMNIST(unittest.TestCase):
    """
    Tests the model.reset() method's retain_parameters functionality with STDP learning.

    This test suite verifies that:
    - STDP learning actually changes synapse weights during simulation
    - When retain_parameters=True, learned weights are preserved after reset
    - When retain_parameters=False, weights are reset to their default values
    """

    def setUp(self):
        """Set up a fresh model for each test."""
        self.ConvNet = Conv2dtNet()
        self.ConvNet.model.setup(use_gpu=True)

        # fixed kernel list (removed the double assignment typo)
        self.Conv_Kernel_List = [
            [(3, 3), (2, 3), (4, 4), (4, 4), (3, 4), (4, 3)],
            [(3, 3), (2, 3), (3, 2), (3, 3)],
            [(3, 3), (2, 3), (3, 2), (4, 4)],
            [(3, 3), (2, 3), (3, 2), (4, 4)],
            [(3, 3), (2, 3), (3, 2), (4, 4)]
        ]

        self.ConvNet.ConstructionOfConvKernel(self.Conv_Kernel_List, 10)

    def log_kernel_weights(self, forward_pass_id=None):
        """Logs every learned synapse weight from each kernel in each layer into a pandas DataFrame."""
        rows = []
        for layer_idx, kernels in self.ConvNet.kernel_vals.items():  # layer → list of kernels
            for kernel_idx, (_, kernel_matrix) in enumerate(kernels):
                W, H = kernel_matrix.shape
                for i in range(W):
                    for j in range(H):
                        synapse = kernel_matrix[i][j]
                        hyper = self.ConvNet.model.get_agent_property_value(
                            id=synapse,
                            property_name="hyperparameters"
                        )
                        learned_weight = hyper[0]
                        rows.append({
                            "forward_pass": forward_pass_id,
                            "layer": layer_idx,
                            "kernel": kernel_idx,
                            "i": i,
                            "j": j,
                            "weight": learned_weight
                        })
        return pd.DataFrame(rows)

    def NMNIST(self, max_total_samples=1000, do_plot=True, passes_per_plot=25):
        """Run up to `max_total_samples` samples from NMNIST and plot every 25 forward passes."""
        root = "./superneuroabm/ssn/data/NMNIST/Train"
        assert os.path.isdir(root), f"NMNIST root not found: {root}"

        save_dir = "./ff_weight_maps"
        os.makedirs(save_dir, exist_ok=True)

        self.weight_history = pd.DataFrame()
        results = []
        pass_idx = 0
        processed = 0
        weight_snapshots = []

        # --- Load all digit subfolders (0–9) ---
        for digit in sorted(os.listdir(root)):
            digit_path = os.path.join(root, digit)
            if not os.path.isdir(digit_path):
                continue

            bin_files = sorted([f for f in os.listdir(digit_path) if f.endswith(".bin")])[:5]
            for bin_file in bin_files:
                dataset_path = os.path.join(digit_path, bin_file)
                print(f"[Forward pass {pass_idx}] → {dataset_path}")

                predicted_class = self.ConvNet.ForwardPass(dataset_path, 100)
                self.ConvNet.model.reset()

                results.append({
                    "true_class": int(digit),
                    "predicted_class": predicted_class,
                    "file_path": dataset_path
                })

                # --- Log kernel weights for this pass ---
                dfw = self.log_kernel_weights(forward_pass_id=pass_idx)
                self.weight_history = pd.concat([self.weight_history, dfw], ignore_index=True)

                # --- Collect snapshot for comparison plotting ---
                snapshot = {}
                for out_soma, syn_list in self.ConvNet.FF[2].items():
                    weights = []
                    for syn in syn_list:
                        hyper = self.ConvNet.model.get_agent_property_value(
                            id=syn, property_name="hyperparameters"
                        )
                        weights.append(hyper[0])
                    snapshot[out_soma] = weights
                weight_snapshots.append(snapshot)

                pass_idx += 1
                processed += 1

                # --- Every N passes: generate visual comparison ---
                if do_plot and pass_idx % passes_per_plot == 0:
                    print(f"Generating weight map comparison at pass {pass_idx}")

                    num_classes = len(self.ConvNet.FF[2])
                    num_passes = len(weight_snapshots)
                    all_w = [w for snap in weight_snapshots for ws in snap.values() for w in ws]
                    vmin, vmax = np.min(all_w), np.max(all_w)

                    fig_height = max(6, 1.5 * num_classes)
                    fig_width = max(8, 2.5 * num_passes)
                    fig, axs = plt.subplots(num_classes, num_passes, figsize=(fig_width, fig_height))

                    if num_classes == 1:
                        axs = np.expand_dims(axs, axis=0)
                    if num_passes == 1:
                        axs = np.expand_dims(axs, axis=1)

                    for i, (out_soma, _) in enumerate(self.ConvNet.FF[2].items()):
                        for j, snapshot in enumerate(weight_snapshots):
                            weights = np.array(snapshot[out_soma])
                            n_cols = math.ceil(math.sqrt(len(weights)))
                            n_rows = math.ceil(len(weights) / n_cols)
                            padded = np.full(n_rows * n_cols, np.nan)
                            padded[:len(weights)] = weights
                            grid = padded.reshape(n_rows, n_cols)

                            ax = axs[i, j]
                            im = ax.imshow(grid, vmin=vmin, vmax=vmax, cmap="viridis")
                            if i == 0:
                                ax.set_title(f"Pass {j + 1}")
                            if j == 0:
                                ax.set_ylabel(f"Neuron {i}")
                            ax.set_xticks([])
                            ax.set_yticks([])

                    # single colorbar on right
                    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
                    fig.colorbar(im, cax=cbar_ax)

                    plt.suptitle(f"Output Layer Weights Evolution (up to pass {pass_idx})", fontsize=14)
                    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
                    save_path = os.path.join(save_dir, f"output_layer_comparison_pass_{pass_idx}.png")
                    plt.savefig(save_path, dpi=250, bbox_inches="tight")
                    plt.close()
                    print(f"Saved comparison figure: {save_path}")

                # --- Stop after total limit ---
                if processed >= max_total_samples:
                    print(f"Reached {max_total_samples} total samples.")
                    return pd.DataFrame(results)

        print(f"Completed {processed} total samples.")
        return pd.DataFrame(results)

    def test_NMNIST(self):
        """Run the full NMNIST test."""
        df = self.NMNIST(max_total_samples=1000, do_plot=True, passes_per_plot=25)
        print(df.head())
