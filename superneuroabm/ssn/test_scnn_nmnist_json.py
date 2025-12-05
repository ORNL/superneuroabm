import os
import unittest
import argparse
import numpy as np
from SpikingConv2dt import Conv2dtNet
from Conv2dtLoader import load_network_from_json


# -------------------- Command-line arguments --------------------
parser = argparse.ArgumentParser(description="Conv2dtNet Output Layer Weight Tests")
parser.add_argument("--data_path", required=True, help="Path to NMNIST Train directory")
parser.add_argument("--output_path", required=True, help="Directory to save output plots")
parser.add_argument("--model_path", required=True, help="Path for saved model Path")
args, remaining_argv = parser.parse_known_args()

# Make unittest ignore our custom CLI args
import sys
sys.argv = [sys.argv[0]] + remaining_argv


class TestOutputLayerWeights(unittest.TestCase):
    """
    Integration-style test that:
    - builds a Conv2dtNet
    - runs 100 forward passes per digit
    - collects FF weights
    - saves plot every 25 passes
    """

    def setUp(self):
        Model, soma_map, syn_map =load_network_from_json(f'{args.model_path}')
        self.ConvNet = Model

        print("Network constructed")
        print("Number of somas:", len(self.ConvNet.model._soma_ids))
        print("Number of synapses:", len(self.ConvNet.model._synapse_ids))

    def test_output_layer_weight_maps(self):
        root = args.data_path
        self.assertTrue(os.path.isdir(root), f"NMNIST directory not found: {root}")

        output_dir = args.output_path
        os.makedirs(output_dir, exist_ok=True)

        digit_dirs = sorted(os.listdir(root))

        Total_Sim_Time = 100
        passes_per_digit = 100
        save_every = 25

        print("\n=== Running 100 forward passes per digit ===")

        for digit in digit_dirs:
            digit_path = os.path.join(root, digit)
            if not os.path.isdir(digit_path):
                continue

            bin_files = [f for f in os.listdir(digit_path) if f.endswith(".bin")]
            if not bin_files:
                continue

            dataset_path = os.path.join(digit_path, bin_files[0])
            print(f"\n--- Digit {digit} | Using example file: {bin_files[0]} ---")

            for p in range(1, passes_per_digit + 1):
                print(f"Digit {digit} → Pass {p}/100")

                _ = self.ConvNet.ForwardPass(
                    dataset_path,
                    Total_Sim_Time=Total_Sim_Time,
                )

                # ---- Every 25 passes: save full output layer maps + network JSON ----
                if p % save_every == 0 or p == 1:

                    # -------- Create subdirectories --------
                    plots_dir = os.path.join(output_dir, "plots")
                    json_dir = os.path.join(output_dir, "json")
                    os.makedirs(plots_dir, exist_ok=True)
                    os.makedirs(json_dir, exist_ok=True)

                    # -------- Save full FF weight map (using existing function) --------
                    plot_path = os.path.join(
                        plots_dir, f"digit_{digit}_pass_{p}.png"
                    )
                    self.ConvNet.plot_all_output_neurons_single(plot_path)
                    print(f"Saved plot: {plot_path}")

                    # -------- Save full network JSON snapshot (using existing function) --------
                    json_path = os.path.join(
                        json_dir, f"digit_{digit}_pass_{p}.json"
                    )
                    self.ConvNet.extract_full_network(json_path)
                    print(f"Saved JSON: {json_path}")

                # ---- Reset neuron internal states but keep learned weights ----
                self.ConvNet.model.reset()

        print("\n=== Completed all digits ===")

if __name__ == '__main__':
    unittest.main()