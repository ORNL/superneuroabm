import os
import unittest
import argparse
import numpy as np
from superneuroabm.ssn.SpikingConv2dt import Conv2dtNet

# -------------------- Command-line arguments --------------------
parser = argparse.ArgumentParser(description="Conv2dtNet Output Layer Weight Tests")
parser.add_argument("--data_path", required=True, help="Path to NMNIST Train directory")
parser.add_argument("--output_path", required=True, help="Directory to save output plots")
parser.add_argument("--resume_from_run", type=int, default=0,
                    help="Run number to resume from (0 means start from scratch)")
args, remaining_argv = parser.parse_known_args()

# Make unittest ignore our custom CLI args
import sys
sys.argv = [sys.argv[0]] + remaining_argv


# -------------------- Helper: Save current run to file --------------------
def save_run_number(run_number, output_dir):
    path = os.path.join(output_dir, "most_recent_run.txt")
    with open(path, "w") as f:
        f.write(str(run_number))


# -------------------- Test Class --------------------
class TestOutputLayerWeights(unittest.TestCase):

    def setUp(self):
        self.ConvNet = Conv2dtNet()
        self.ConvNet.model.setup(use_gpu=True)
        print("GPU setup complete")

        Conv_Kernel_List = [
            [(3, 3, 2)],
            [(5, 5, 1)]
        ]

        self.ConvNet.NetworkConstruction(
            Conv_Kernel_List=Conv_Kernel_List,
            output_classes=30,
            Input_W=28,
            Input_H=28,
        )

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

        # ---- Training schedule ----
        cycles = 3          # 3 × (100 passes × 10 digits) = 3000 total
        passes_per_cycle = 100

        total_runs = cycles * passes_per_cycle * len(digit_dirs)

        print(f"\n=== TOTAL RUNS = {total_runs} ===")
        print(f"Resuming from run {args.resume_from_run}\n")

        current_run = args.resume_from_run

        # Iterate through cycles
        for cycle in range(1, cycles + 1):
            print(f"\n=== Starting Cycle {cycle}/3 ===")

            for digit in digit_dirs:
                digit_path = os.path.join(root, digit)
                if not os.path.isdir(digit_path):
                    continue

                bin_files = [f for f in os.listdir(digit_path) if f.endswith(".bin")]
                if not bin_files:
                    continue

                dataset_path = os.path.join(digit_path, bin_files[0])
                print(f"\n--- Cycle {cycle} | Digit {digit} | Using file: {bin_files[0]} ---")

                # 100 forward passes per digit per cycle
                for p in range(1, passes_per_cycle + 1):
                    current_run += 1

                    # Skip until we reach resume point
                    if current_run <= args.resume_from_run:
                        continue

                    print(f"[Run {current_run}/{total_runs}] Cycle {cycle} | Digit {digit} | Pass {p}/100")

                    _ = self.ConvNet.ForwardPass(
                        dataset_path,
                        Total_Sim_Time=Total_Sim_Time,
                    )

                    # ---- Save most recent run number ----
                    save_run_number(current_run, output_dir)

                    # ---- Save plots and JSON every 25 passes ----
                    if p % 25 == 0 or p == 1:
                        plots_dir = os.path.join(output_dir, "plots")
                        json_dir = os.path.join(output_dir, "json")
                        os.makedirs(plots_dir, exist_ok=True)
                        os.makedirs(json_dir, exist_ok=True)

                        plot_path = os.path.join(
                            plots_dir, f"cycle_{cycle}_digit_{digit}_pass_{p}.png"
                        )
                        self.ConvNet.plot_all_output_neurons_single(plot_path)
                        print(f"Saved plot: {plot_path}")

                        json_path = os.path.join(
                            json_dir, f"cycle_{cycle}_digit_{digit}_pass_{p}.json"
                        )
                        self.ConvNet.extract_full_network(json_path)
                        print(f"Saved JSON: {json_path}")

                    # ---- Reset internal neuron states but keep weights ----
                    self.ConvNet.model.reset()

        print("\n=== Completed all cycles ===")


if __name__ == '__main__':
    unittest.main()
