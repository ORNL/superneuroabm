import os
import unittest
import argparse
import sys
from SpikingConv2dt import Conv2dtNet
from Conv2dtLoader import load_network_from_json


# -------------------- Command-line arguments --------------------
parser = argparse.ArgumentParser(description="Conv2dtNet Output Layer Weight Tests")
parser.add_argument("--data_path", required=True, help="Path to NMNIST Train directory")
parser.add_argument("--output_path", required=True, help="Directory to save output plots + JSON")
parser.add_argument("--model_path", required=True, help="Path to saved model JSON")
parser.add_argument("--resume_from_run", type=int, default=0,
                    help="Global run number to resume from (0 means start from scratch)")

args, remaining_argv = parser.parse_known_args()

# Make unittest ignore our custom arguments
sys.argv = [sys.argv[0]] + remaining_argv


# -------------------- Save run number --------------------
def save_run_number(run_number, output_dir):
    path = os.path.join(output_dir, "most_recent_run.txt")
    with open(path, "w") as f:
        f.write(str(run_number))


# -------------------- Main Test Class --------------------
class TestOutputLayerWeights(unittest.TestCase):

    def setUp(self):
        Model, soma_map, syn_map = load_network_from_json(args.model_path)
        self.ConvNet = Model

        print("Loaded model from JSON")
        print("Number of somas:", len(self.ConvNet.model._soma_ids))
        print("Number of synapses:", len(self.ConvNet.model._synapse_ids))

    def test_output_layer_weight_maps(self):

        root = args.data_path
        self.assertTrue(os.path.isdir(root), f"NMNIST directory not found: {root}")

        output_dir = args.output_path
        os.makedirs(output_dir, exist_ok=True)

        digit_dirs = sorted(os.listdir(root))  # expect ["0", "1", ..., "9"]

        # -------------------- Training parameters --------------------
        Total_Sim_Time = 100
        passes_per_digit = 100
        digits = 10
        cycles = 3

        runs_per_cycle = passes_per_digit * digits       # = 1000
        total_runs = runs_per_cycle * cycles             # = 3000

        save_every = 25

        resume = args.resume_from_run
        current_run = resume

        # -------------------- Decode resume position --------------------
        cycle_start = resume // runs_per_cycle
        run_in_cycle = resume % runs_per_cycle
        digit_start = run_in_cycle // passes_per_digit
        pass_start = run_in_cycle % passes_per_digit

        print("\n=== TRAINING CONFIGURATION ===")
        print(f"Total Runs: {total_runs}")
        print(f"Resume From Run: {resume}")
        print(f" → Cycle Start: {cycle_start}")
        print(f" → Digit Start: {digit_start}")
        print(f" → Pass Start : {pass_start}\n")

        # -------------------- Begin Training --------------------
        for cycle in range(cycle_start, cycles):

            print(f"\n=== Starting Cycle {cycle+1}/{cycles} ===")

            # Determine digit start location for this cycle
            for digit_idx in range(digit_start if cycle == cycle_start else 0, digits):

                digit = digit_dirs[digit_idx]
                digit_path = os.path.join(root, digit)
                if not os.path.isdir(digit_path):
                    continue

                # Load first .bin file for the digit
                bin_files = [f for f in os.listdir(digit_path) if f.endswith(".bin")]
                if not bin_files:
                    continue

                dataset_path = os.path.join(digit_path, bin_files[0])

                print(f"\n--- Cycle {cycle+1} | Digit {digit} | Using file: {bin_files[0]} ---")

                # Determine pass start
                if cycle == cycle_start and digit_idx == digit_start:
                    p_start = pass_start + 1
                else:
                    p_start = 1

                for p in range(p_start, passes_per_digit + 1):

                    current_run += 1
                    print(f"[Run {current_run}/{total_runs}] Cycle {cycle+1} | Digit {digit} | Pass {p}/{passes_per_digit}")

                    # Run Forward Pass
                    _ = self.ConvNet.ForwardPass(dataset_path, Total_Sim_Time=Total_Sim_Time)

                    # Save run progress
                    save_run_number(current_run, output_dir)

                    # Save outputs every 25 passes
                    if p % save_every == 0 or p == 1:
                        plots_dir = os.path.join(output_dir, "plots")
                        json_dir = os.path.join(output_dir, "json")
                        os.makedirs(plots_dir, exist_ok=True)
                        os.makedirs(json_dir, exist_ok=True)

                        plot_path = os.path.join(
                            plots_dir, f"run_{current_run}_cycle{cycle+1}_digit{digit}_pass{p}.png"
                        )
                        self.ConvNet.plot_all_output_neurons_single(plot_path)
                        print(f"Saved plot: {plot_path}")

                        json_path = os.path.join(
                            json_dir, f"run_{current_run}_cycle{cycle+1}_digit{digit}_pass{p}.json"
                        )
                        self.ConvNet.extract_full_network_with_topology(json_path)
                        print(f"Saved JSON: {json_path}")

                    # Reset states but keep learned weights
                    self.ConvNet.model.reset()

            # Reset digit and pass start after first cycle
            digit_start = 0
            pass_start = 0

        print("\n=== Completed All Cycles (3000 Runs) ===\n")


if __name__ == '__main__':
    unittest.main()
