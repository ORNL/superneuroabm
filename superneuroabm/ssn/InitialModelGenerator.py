import argparse
import sys
import json
from SpikingConv2dt import Conv2dtNet   # correct class import


# ---------------- Argument parser ----------------
parser = argparse.ArgumentParser(description="Conv2dtNet Parameter Loader")
parser.add_argument("--Conv_Parameters", required=True, help="Path to Conv parameter JSON file")

args, remaining_argv = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining_argv


# ---------------- Helper Functions ----------------
def Generate_Model(Conv_Kernel_List, Output_Classes, Input_W, Input_H):
    ConvNet = Conv2dtNet()
    ConvNet.model.setup(use_gpu=True)
    ConvNet.NetworkConstruction(
        Conv_Kernel_List=Conv_Kernel_List,
        output_classes=Output_Classes,
        Input_W=Input_W,
        Input_H=Input_H
    )
    return ConvNet


def Extract_Model_JSON(ConvNet, JSON_Path):
    ConvNet.extract_full_network_with_topology(JSON_Path)   # corrected name


# ---------------- Main ----------------
if __name__ == "__main__":

    param_file = args.Conv_Parameters

    print(f"\nLoading Conv Parameters from: {param_file}\n")

    # ---- Open JSON file and print each line ----
    with open(param_file, "r") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines, start=1):
        print(f"Line {idx}: {line.rstrip()}")

    params = [json.loads(line) for line in lines if line.strip()]

    print("\nParsed JSONL parameters:")
    for p in params:
        ConvNet=Generate_Model(p['Conv_Kernel_List'], p['Output_Classes'], p['Input_W'], p['Input_H'])
        ModelName=p['ModelName']
        Extract_Model_JSON(ConvNet,f'./InitialModel/{ModelName}.json')
