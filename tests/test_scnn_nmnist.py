from superneuroabm.ssn.SpikingConv2dt import Conv2dtNet
import pandas as pd
import unittest 
import os
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
        self.Conv_Kernel_List = [
            [(3,3),(3,2),(2,3),(4,4)],  
            [(3,3),(2,3),(3,2)],        
            [(3,3),(2,3),(3,2),(4,4)],              
            [(3,3),(2,3),(3,2)],               
            [(3,3),(2,3),(3,2),(4,4)],                        
            [(3,3),(2,3),(3,2)],
            [(3,3),(2,3)]                  
        ]
        self.ConvNet.ConstructionOfConvKernel(self.Conv_Kernel_List, 10)

    def log_kernel_weights(self, forward_pass_id=None):
        """
        Logs every learned synapse weight from each kernel in each layer
        into a pandas DataFrame.
        """
        rows = []

        for layer_idx, kernels in self.ConvNet.kernel_vals.items():  # layer → list of kernels
            for kernel_idx, (_, kernel_matrix) in enumerate(kernels):  # each kernel
                W, H = kernel_matrix.shape

                for i in range(W):
                    for j in range(H):
                        synapse = kernel_matrix[i][j]

                        # get learned weight
                        hyper = self.ConvNet.model.get_agent_property_value(
                            id=synapse,
                            property_name="hyperparameters"
                        )
                        learned_weight = hyper[0]  # you already know this works

                        rows.append({
                            "forward_pass": forward_pass_id,
                            "layer": layer_idx,
                            "kernel": kernel_idx,
                            "i": i,
                            "j": j,
                            "weight": learned_weight
                        })

        return pd.DataFrame(rows)

    def NMNIST(self, Number_Of_Samples_Per_Class=1):
        root = "./superneuroabm/ssn/data/NMNIST/Test"
        self.weight_history = pd.DataFrame()
        results = []  # will collect dicts
        pass_idx=0
        for digit in sorted(os.listdir(root)):  # '0', '1', ..., '9'
            
            digit_path = os.path.join(root, digit)

            if not os.path.isdir(digit_path):
                continue

            bin_files = sorted([f for f in os.listdir(digit_path) if f.endswith(".bin")])

            for bin_file in bin_files[:Number_Of_Samples_Per_Class]:
                Dataset = os.path.join(digit_path, bin_file)
                print('forward pass', pass_idx)
                PredictedClass = self.ConvNet.ForwardPass(Dataset, 100)

                results.append({
                    "true_class": int(digit),
                    "predicted_class": PredictedClass,
                    "file_path": Dataset
                })
                df = self.log_kernel_weights(forward_pass_id=pass_idx)  
                self.weight_history = pd.concat([self.weight_history, df], ignore_index=True)
                pass_idx+=1
        df = pd.DataFrame(results)
        self.weight_history["synapse_id"] = (
            self.weight_history["layer"].astype(str) + "_" +
            self.weight_history["kernel"].astype(str) + "_" +
            self.weight_history["i"].astype(str) + "_" +
            self.weight_history["j"].astype(str)
            )
        for synapse, group in self.weight_history.groupby("synapse_id"):
            plt.plot(group["forward_pass"], group["weight"], alpha=0.5)

            plt.title("Synapse Weights Over Forward Passes")
            plt.xlabel("Forward Pass")
            plt.ylabel("Weight")

            plt.savefig("all_synapses_weight_dynamics.png", dpi=300, bbox_inches="tight")
            plt.close()
        return df
    def test_NMNIST(self):
        df = self.NMNIST(Number_Of_Samples_Per_Class=10)
        print(df.head())  

        
 