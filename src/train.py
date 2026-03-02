# Motivation :
#  -- training loop
import sys
from .dataset import load_data_and_evaluate_pe

# Step 1: 
if __name__ == "__main__":

    input_data = sys.argv[1]
    print("Step 1: Loading training dataset from Cora :: building up the positional encodings and concat")
    dataset_info = load_data_and_evaluate_pe(input_data)
