# Motivation :
#  -- training loop
import sys
from .dataset import load_data

# Step 1: 
if __name__ == "__main__":

    input_data = sys.argv[1]
    print("Step 1: Loading training dataset from Cora :: building up the positional encodings and concat")
    load_data(input_data)
