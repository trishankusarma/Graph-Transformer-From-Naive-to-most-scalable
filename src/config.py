# Motivation :
# -- all hyperparameters in one place
from dataclasses import dataclass

@dataclass
class Config:
    # A. Model params:
    d_model    :int  = 64
    num_heads  :int = 4
    num_layers :int = 4
    dropout    :float = 0.4
    k_lap_pe   :int = 32     # (number of LapPE eigenvectors)
    max_dist   :int = 10    #  (SPD beyond this gets bucketed together)

    # B.Training params:
    lr            :float  = 0.005
    epochs        :int  = 200
    weight_decay  :float  = 5e-4

    seed : int = 42    # for reproducibility