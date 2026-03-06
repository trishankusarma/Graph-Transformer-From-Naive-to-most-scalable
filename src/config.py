# Motivation :
# -- all hyperparameters in one place
from dataclasses import dataclass

@dataclass
class Config:
    # A. Model params:
    d_model    :int  = 64
    num_heads  :int = 4
    num_layers :int = 2 # this refers to transformer layers
    num_gcn_layers :int = 2 
    dropout    :float = 0.4
    k_lap_pe   :int = 32     # (number of LapPE eigenvectors)
    max_dist   :int = 10    #  (SPD beyond this gets bucketed together)
    d_ff       :int = 256   # 4*d_model

    # B.Training params:
    lr            :float  = 5e-4
    epochs        :int  = 600
    weight_decay  :float  = 1e-2
    max_gradient_clip :float = 1.0
    eta_min :float  = 1e-5

    seed :int = 42    # for reproducibility

    # C. Database params
    d_classes :int = 7

    # D. Miscellenous
    skip_accuracy_freq :int = 10
    patience :int = 50
    dropout_input :float = 0.5
    no_of_cpu_threads :int = 4
    approach :int = 3

    # Approach 3
    num_lg_layers :int = 2
    leaky_relu_slope :float = 0.2

    # Approach 4
    max_degree = 500