# Motivation :
# -- all hyperparameters in one place
# A. Model params:
d_model     = 64
num_heads   = 4
num_layers  = 4
dropout     = 0.6
k_lap_pe    = 8     # (number of LapPE eigenvectors)
max_dist    = 10    #  (SPD beyond this gets bucketed together)

# B.Training params:
lr              = 0.005
epochs          = 200
weight_decay    = 5e-4

# C. Data params:
dataset_path  = './data/cora'
num_classes   = 7
feat_dim      = 1433