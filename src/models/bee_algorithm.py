import tensorflow as tf

# Input is 2-D array, example:
#                       D1 D2 D3 
#                  Ap1 [ 6  9  7 
#                  Ap2   2  0  2 
#                  Ap3   1  0  0 ]

# Potential food sources => Admisible paths (m')
# Number of best food sources => Best admisible paths to exploit (e)
# Quality assesment => Number of nodes (jumps) in the path - the less is better or loss function / target function (less is better).
# Hive => Demand (Di)
# Bees => Demand value (n)

# 1st strategy (aggregation) => all bees assigned to the best food source
# 2nd strategy (disaggregation) => some bees assigned to the best food source and rest randomly to the remaining ones

demands_count = 3
admisible_paths_count = 3

randomnes_seed = 77
# custom flavor for the algorithm
local_exp_step = 10

chromosome = tf.zeros((admisible_paths_count, demands_count), dtype='float32') # Food Sources
print(chromosome)

def train_model(data, colony_size = 10, maximum_cycles = 10) -> list:
    # Scouting phase
    pass
