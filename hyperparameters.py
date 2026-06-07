minibatch_size = 32 ### might want to increase
replay_memory_size = 200000 ### CANNOT USE MORE THAN 100K - 17 GB AT 80000 TUPLES
agent_history_length = 4
target_network_frequency_update = 10000
discount_factor = 0.99
action_repeat = 4
update_frequency = 4

learning_rate = 0.00025
pcnn_learning_rate = 0.00025

gradient_momentum = 0.95
squared_gradient_momentum = 0.95
min_squared_gradient = 0.01

initial_exploration = 1.0
final_exploration = 0.1
final_exploration_frame = 500000

replay_start_size = 50000
no_op_max = 30

TARGET_UPDATE_FREQUENCY = 10000  # Hard update every 10000 steps (Mnih et al. 2015)
USE_SOFT_UPDATE = False       # Use hard updates
SOFT_UPDATE_TAU = 0.01        # Only for soft updates
TOTAL_FRAMES = 10000000

