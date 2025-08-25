minibatch_size = 64 ### might want to increase to 128 even
replay_memory_size = 500000
agent_history_length = 4
target_network_frequency_update = 10000 # TODO
discount_factor = 0.99
action_repeat = 4
update_frequency = 4

learning_rate = 0.00025
pcnn_learning_rate = 0.00025

gradient_momentum = 0.95 # TODO
squared_gradient_momentum = 0.95
min_squared_gradient = 0.01

initial_exploration = 1.0
final_exploration = 0.1
final_exploration_frame = 500000

replay_start_size = 80000
no_op_max = 30

TARGET_UPDATE_FREQUENCY = 1000  # Hard update every 1000 steps
USE_SOFT_UPDATE = False       # Use hard updates
SOFT_UPDATE_TAU = 0.01        # Only for soft updates
TOTAL_FRAMES = 20000000

ENABLE_MIXED_PRECISION = True
COMPILE_MODELS = False
NON_BLOCKING_TRANSFER = True