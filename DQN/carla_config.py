import os

env_params = {
    'save_freq' : 50
}

#TRAINING MODES
TRAIN_PLAY_MODE = 0  # Train=1 Play=0
SAVE_DATA = 0 # 1 ; USED IN EVALUATION ONLY
TRAIN_PLAY = ["PLAY", "TRAIN"]
WORKING_MODE_OPTIONS = ["WAYPOINTS_CARLA","WAYPOINTS_IMAGE","CNN_BW_TRAJECTORY","CNN_RGB","CNN_RGB_TRAJECTORY",
                        "CNN_SEMANTIC", "CNN_GRAYSCALE", "CNN_FLATTEN", "TP_ANG", "PRE_TRAINED_CNN"]

# "WAYPOINTS_CARLA" ->  DRL-Carla-Waypoints Agent
# "WAYPOINTS_IMAGE" -> DRL-Pre-CNN Agent
# "CNN_BW_TRAJECTORY" -> DRL-CNN Agent
# "CNN_RGB" -> DRL-CNN Agent
# "CNN_RGB_TRAJECTORY" -> DRL-CNN Agent
# "CNN_SEMANTIC" -> DRL-CNN Agent
# "CNN_GRAYSCALE" -> DRL-CNN Agent
# "CNN_FLATTEN" -> DRL-Flatten-Image Agent
# "TP_ANG" -> Not in the paper
# "PRE_TRAINED_CNN" ->  DRL-Pre-CNN Agent

WORKING_MODE = "WAYPOINTS_CARLA" 

#TRAINING STAGES
CARLA_MAP = 'Town01'   # "mapa_oscar_v5"
TRAIN_MODE_OPTIONS=["RANDOM", "STRAIGHT", "TURN_LEFT", "TURN_RIGHT", "TURN_RIGHT_LEFT", "TURN_LEFT_RIGHT", "ALTERNATIVE", "RANDOM_TURN"]
TRAIN_MODE = "TURN_RIGHT_LEFT" # "TURN_RIGHT_LEFT" 

path2CARLA = "/home/jorge-daniel/Desktop/carla/"

#IMAGE CONFIGURATION
IM_WIDTH_VISUALIZATION = 640*2
IM_HEIGHT_VISUALIZATION = 480
# for Flatten-Image Agent, the paper resizes image to 11×11
IM_WIDTH_CNN = 11 
IM_HEIGHT_CNN = 11 

tau = 0.001  # Target Network HyperParameter
lra = 0.0001  # Learning rate for Actor
lrc = 0.001  # Learning rate for Critic
episodes_num = 3500
max_steps = 100000
buffer_size = 100000
batch_size = 32
gamma = 0.99  # discount factor
hidden_units = (300, 600)
SECONDS_PER_EPISODE = 70
SHOW_CAM = 1
SHOW_WAYPOINTS = 1
SHOW_CAM_RESIZE = 0

########################   ADD FOR DQN   #####################
ACTIONS_NAMES = {
    0: 'forward_slow',
#    1: 'forward_medium',
    1: 'left_slow',
    2: 'left_medium',
    3: 'right_slow',
    4: 'right_medium',
#    6: 'brake_light',
    #3: 'no_action',
}
N_actions = len(ACTIONS_NAMES)

ACTION_CONTROL = {
    0: [0.55, 0, 0],
    # 1: [0.7, 0, 0],
    1: [0.4, 0, -0.1],
    2: [0.4, 0, -0.4],
    3: [0.4, 0, 0.1],
    4: [0.4, 0, 0.4],
    # 6: [0, 0.3, 0],
    #3: None,
}

reward_mode = 2
STEER_AMT = 0.2
CNN_MODEL = 2
DISCOUNT = 0.99
if TRAIN_PLAY_MODE == 1:
    epsilon = 1
elif TRAIN_PLAY_MODE == 0:
    epsilon = 0

EPSILON_DECAY = 0.9995
MIN_EPSILON = 0.01
REPLAY_MEMORY_SIZE = 5_000
MIN_REPLAY_MEMORY_SIZE = 1_000
MINIBATCH_SIZE = 16
PREDICTION_BATCH_SIZE = 4
TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 8
UPDATE_TARGET_EVERY = 5
EPISODES = 20_000
AGGREGATE_STATS_EVERY = 10
AGENT_PATH = "../models/data_" + str(WORKING_MODE) + "/"
MODEL_PATH = "../models/data_WAYPOINTS_CARLA/TURN_RIGHT_LEFT_best_reward_ep_4838_model.model" 

N_save_stats = 100
########################   ADD FOR DQN   #####################

#Vista de pájaro
BEV_PRE_CNN = 0

#WORKING TYPE SELECTION
WAYPOINTS = 'X'     # X to only use X, XY for 2D
THRESHOLD = 0  
DRAW_TRAJECTORY = 0 
IM_LAYERS = 1
state_dim = 16 # Input dimension to network
dimension_vector_estado = 16 
if WORKING_MODE == WORKING_MODE_OPTIONS[0]:         # WAYPOINTS_CARLA
    if WAYPOINTS == 'XY':
        state_dim = 31
        dimension_vector_estado = state_dim
    elif WAYPOINTS == 'X':
        state_dim = 16
    CAM_X = 1.0
    CAM_Z = 1.8
    CAM_PITCH = -20.0
    CAM_YAW = 0.0
    CAM_ROLL = 0.0
elif WORKING_MODE == WORKING_MODE_OPTIONS[1]:       # WAYPOINTS_IMAGE
    DRAW_TRAJECTORY = 1
    state_dim = 17
elif WORKING_MODE == WORKING_MODE_OPTIONS[2]:       # CNN_BW_TRAJECTORY
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    THRESHOLD = 1                               # FLAG DE UMBRALIZACIÓN
    DRAW_TRAJECTORY = 1
elif WORKING_MODE == WORKING_MODE_OPTIONS[3]:       # CNN_RGB
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[4]:       # CNN_RGB_TRAJECTORY
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    DRAW_TRAJECTORY = 1
elif WORKING_MODE == WORKING_MODE_OPTIONS[5]:       # CNN_SEMANTIC
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 1                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[6]:       # CNN_GRAYSCALE
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
elif WORKING_MODE == WORKING_MODE_OPTIONS[7]:       # CNN_FLATTEN
    IM_LAYERS = 1                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    THRESHOLD = 1                               
    DRAW_TRAJECTORY = 1
    state_dim = IM_WIDTH_CNN * IM_HEIGHT_CNN

elif WORKING_MODE == WORKING_MODE_OPTIONS[8]:       # TRACKPOS_ANGLE
    state_dim = 2

elif WORKING_MODE == WORKING_MODE_OPTIONS[9]:       # PRE-TAINED-CNN
    if WAYPOINTS == 'XY':
        state_dim = 31
        dimension_vector_estado = state_dim
    elif WAYPOINTS == 'X':
        state_dim = 16
    IM_LAYERS = 3                               # GrayScale = 1, Color = 3
    IM_TYPE = 0                                 # RGB=0, SemanticSegmetnation=1
    DRAW_TRAJECTORY = 1
    THRESHOLD = 0                               
    IM_WIDTH_VISUALIZATION = 2*640
    CAM_X = 1.0
    CAM_Z = 2
    CAM_PITCH = -20.0
    CAM_YAW = 0.0
    CAM_ROLL = 0.0
    BEV_PRE_CNN = 1