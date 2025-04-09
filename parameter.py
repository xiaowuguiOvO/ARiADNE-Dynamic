# saving path
FOLDER_NAME = 'ariadne1'
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

# save training data
SUMMARY_WINDOW = 32  # how many training steps before writing data to tensorboard
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 40  # how many episodes before saving a gif

# map and planning resolution
CELL_SIZE = 0.1  # meter, your map resolution
NODE_RESOLUTION = 1  # meter, your node resolution
FRONTIER_CELL_SIZE = 2 * CELL_SIZE  # do you want to downsample the frontiers

# map representation
FREE = 255  # value of free cells in the map
OCCUPIED = 1  # value of obstacle cells in the map
UNKNOWN = 127  # value of unknown cells in the map

# sensor and utility range
SENSOR_RANGE = 16  # meter
UTILITY_RANGE = 0.8 * SENSOR_RANGE  # consider frontiers within this range as observable
MIN_UTILITY = 2  # ignore the utility if observable frontiers are less than this value

# updating map range w.r.t the robot
UPDATING_MAP_SIZE = 4 * SENSOR_RANGE + 4 * NODE_RESOLUTION  # nodes outside this range will not be affected by current measurements

# training parameters
REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 100
BATCH_SIZE = 64
LR = 1e-5
GAMMA = 1
NUM_META_AGENT = 16  # how many threads does your CPU have

# network parameters
# NODE_INPUT_DIM = 4
NODE_INPUT_DIM = 4  #node_x, node_y, node_utility, node_guidepost, obstacle_velocity_x, obstacle_velocity_y
EMBEDDING_DIM = 128

# Graph parameters
K_SIZE = 25  # the number of neighboring nodes, fixed
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value, need it for batch training

# GPU usage
USE_GPU = False  # do you want to collect training data using GPUs (better not)
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 0  # 0 unless you want to collect data using GPUs

SPEED_SCALE = 10
# 动态障碍物参数
NUM_DYNAMIC_OBSTACLES = 10      # 动态障碍物数量
MIN_OBSTACLE_SPEED = 0.5 * SPEED_SCALE      # 最小障碍物速度
MAX_OBSTACLE_SPEED = 2 * SPEED_SCALE       # 最大障碍物速度
OBSTACLE_RADIUS = 0.4          # 障碍物半径 (m)
ROBOT_RADIUS = 0.3             # 机器人半径 (m)
COLLISION_PENALTY = 0       # 碰撞惩罚
SAFE_DISTANCE = 1.5            # 安全距离 (m)
PROXIMITY_PENALTY = 1        # 接近惩罚系数
WALL_COLLISION_PENALTY = 0.05  # 墙壁碰撞惩罚

OBSTACLE_CURVE_MIN_DIST = 5  # 最小障碍物轨迹距离
OBSTACLE_CURVE_MAX_DIST = 20  # 最大障碍物轨迹距离

# 时间和频率控制
MAX_EPISODE_STEP = 1280
MAX_EPISODE_TIME = 300.0         # 最大模拟时间(秒)
STEP_SIZE = 0.2               # 仿真时间步长(秒)
DECISION_INTERVAL = 0.1         # 决策间隔时间(秒)
DECISION_DISTANCE = 2.0          # 决策距离阈值(米)
VISUALIZATION_INTERVAL = 20     # 可视化帧间隔(步数)

# MAX_VELOCITY = 12.0
# MIN_VELOCITY = -12.0
MAX_LINEAR_VELOCITY = 6 * SPEED_SCALE
MIN_LINEAR_VELOCITY = 0.5
MAX_ANGULAR_VELOCITY = 1.0 * SPEED_SCALE
WAYPOINT_THRESHOLD = 0.5
# 目标网络更新参数
TARGET_UPDATE_INTERVAL = 10  # 每隔多少步更新一次目标网络
TAU = 0.005  # 软更新比例
SAVE_INTERVAL = 100  # 每隔多少轮保存一次模型

ENTROPY_TARGET = -2.0  

LOCAL_CONTROLLER_PATH = 'saved_models/best_model/actor_good.pth'