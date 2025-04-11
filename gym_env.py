# -*- coding: utf-8 -*-
import os
import gymnasium as gym # 导入 Gymnasium 库
from gymnasium import spaces
import numpy as np
from skimage import io
from skimage.measure import block_reduce
import matplotlib.pyplot as plt
import time
import traceback # 用于打印详细错误

# --- Gym 环境定义 ---
class NavigationEnv(gym.Env):
    """
    用于机器人在2D栅格地图上导航的自定义 Gym 环境。

    观测空间 (Observation): 机器人的当前状态 [行, 列, 角度θ] (连续值)。
    动作空间 (Action):    期望的控制输入 [线速度, 角速度] (连续值)。
    奖励 (Reward):      碰撞时为 -10，否则每步为 -0.1。
    终止条件 (Termination): 发生碰撞。
    截断条件 (Truncation):  达到最大步数限制。
    """
    # metadata 用于指定渲染模式和帧率
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self,
                 map_dir='maps',                    # 地图文件所在的目录
                 resolution=0.1,                    # 地图分辨率，每像素代表多少米
                 max_linear_speed=10.0,              # 最大线速度 (米/秒)
                 max_angular_speed=1.0,             # 最大角速度 (弧度/秒)
                 dt=0.1,                            # 模拟的时间步长(秒)
                 max_episode_steps=500,             # 每回合最大步数
                 render_mode=None):                 # 渲染模式 ('human', 'rgb_array', or None)
        super().__init__() # 初始化父类

        self.map_dir = map_dir
        self.resolution = resolution              # 地图分辨率，单位：米/像素
        self.max_linear_speed = max_linear_speed  # 最大线速度，单位：米/秒
        self.max_angular_speed = max_angular_speed # 最大角速度，单位：弧度/秒
        self.dt = dt
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # --- 检查地图目录 ---
        if not os.path.isdir(self.map_dir):
            raise FileNotFoundError(f"地图目录 '{self.map_dir}' 未找到。")
        # 获取并排序地图文件列表
        self.map_list = sorted([f for f in os.listdir(map_dir) if os.path.isfile(os.path.join(map_dir, f))])
        if not self.map_list:
            raise FileNotFoundError(f"在目录 '{self.map_dir}' 中没有找到地图文件。")

        self.episode_index = -1 # 当前回合使用的地图索引
        self.current_step = 0   # 当前回合的步数

        # --- 状态变量 ---
        self.ground_truth = None # 处理后的二值地图 (1=自由, 0=障碍)
        self.robot_pos = None    # 机器人连续位置 [行, 列]，单位：米
        self.robot_theta = None  # 机器人连续朝向角度 (弧度)
        self.map_shape = None    # 加载地图的形状，单位：像素
        self.map_size_meters = None # 地图尺寸，单位：米

        # --- 定义动作空间 ---
        # 连续控制: [线速度, 角速度]
        # 如果允许后退，线速度下限可以为负
        self.action_space = spaces.Box(
            low=np.array([-self.max_linear_speed, -self.max_angular_speed]),
            high=np.array([self.max_linear_speed, self.max_angular_speed]),
            shape=(2,),
            dtype=np.float32
        )

        # --- 定义观测空间 ---
        # 状态: [行, 列, 角度θ]
        # 行/列的边界依赖于地图, 角度θ范围是 [-pi, pi]
        # 这里先定义结构，具体边界在 reset() 中根据地图大小确定
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi]),
            high=np.array([np.inf, np.inf, np.pi]), # 地图边界在 reset 中设置
            shape=(3,),
            dtype=np.float32
        )

        # --- 渲染相关的变量 ---
        self.fig = None
        self.ax = None
        self.map_handle = None    # 地图图像的句柄
        self.robot_handle = None  # 机器人中心点的句柄
        self.orient_handle = None # 机器人朝向线的句柄

    def _import_ground_truth(self, episode_index):
        """加载并处理一个地图文件。"""
        map_index = episode_index % len(self.map_list) # 循环使用地图
        map_path = os.path.join(self.map_dir, self.map_list[map_index])
        print(f"==== loading new map: {self.map_list[map_index]} (index: {map_index}), episode: {episode_index} ====")

        try:
            # 以灰度模式读取图像
            ground_truth_raw = io.imread(map_path, as_gray=True)
            print(f"original map - shape: {ground_truth_raw.shape}, dtype: {ground_truth_raw.dtype}, max value: {ground_truth_raw.max()}, file: {map_path}")
            # 如果像素值在 [0, 1] 范围，则缩放到 [0, 255]
            if ground_truth_raw.max() <= 1.0 + 1e-6: # 加一点容差
                 ground_truth_raw = (ground_truth_raw * 255)
            ground_truth_raw = ground_truth_raw.astype(int)
        except Exception as e:
            print(f"!!!!!! error loading map file {map_path}: {e} !!!!!!")
            # 返回 None 以便 reset 函数处理错误
            return None, None

        # 使用 block_reduce 进行降采样，因子为 2x2，使用最小值聚合
        try:
            ground_truth_reduced = block_reduce(ground_truth_raw, (2, 2), np.min)
        except ValueError as e:
             print(f"!!!!!! error downsampling map (block_reduce) {map_path}: {e}. maybe the map is too small or irregular. !!!!!!")
             return None, None

        # 寻找机器人起始位置 (像素值为 208 的单元格)
        robot_start_value = 208
        robot_cells = np.nonzero(ground_truth_reduced == robot_start_value)

        if len(robot_cells[0]) > 0:
            # 获取找到的第一个单元格的索引 [行, 列]
            robot_start_cell_indices = np.array([robot_cells[0][0], robot_cells[1][0]])
            print(f"found the specified robot start cell index: {robot_start_cell_indices}")
        else:
            # 如果找不到 208，打印警告并使用默认位置 [1, 1]
            print(f"警告: 在地图中未找到机器人起始值 {robot_start_value}。将机器人放置在默认位置 [1, 1]。")
            robot_start_cell_indices = np.array([1, 1])

        # --- 根据你的逻辑处理地图: 1 代表自由空间, 0 代表障碍物 ---
        # 自由空间条件: > 150 或者 (>= 50 且 <= 80)
        is_free = (ground_truth_reduced > 150) | ((ground_truth_reduced <= 80) & (ground_truth_reduced >= 50))
        ground_truth_processed = is_free.astype(np.uint8) # 转换为 uint8 类型 (1 = 自由, 0 = 障碍)
        # print(f"处理后地图 - 形状: {ground_truth_processed.shape}, 自由空间单元数: {np.sum(ground_truth_processed)}")

        # --- 验证并调整起始位置 ---
        r_idx, c_idx = robot_start_cell_indices
        map_rows, map_cols = ground_truth_processed.shape
        # 检查起始索引是否在界内且是自由空间
        if not (0 <= r_idx < map_rows and 0 <= c_idx < map_cols and ground_truth_processed[r_idx, c_idx] == 1):
            print(f"警告: 初始机器人位置 {robot_start_cell_indices} 是障碍物或越界。正在搜索附近的自由单元格...")
            # 查找所有自由空间的单元格
            free_cells = np.argwhere(ground_truth_processed == 1)
            if len(free_cells) > 0:
                # 简单策略：使用找到的第一个自由单元格
                # 更优策略：找到离原定起始点最近的自由单元格
                robot_start_cell_indices = free_cells[0]
                print(f"已将机器人起始位置重新定位到第一个可用的自由单元格: {robot_start_cell_indices}")
            else:
                 # 如果地图完全没有自由空间，这是个严重错误
                 print("!!!!!! 严重错误: 地图中找不到任何自由空间可供机器人起始。 !!!!!!")
                 raise ValueError("地图中找不到任何自由空间可供机器人起始。")

        # 将起始单元格索引转换为连续位置 (单元格中心)
        robot_start_pos = robot_start_cell_indices.astype(np.float32) + 0.5
        print(f"机器人起始位置 [行, 列]: {robot_start_pos}")

        return ground_truth_processed, robot_start_pos # 返回处理后的地图和起始位置 [行, 列]

    def _normalize_angle(self, angle):
        """将角度归一化到 [-pi, pi] 范围内。"""
        return np.arctan2(np.sin(angle), np.cos(angle))

    def reset(self, seed=None, options=None):
        """重置环境到初始状态。"""
        super().reset(seed=seed) # 调用父类的 reset 以处理随机种子

        self.episode_index += 1 # 更新回合索引
        self.current_step = 0   # 重置步数计数器

        # 加载地图和初始位置 [行, 列]
        self.ground_truth, robot_start_pos = self._import_ground_truth(self.episode_index)

        # 检查地图加载是否成功
        if self.ground_truth is None:
             # 可以尝试加载下一个地图，或者直接抛出错误
             raise RuntimeError(f"为回合 {self.episode_index} 加载地图失败。")

        self.map_shape = self.ground_truth.shape # 获取地图形状

        # 确保机器人位置在地图内部
        self.robot_pos = np.clip(
            robot_start_pos,
            [0.5, 0.5],  # 最小值，确保至少在第一个单元格的中心
            [self.map_shape[0] - 0.5, self.map_shape[1] - 0.5]  # 最大值，确保在地图内
        )

        # 初始化机器人朝向角度 (例如，初始朝向右方，沿列方向)
        self.robot_theta = 0.0
        # 或者可以随机初始化朝向:
        # self.robot_theta = self.np_random.uniform(-np.pi, np.pi)

        # 根据加载的地图更新观测空间的边界
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.pi]), # 下边界 [行, 列, 角度]
            high=np.array([self.map_shape[0], self.map_shape[1], np.pi], dtype=np.float32), # 上边界
            shape=(3,),
            dtype=np.float32
        )
        print(f"Environment reset. Map shape: {self.map_shape}, Initial state: [{self.robot_pos[0]:.2f}, {self.robot_pos[1]:.2f}, {self.robot_theta:.2f}]")

        # 获取初始观测值和信息字典
        observation = self._get_obs()
        info = self._get_info()

        # 如果是 human 模式并且渲染窗口已创建，则更新渲染
        if self.render_mode == 'human' and self.fig is not None:
             # 确保渲染器也重置到新地图和状态
             self._render_frame()

        return observation, info # Gymnasium 的 reset 返回 observation 和 info

    def step(self, action):
        """执行一个时间步的环境模拟。"""
        # 1. 获取动作并限制在有效范围内
        linear_v = np.clip(action[0], -self.max_linear_speed, self.max_linear_speed)
        angular_w = np.clip(action[1], -self.max_angular_speed, self.max_angular_speed)

        # 2. 存储当前状态，用于运动学计算
        current_pos = self.robot_pos.copy()  # 使用副本，避免直接修改
        current_theta = self.robot_theta

        # 3. 使用独轮车模型运动学计算状态变化 (使用简单的欧拉积分)
        # 首先更新角度
        potential_theta = current_theta + angular_w * self.dt
        potential_theta = self._normalize_angle(potential_theta) # 将角度保持在 [-pi, pi]

        # 根据 *当前* 角度更新位置 (简单欧拉法)
        # 注意: theta=0 对应 +列 (+x) 方向, theta=pi/2 对应 +行 (+y) 方向 (假设 origin='lower')
        delta_row = linear_v * np.sin(current_theta) * self.dt
        delta_col = linear_v * np.cos(current_theta) * self.dt
        potential_pos = current_pos + np.array([delta_row, delta_col])

        # 4. 进行碰撞检测
        collision = False
        terminated = False # 是否因碰撞等终止
        truncated = False  # 是否因达到最大步数而截断

        next_row, next_col = potential_pos
        # 检查地图边界
        if not (0 <= next_row < self.map_shape[0] and 0 <= next_col < self.map_shape[1]):
            collision = True
            # print("碰撞: 超出边界") # 可以取消注释以获取详细碰撞信息
        else:
            # 检查是否与障碍物碰撞 (使用整数化后的单元格索引)
            cell_row, cell_col = int(np.floor(next_row)), int(np.floor(next_col))
            # 再次检查索引是否在有效范围内 (虽然理论上边界检查已覆盖，但为了安全)
            if not (0 <= cell_row < self.map_shape[0] and 0 <= cell_col < self.map_shape[1]):
                 collision = True
                 # print("碰撞: 地板除后越界")
            elif self.ground_truth[cell_row, cell_col] == 0: # 值为 0 代表障碍物
                collision = True
                # print(f"碰撞: 在单元格 ({cell_row}, {cell_col}) 遇到障碍物")

        # 5. 更新状态和计算奖励
        if collision:
            # 如果发生碰撞，机器人的状态保持不变
            # 确保机器人位置仍在地图内
            self.robot_pos = np.clip(
                current_pos,  # 使用当前位置（碰撞前的位置）
                [0.5, 0.5],  # 最小值
                [self.map_shape[0] - 0.5, self.map_shape[1] - 0.5]  # 最大值
            )
            reward = -10.0   # 碰撞惩罚
            terminated = True # 发生碰撞，回合终止
        else:
            # 如果没有碰撞，更新机器人的位置和角度
            self.robot_pos = potential_pos
            self.robot_theta = potential_theta
            reward = -0.1    # 每走一步给予小的负奖励，鼓励效率

        # 6. 检查是否达到最大步数限制 (截断)
        self.current_step += 1
        if self.current_step >= self._max_episode_steps:
            truncated = True # 达到最大步数，回合截断
            # print(f"回合截断: 已达到最大步数 {self._max_episode_steps}")

        # 7. 获取当前的观测值和信息字典
        observation = self._get_obs()
        info = self._get_info() # 当前信息字典为空

        # 8. 如果需要，进行渲染
        if self.render_mode == 'human':
            self._render_frame()

        # 返回 Gymnasium step 方法的标准 5 个值
        return observation, reward, terminated, truncated, info

    def _get_obs(self):
        """获取当前的环境观测值。"""
        # 确保机器人位置在观测空间边界内
        if self.robot_pos is not None and self.map_shape is not None:
            self.robot_pos = np.clip(
                self.robot_pos,
                [0.0, 0.0],  # 下边界
                [self.map_shape[0] - 1e-6, self.map_shape[1] - 1e-6]  # 上边界，稍微缩小以防止精度问题
            )
        
        # 观测值是机器人的状态: [行, 列, 角度θ]
        return np.array([self.robot_pos[0], self.robot_pos[1], self.robot_theta], dtype=np.float32)

    def _get_info(self):
        """获取环境的辅助信息。"""
        # 目前为空，未来可以加入例如到目标的距离、碰撞状态等信息
        return {}

    def render(self):
        """根据 render_mode 渲染环境。"""
        if self.render_mode == 'rgb_array':
            # 返回渲染结果的 RGB 图像数组
            return self._render_frame(capture=True)
        elif self.render_mode == 'human':
             # 在窗口中显示渲染结果
             self._render_frame(capture=False)

    def _render_frame(self, capture=False):
        """执行实际的渲染操作 (使用 Matplotlib)。"""
        if self.render_mode is None:
             gym.logger.warn("你正在调用 render 方法，但没有指定任何渲染模式。")
             return
        # 检查状态是否已初始化，防止在 reset 完成前调用 render
        if self.robot_pos is None or self.robot_theta is None or self.ground_truth is None:
             # print("无法渲染，环境状态尚未完全初始化。")
             return # 在状态准备好之前不进行渲染

        line_length = 3.0 # 绘制的机器人朝向线的长度 (与坐标单位相同)

        # --- 初始化 Matplotlib 图形和坐标轴 (仅在首次调用时执行) ---
        if self.fig is None:
            plt.ion() # 打开交互模式
            self.fig, self.ax = plt.subplots(1, 1, figsize=(7, 7)) # 调整图像大小

            # 显示地图图像 (将 0(障碍) 显示为白色, 1(自由) 显示为黑色，使用 cmap='gray_r')
            # 或者 (1 - self.ground_truth) 使障碍物为黑色
            self.map_handle = self.ax.imshow(1 - self.ground_truth, cmap='gray', origin='lower',
                                             extent=[0, self.map_shape[1], 0, self.map_shape[0]])

            # 绘制机器人中心点 (一个小圆圈)
            self.robot_handle, = self.ax.plot([], [],
                                               marker='o', markersize=6, # 调整标记大小
                                               linestyle='None', color='r', label='agent')

            # 绘制机器人朝向线
            # 计算线的初始起点和终点
            start_col = self.robot_pos[1]
            start_row = self.robot_pos[0]
            end_col = start_col + line_length * np.cos(self.robot_theta)
            end_row = start_row + line_length * np.sin(self.robot_theta)
            # 绘制线段，设置颜色、线宽和 zorder (控制绘制层级)
            self.orient_handle, = self.ax.plot([start_col, end_col], [start_row, end_row],
                                                color='blue', linewidth=2, zorder=4) # 红色粗线

            # 设置图形属性
            self.ax.set_title("Navigation Environment")
            self.ax.set_xlabel("Column (X coordinate)")
            self.ax.set_ylabel("Row (Y coordinate)")
            # self.ax.legend() # 如果觉得圆点+线已经足够清晰，可以注释掉图例
            self.ax.grid(False) # 不显示网格
            self.ax.set_aspect('equal', adjustable='box') # 保持横纵轴比例一致，防止地图变形
            self.ax.set_xlim(0, self.map_shape[1]) # 设置 X 轴范围
            self.ax.set_ylim(0, self.map_shape[0]) # 设置 Y 轴范围
            plt.show(block=False) # 非阻塞模式显示

        # --- 更新绘图元素 ---
        # 更新机器人中心点的位置 (注意 plot 使用 x, y 对应 col, row)
        self.robot_handle.set_data(self.robot_pos[1], self.robot_pos[0])

        # 更新机器人朝向线的位置和方向
        start_col = self.robot_pos[1]
        start_row = self.robot_pos[0]
        end_col = start_col + line_length * np.cos(self.robot_theta)
        end_row = start_row + line_length * np.sin(self.robot_theta)
        self.orient_handle.set_data([start_col, end_col], [start_row, end_row])

        # 更新地图显示 - 无论地图尺寸是否改变，都更新内容
        self.map_handle.set_data(1 - self.ground_truth)
        current_extent = [0, self.map_shape[1], 0, self.map_shape[0]]
        if self.map_handle.get_extent() != current_extent:
             # 如果地图尺寸变化，也需要更新extent
             self.map_handle.set_extent(current_extent)
             # 如果地图尺寸变化，需要重新调整坐标轴范围
             self.ax.set_xlim(0, self.map_shape[1])
             self.ax.set_ylim(0, self.map_shape[0])
             self.ax.set_aspect('equal', adjustable='box')

        # --- 刷新画布 ---
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception as e:
            print(f"渲染时发生错误: {e}")
            # 可以选择关闭图形或尝试重新初始化
            # self.close() # 强制关闭可能导致后续无法渲染

        # --- 处理返回或暂停 ---
        if capture:
            # 如果需要返回 RGB 数组
            buf = self.fig.canvas.buffer_rgba()
            rgb_array = np.asarray(buf)
            return rgb_array[:, :, :3] # 返回 RGB 部分
        else:
            # 在 'human' 模式下暂停，以便能看清动画
            # 使用 max(0.001, ...) 防止除零错误或暂停时间过长
            plt.pause(max(0.001, 1.0 / self.metadata['render_fps']))

    def close(self):
        """关闭环境，清理资源 (例如 Matplotlib 图形窗口)。"""
        if self.fig is not None:
            print("正在关闭 Matplotlib 窗口...")
            plt.close(self.fig)
            plt.ioff() # 关闭交互模式
            # 清理句柄变量
            self.fig = None
            self.ax = None
            self.map_handle = None
            self.robot_handle = None
            self.orient_handle = None
        print("Environment closed.")

# --- 主程序入口：示例用法 ---
if __name__ == '__main__':
    print("Setting up environment...")
    # --- 创建虚拟地图文件 (如果 'maps' 目录或文件不存在) ---
    map_dir_name = 'maps'
    if not os.path.exists(map_dir_name):
        os.makedirs(map_dir_name)
        print(f"Directory '{map_dir_name}' created.")


    # --- 运行环境测试 ---
    env = None # 初始化为 None，以便在 finally 中安全关闭
    try:
        # 实例化环境，使用 'human' 模式进行可视化
        env = NavigationEnv(map_dir=map_dir_name,
                            render_mode='human',
                            max_episode_steps=300,) # 减少最大步数以便快速测试)         # 设置渲染帧率

        num_episodes = 3 # 测试回合数
        for episode in range(num_episodes):
            print(f"\n--- start episode {episode + 1} ---")
            # 重置环境，传入种子以保证可复现性
            obs, info = env.reset(seed=episode)
            terminated = False
            truncated = False
            total_reward = 0
            step_count = 0
            print(f"initial state [row, col, theta]: {obs}")

            # 回合循环，直到终止或截断
            while not terminated and not truncated:
                # 从动作空间中随机采样一个动作 [线速度, 角速度]
                action = env.action_space.sample()
                action = [10, 1]
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                step_count += 1

                # 可以取消注释下面的打印语句来查看每一步的详细信息
                # print(f"步: {step_count}, 动作: [{action[0]:.2f}, {action[1]:.2f}], 观测: [{obs[0]:.2f}, {obs[1]:.2f}, {obs[2]:.2f}], 奖励: {reward:.2f}")

                # 在 human 模式下，render() 会在 step() 内部被调用
                # time.sleep(0.01) # 如果动画太快，可以取消注释加入微小延迟

            # 回合结束信息
            print(f"Episode {episode + 1} ends in {step_count} steps.")
            print(f"Total reward: {total_reward:.2f}")
            print(f"Final state [row, col, theta]: {obs}")
            if terminated: print("End reason: collision")
            if truncated: print("End reason: max steps")

        print("\nEnvironment test completed.")

    except FileNotFoundError as e:
        # 处理文件未找到错误
        print(f"\nError initializing environment: {e}")
        print(f"Please ensure '{map_dir_name}' directory exists and contains valid map image files.")
    except ValueError as e:
        # 处理值错误，例如找不到自由空间
         print(f"\nError occurred during test: {e}")
         traceback.print_exc() # 打印详细的回溯信息
    except Exception as e:
        # 处理其他所有意外错误
         print(f"\nUnexpected error occurred during test:")
         traceback.print_exc() # 打印详细的回溯信息
    finally:
        # 确保无论是否发生错误，都尝试关闭环境（清理渲染窗口等）
        if env is not None:
            env.close()