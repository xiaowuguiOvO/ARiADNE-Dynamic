import torch

from env import Env
from agent import Agent
from utils import *
from model import PolicyNet

if not os.path.exists(gifs_path):
    os.makedirs(gifs_path)


class Worker:
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', save_image=False):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = save_image
        self.device = device

        self.env = Env(global_step, plot=self.save_image)
        self.robot = Agent(policy_net, self.device, self.save_image)
        self.robot.env = self.env
        self.episode_buffer = []
        self.perf_metrics = dict()
        for i in range(15):
            self.episode_buffer.append([])

    # def run_episode(self):
    #     done = False
    #     self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
    #     observation = self.robot.get_observation()

    #     if self.save_image:
    #         self.robot.plot_env()
    #         self.env.plot_env(0)

    #     for i in range(MAX_EPISODE_STEP):

    #         self.save_observation(observation)

    #         next_location, action_index = self.robot.select_next_waypoint(observation)
    #         self.save_action(action_index)

    #         node = self.robot.node_manager.nodes_dict.find((self.robot.location[0], self.robot.location[1]))
    #         check = np.array(list(node.data.neighbor_set)).reshape(-1, 2)
    #         assert next_location[0] + next_location[1] * 1j in check[:, 0] + check[:, 1] * 1j, print(next_location, self.robot.location, node.data.neighbor_set)
    #         assert next_location[0] != self.robot.location[0] or next_location[1] != self.robot.location[1]

    #         reward = self.env.step(next_location)

    #         self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
    #         if self.robot.utility.sum() == 0:
    #             done = True
    #             reward += 20
    #         self.save_reward_done(reward, done)

    #         observation = self.robot.get_observation()
    #         self.save_next_observations(observation)

    #         if self.save_image:
    #             self.robot.plot_env()
    #             self.env.plot_env(i+1)

    #         if done:
    #             break

    #     # save metrics
    #     self.perf_metrics['travel_dist'] = self.env.travel_dist
    #     self.perf_metrics['explored_rate'] = self.env.explored_rate
    #     self.perf_metrics['success_rate'] = done
    #     self.perf_metrics['collision_count'] = self.env.collision_count

    #     # save gif
    #     if self.save_image:
    #         make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)
    def run_episode(self):
        # 初始化环境和机器人状态
        done = False
        need_decision = True
        simulation_time = 0.0
        self.env.set_agent(self.robot)
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        
        # 初始化可视化设置
        if self.save_image:
            self.robot.plot_env()
            self.env.plot_env(0)
        
        # 主循环
        max_simulation_time = MAX_EPISODE_TIME
        step_count = 0
        
        # # 初始化机器人速度
        # if need_decision:
        #     observation = self.robot.get_observation()
        #     self.save_observation(observation)
        #     velocity = self.robot.cal_next_velocity(observation)
        #     self.save_velocity(velocity)
        #     self.robot.velocity = velocity
        
        while simulation_time < max_simulation_time and step_count < MAX_EPISODE_STEP and not done:
            # 环境始终在执行
            reward, collision, need_decision = self.env.step()
            step_count += 1
            simulation_time += self.env.step_size
            
            # 只在需要决策时更新速度
            if need_decision:
                observation = self.robot.get_observation()
                self.save_observation(observation)
                
                # 计算新的速度命令
                velocity = self.robot.cal_next_velocity(observation)
                self.save_velocity(velocity)
                self.robot.velocity = velocity
                
                # 保存观察和下一个状态（用于训练）
                next_observation = self.robot.get_observation()
                self.save_next_observations(next_observation)
            
            # 更新机器人规划状态
            self.robot.update_planning_state_use_nearest_node(self.env.belief_info, self.env.robot_location)
            
            # 检查是否完成探索
            if self.robot.utility.sum() == 0:
                done = True
                reward += 20
            
            # 保存奖励和完成状态
            self.save_reward_done(reward, done or collision)
            
            # 可视化
            if self.save_image and (need_decision or done or collision or step_count % VISUALIZATION_INTERVAL == 0):
                self.robot.plot_env()
                self.env.plot_env(step_count)
        
        # 保存性能指标
        self.perf_metrics['travel_dist'] = self.env.travel_dist
        self.perf_metrics['explored_rate'] = self.env.explored_rate
        self.perf_metrics['success_rate'] = 1 if done and not collision else 0
        self.perf_metrics['collision_count'] = self.env.collision_count
        self.perf_metrics['simulation_time'] = simulation_time
        
        # 保存GIF
        if self.save_image:
            make_gif(gifs_path, self.global_step, self.env.frame_files, self.env.explored_rate)
            
    def save_observation(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[0] += node_inputs
        self.episode_buffer[1] += node_padding_mask.bool()
        self.episode_buffer[2] += edge_mask.bool()
        self.episode_buffer[3] += current_index
        self.episode_buffer[4] += current_edge
        self.episode_buffer[5] += edge_padding_mask.bool()

    def save_action(self, velocity):
        action_tensor = torch.tensor(velocity).reshape(1, 1, 1)
        self.episode_buffer[6] += action_index.reshape(1, 1, 1)
        
    def save_velocity(self, velocity_vector):
        """保存速度向量到经验缓冲区"""
        if not isinstance(velocity_vector, torch.Tensor):
            velocity_vector = torch.tensor(velocity_vector, device=self.device)
        self.episode_buffer[6] += velocity_vector.reshape(1, 2, 1)
        
    def save_reward_done(self, reward, done):
        self.episode_buffer[7] += torch.FloatTensor([reward]).reshape(1, 1, 1).to(self.device)
        self.episode_buffer[8] += torch.tensor([int(done)]).reshape(1, 1, 1).to(self.device)

    def save_next_observations(self, observation):
        node_inputs, node_padding_mask, edge_mask, current_index, current_edge, edge_padding_mask = observation
        self.episode_buffer[9] += node_inputs
        self.episode_buffer[10] += node_padding_mask.bool()
        self.episode_buffer[11] += edge_mask.bool()
        self.episode_buffer[12] += current_index
        self.episode_buffer[13] += current_edge
        self.episode_buffer[14] += edge_padding_mask.bool()


if __name__ == "__main__":
    torch.manual_seed(4777)
    np.random.seed(4777)
    model = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)
    # checkpoint = torch.load(model_path + '/checkpoint.pth', map_location='cpu')
    # model.load_state_dict(checkpoint['policy_model'])
    worker = Worker(0, model, 78, save_image=True)
    worker.run_episode()
