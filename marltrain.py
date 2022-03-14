import scipy.io
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import Environment_marl
import random
import os
from replay_memory import ReplayMemory

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ################## SETTINGS ######################
up_lanes = [i/2.0 for i in [3.5/2, 3.5/2 + 3.5, 250+3.5/2, 250+3.5+3.5/2, 500+3.5/2, 500+3.5+3.5/2]]
down_lanes = [i/2.0 for i in [250-3.5-3.5/2, 250-3.5/2, 500-3.5-3.5/2, 500-3.5/2, 750-3.5-3.5/2, 750-3.5/2]]
left_lanes = [i/2.0 for i in [3.5/2, 3.5/2 + 3.5, 433+3.5/2, 433+3.5+3.5/2, 866+3.5/2, 866+3.5+3.5/2]]
right_lanes = [i/2.0 for i in [433-3.5-3.5/2, 433-3.5/2, 866-3.5-3.5/2, 866-3.5/2, 1299-3.5-3.5/2, 1299-3.5/2]]
width = 750/2
height = 1298/2
label = 'marl_model'
n_veh = 4 # For the number of the vehicles in each direction remain same, n_veh % 4 == 0
n_neighbor = 1
n_RB = n_veh
# Environment Parameters
env = Environment_marl.Environ(down_lanes, up_lanes, left_lanes, right_lanes, width, height, n_veh, n_neighbor)
env.new_random_game()   # initialize parameters in env
n_episode = 3000
n_step_per_episode = int(env.time_slow/env.time_fast)
epsi_final = 0.02
epsi_anneal_length = int(0.85*n_episode)
mini_batch_step = n_step_per_episode
target_update_step = n_step_per_episode*4
n_episode_test = 100  # test episodes
n_hidden_1 = 500
n_hidden_2 = 250
n_hidden_3 = 120
######################################################


def get_state(env, idx=(0, 0), ind_episode=1., epsi=0.02):
    """ Get state from the environment """

    # V2I_channel = (env.V2I_channels_with_fastfading[idx[0], :] - 80) / 60
    V2I_fast = (env.V2I_channels_with_fastfading[idx[0], :] - env.V2I_channels_abs[idx[0]] + 10)/35

    # V2V_channel = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - 80) / 60
    V2V_fast = (env.V2V_channels_with_fastfading[:, env.vehicles[idx[0]].destinations[idx[1]], :] - env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] + 10)/35

    V2V_interference = (-env.V2V_Interference_all[idx[0], idx[1], :] - 60) / 60

    V2I_abs = (env.V2I_channels_abs[idx[0]] - 80) / 60.0
    V2V_abs = (env.V2V_channels_abs[:, env.vehicles[idx[0]].destinations[idx[1]]] - 80)/60.0

    load_remaining = np.asarray([env.demand[idx[0], idx[1]] / env.demand_size])
    time_remaining = np.asarray([env.individual_time_limit[idx[0], idx[1]] / env.time_slow])

    return np.concatenate((V2I_fast, np.reshape(V2V_fast, -1), V2V_interference, np.asarray([V2I_abs]), V2V_abs, time_remaining, load_remaining, np.asarray([ind_episode, epsi])))


n_input_size = len(get_state(env=env))
n_output_size = n_RB * len(env.V2V_power_dB_List)


class DQN(nn.Module):
    def __init__(self, input_size, n_hidden1, n_hidden2, n_hidden3, output_size):
        super(DQN, self).__init__()
        self.fc_1 = nn.Linear(input_size, n_hidden1)
        self.fc_1.weight.data.normal_(0, 0.1)
        self.fc_2 = nn.Linear(n_hidden1, n_hidden2)
        self.fc_2.weight.data.normal_(0, 0.1)
        self.fc_3 = nn.Linear(n_hidden2, n_hidden3)
        self.fc_3.weight.data.normal_(0, 0.1)
        self.fc_4 = nn.Linear(n_hidden3, output_size)
        self.fc_4.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc_1(x))
        x = F.relu(self.fc_2(x))
        x = F.relu(self.fc_3(x))
        x = F.relu(self.fc_4(x))
        return x

class Agent:
    def __init__(self, memory_entry_size):
        self.discount = 1
        self.double_q = False
        self.memory_entry_size = memory_entry_size
        self.memory = ReplayMemory(self.memory_entry_size)
        self.model = DQN(n_input_size, n_hidden_1, n_hidden_2, n_hidden_3, n_output_size).to(device)
        # if torch.cuda.device_count()>1:
        #    self.model = nn.DataParallel(self.model)
        # self.model.to(device)
        self.target_model = DQN(n_input_size, n_hidden_1, n_hidden_2, n_hidden_3, n_output_size).to(device)  # Target Model
        self.target_model.eval()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=0.001, momentum=0.05, eps=0.01)
        self.loss_func = nn.MSELoss()

    def predict(self, s_t, ep=0.):
        n_power_levels = len(env.V2V_power_dB_List)
        # state_t = torch.from_numpy(s_t).type(torch.float32).view([1, self.memory_entry_size])
        if random.random() > ep:
            with torch.no_grad():
                q_values = self.model(torch.tensor(s_t, dtype=torch.float32).unsqueeze(0).to(device))
                return q_values.max(1)[1].item()
        else:
            return random.choice(range(n_power_levels))

    def Q_Learning_mini_batch(self):  # Double Q-Learning
        batch_s_t, batch_s_t_plus_1, batch_action, batch_reward = self.memory.sample()
        action = torch.LongTensor(batch_action).to(device)
        reward = torch.FloatTensor(batch_reward).to(device)
        state = torch.FloatTensor(np.float32(batch_s_t)).to(device)
        next_state = torch.FloatTensor(np.float32(batch_s_t_plus_1)).to(device)
        if self.double_q:
            next_action = self.model(next_state).max(1)[1]
            next_q_values = self.target_model(next_state)
            next_q_value = next_q_values.gather(1, next_action.unsqueeze(1)).squeeze(1)
            expected_q_value = reward + self.discount * next_q_value
        else:
            next_q_value = self.target_model(next_state).max(1)[0]
            expected_q_value = reward + self.discount * next_q_value
        q_values = self.model(state)
        q_acted = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        loss = self.loss_func(expected_q_value.detach(), q_acted)
        # backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def save_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        torch.save(self.model.state_dict(), model_path+'.ckpt')
        torch.save(self.target_model.state_dict(), model_path+'_t.ckpt')

    def load_models(self, model_path):
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, "model/" + model_path)
        self.model.load_state_dict(torch.load(model_path + '.ckpt'))
        self.target_model.load_state_dict(torch.load(model_path + '_t.ckpt'))


# ----------------------------------------------------------------------------
print(device)
agents = []
for ind_agent in range(n_veh * n_neighbor):  # initialize agents
    print("Initializing agent", ind_agent)
    agent = Agent(memory_entry_size=len(get_state(env)))
    agents.append(agent)
# ----------------------------Training----------------------------------------
record_reward = np.zeros([n_episode*n_step_per_episode, 1])
record_loss = []
for i_episode in range(n_episode):
    if i_episode < epsi_anneal_length:
        epsi = 1 - i_episode * (1 - epsi_final) / (epsi_anneal_length - 1)  # epsilon decreases over each episode
    else:
        epsi = epsi_final
    if i_episode % 100 == 0:
        env.renew_positions()  # update vehicle position
        env.renew_neighbor()
        env.renew_channel()  # update channel slow fading
        env.renew_channels_fastfading()  # update channel fast fading

    env.demand = env.demand_size * np.ones((env.n_Veh, env.n_neighbor))
    env.individual_time_limit = env.time_slow * np.ones((env.n_Veh, env.n_neighbor))
    env.active_links = np.ones((env.n_Veh, env.n_neighbor), dtype='bool')

    for i_step in range(n_step_per_episode):
        time_step = i_episode * n_step_per_episode + i_step
        state_old_all = []
        action_all = []
        action_all_training = np.zeros([n_veh, n_neighbor, 2], dtype='int32')
        for i in range(n_veh):
            for j in range(n_neighbor):
                state = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                state_old_all.append(state)
                action = agents[i * n_neighbor + j].predict(state, epsi)
                action_all.append(action)

                action_all_training[i, j, 0] = action % n_RB  # chosen RB
                action_all_training[i, j, 1] = int(np.floor(action / n_RB))  # power level

        # All agents take actions simultaneously, obtain shared reward, and update the environment.
        action_temp = action_all_training.copy()
        train_reward = env.act_for_training(action_temp)
        record_reward[time_step] = train_reward

        env.renew_channels_fastfading()
        env.Compute_Interference(action_temp)

        for i in range(n_veh):
            for j in range(n_neighbor):
                state_old = state_old_all[n_neighbor * i + j]
                action = action_all[n_neighbor * i + j]
                state_new = get_state(env, [i, j], i_episode / (n_episode - 1), epsi)
                # add entry to this agent's memory
                agents[i * n_neighbor + j].memory.add(state_old, state_new, train_reward, action)

                # training this agent
                if time_step % mini_batch_step == mini_batch_step - 1:
                    loss_val_batch = agents[i * n_neighbor + j].Q_Learning_mini_batch()
                    record_loss.append(loss_val_batch)
                    if i_episode % 100 == 0 and i == 0 and j == 0:
                        print('Episode:', i_episode, 'agent0_loss', loss_val_batch)
                if time_step % target_update_step == target_update_step - 1:
                    agents[i * n_neighbor + j].update_target_network()
                    # if i == 0 and j == 0:
                    #    print('Update target Q network...')
print('Training Done. Saving models...')
for i in range(n_veh):
    for j in range(n_neighbor):
        model_path = label + '/agent_' + str(i * n_neighbor + j)
        agents[i * n_neighbor + j].save_models(model_path)

current_dir = os.path.dirname(os.path.realpath(__file__))
reward_path = os.path.join(current_dir, "model/" + label + '/reward.mat')
scipy.io.savemat(reward_path, {'reward': record_reward})

record_loss = np.asarray(record_loss).reshape((-1, n_veh * n_neighbor))
loss_path = os.path.join(current_dir, "model/" + label + '/train_loss.mat')
scipy.io.savemat(loss_path, {'train_loss': record_loss})

