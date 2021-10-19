#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.engine import training
# import cv2

from gazebo_env_four_qcar_link1 import envmodel1

env = envmodel1()

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class ReplayBuffer:
    def __init__(self, max_size, input_x, input_y, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_x, input_y))
        self.new_state_memory = np.zeros((self.mem_size, input_x, input_y))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class CriticNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256,
                 name='critic', chkpt_dir='tmp/sac', training=True):
        super(CriticNetwork, self).__init__()

        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_y = 365

        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_sac')

        self.training = training

        f1 = 1. / np.sqrt(self.input_y)
        f2 = 1. / np.sqrt(self.fc1_dims)
        fa = 1. / np.sqrt(1)

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        # self.batch1 = layers.BatchNormalization()

        self.relu = layers.Activation('relu')

        self.lstm = tf.keras.layers.LSTM(self.input_y)

        self.fc1 = Dense(self.fc1_dims, kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                         bias_initializer=tf.random_uniform_initializer(-f1, f1))

        self.action_in = Dense(self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-fa, fa),
                               bias_initializer=tf.random_uniform_initializer(-fa, fa), activation='relu')

        self.fc2 = Dense(self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                         bias_initializer=tf.random_uniform_initializer(-f2, f2))

        self.q = Dense(1, activation=None, kernel_initializer=last_init,
                       bias_initializer=last_init)

    def call(self, state, action):

        x = self.lstm(state)  # (32,512) + (32,1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        action_in = self.action_in(action)
        action_value = tf.concat([x, action_in], axis=1)
        #action_value = self.batch3(action_value, training=self.training)
        #action_value = self.relu(action_value)
        #action_value = self.fc3(action_value)
        #action_value = self.batch4(action_value, training=self.training)
        #action_value = self.relu(action_value)

        q = self.q(action_value)

        return q

# fc1= 400 , fc2= 300


class ValueNetwork(keras.Model):
    def __init__(self, fc1_dims=256, fc2_dims=256,
                 name='value', chkpt_dir='tmp/sac', training=True):
        super(ValueNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.input_y = 365
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_sac')
        self.training = training

        f1 = 1. / np.sqrt(self.input_y)
        f2 = 1. / np.sqrt(self.fc1_dims)

        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        self.relu = layers.Activation('relu')

        self.lstm = tf.keras.layers.LSTM(self.input_y)

        self.fc1 = Dense(self.fc1_dims, kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                         bias_initializer=tf.random_uniform_initializer(-f1, f1), activation='relu')

        self.fc2 = Dense(self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                         bias_initializer=tf.random_uniform_initializer(-f2, f2), activation='relu')

        self.v = Dense(1, activation=None, kernel_initializer=last_init,
                       bias_initializer=last_init)

    def call(self, state):
        state_value = self.lstm(state)  # (32,512) + (32,1)
        state_value = self.fc1(state_value)
        state_value = self.fc2(state_value)

        v = self.v(state_value)

        return v


class ActorNetwork(keras.Model):
    def __init__(self, max_action, fc1_dims=256, fc2_dims=256, n_actions=1, name='actor',
                 chkpt_dir='tmp/sac', training=True):
        super(ActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir,
                                            self.model_name+'_sac')
        self.noise = 1e-6
        self.max_action = max_action

        self.training = training
        self.input_y = 365
        f1 = 1. / np.sqrt(self.input_y)
        f2 = 1. / np.sqrt(self.fc1_dims)
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        self.lstm = layers.LSTM(self.input_y)
        self.relu = layers.Activation('relu')
        #self.sigmoid = layers.Activation('sigmoid')

        #self.batch1 = layers.BatchNormalization()

        self.fc1 = Dense(self.fc1_dims, kernel_initializer=tf.random_uniform_initializer(-f1, f1),
                         bias_initializer=tf.random_uniform_initializer(-f1, f1))
        self.fc2 = Dense(self.fc2_dims, kernel_initializer=tf.random_uniform_initializer(-f2, f2),
                         bias_initializer=tf.random_uniform_initializer(-f2, f2))
        self.mu = Dense(
            self.n_actions, kernel_initializer=last_init, bias_initializer=last_init)
        self.sigma = Dense(
            self.n_actions, kernel_initializer=last_init, bias_initializer=last_init)

    def call(self, state):

        # print(np.array(state).shape)
        x = self.lstm(state)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        mu = self.mu(x)
        sigma = self.sigma(x)
        # might want to come back and change this, perhaps tf plays more nicely with
        # a sigma of ~0
        sigma = tf.clip_by_value(sigma, self.noise, 1)

        return mu, sigma

    def sample_normal(self, state, reparameterize=True):
        mu, sigma = self.call(state)
        probabilities = tfp.distributions.Normal(mu, sigma)

        if reparameterize:
            actions = probabilities.sample()  # + something else if you want to implement
        else:
            actions = probabilities.sample()

        action = tf.math.sigmoid(actions)*self.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action, 2)+self.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs


class SAC:
    def __init__(self, input_x=4, input_y=365, alpha=0.0003, beta=0.0003,
                 gamma=0.90, n_actions=1, max_size=100000, tau=0.001,
                 batch_size=128, noise=0.2):

        # Define State Space and Action Space
        self.input_x = input_x  # stackFrame
        self.input_y = input_y  # 360 + 2 self state + 3 velocities
        self.n_actions = n_actions
        self.max_action = 2.0
        self.min_action = 0.0

        # Define Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.alpha = alpha
        self.beta = beta
        self.noise = noise

        # Define ReplayBuffer Information
        self.memory = ReplayBuffer(max_size, input_x, input_y, n_actions)
        self.batch_size = batch_size

        # Define Training Information
        self.algorithm = 'SAC'
        self.Number = 'train1'
        self.progress = ''
        self.load_path = '/home/sdcnlab025/ROS_test/three_qcar/src/tf_pkg/scripts/saved_networks/two_qcar_links_2021-06-08_test2/qcar1'
        self.step = 1
        self.score = 0
        self.episode = 0
        self.isTraining = True

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())
        # Define and Create Saved Location
        self.save_location = 'saved_networks/' + 'two_qcar_links_' + \
            self.date_time + '_' + self.Number + '/qcar1'
        os.makedirs(self.save_location)

        # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4
        # input image size
        self.img_size = 80

        # Define Network Information
        self.actor = ActorNetwork(
            n_actions=n_actions, chkpt_dir=self.save_location, max_action=self.max_action)
        self.critic_1 = CriticNetwork(
            name='critic_1', chkpt_dir=self.save_location)
        self.critic_2 = CriticNetwork(
            name='critic_2', chkpt_dir=self.save_location)
        self.value = ValueNetwork(chkpt_dir=self.save_location)
        self.target_value = ValueNetwork(
            chkpt_dir=self.save_location, name='target_value')

        self.actor.compile(optimizer=Adam(lr=alpha))
        self.critic_1.compile(optimizer=Adam(lr=beta))
        self.critic_2.compile(optimizer=Adam(lr=beta))
        self.value.compile(optimizer=Adam(lr=beta))
        self.target_value.compile(optimizer=Adam(lr=beta))

        self.scale = 2

        self.update_network_parameters(tau=1)

        self.init_sess()

    @tf.function
    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_weights(self.actor.checkpoint_file)
        self.critic_1.save_weights(self.critic_1.checkpoint_file)
        self.critic_2.save_weights(self.critic_2.checkpoint_file)
        self.value.save_weights(self.value.checkpoint_file)
        self.target_value.save_weights(self.target_value.checkpoint_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_load_path = os.path.join(self.load_path, 'actor_sac')
        self.critic_1_load_path = os.path.join(self.load_path, 'critic_1_sac')
        self.critic_2_load_path = os.path.join(self.load_path, 'critic_2_sac')
        self.value_load_path = os.path.join(
            self.load_path, 'value_sac')
        self.target_value_load_path = os.path.join(
            self.load_path, 'target_value_sac')

        self.actor.load_weights(self.actor_load_path)
        self.critic_1.load_weights(self.critic_1_load_path)
        self.critic_2.load_weights(self.critic_2_load_path)
        self.value.load_weights(self.value_load_path)
        self.target_value.load_weights(self.target_value_load_path)

    def input_initialization(self, env_info):
        state = env_info[0]  # laser info + self state
        state_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.input_y))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
                        :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        observation = env_info[1]  # image info
        observation_set = []
        for i in range(self.Num_skipFrame * self.Num_stackFrame):
            observation_set.append(observation)
            # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        # print("shape of observation stack={}".format(observation_stack.shape))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        observation_stack = np.uint8(observation_stack)

        return observation_stack, observation_set, state_stack, state_set

    # Resize input information
    def resize_input(self, env_info, observation_set, state_set):

        observation = env_info[1]
        observation_set.append(observation)
        # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
        observation_stack = np.zeros(
            (self.img_size, self.img_size, self.Num_stackFrame))
        for stack_frame in range(self.Num_stackFrame):
            observation_stack[:, :, stack_frame] = observation_set[-1 -
                                                                   (self.Num_skipFrame * stack_frame)]
        del observation_set[0]
        observation_stack = np.uint8(observation_stack)

        state = env_info[0]
        state_set.append(state)
        state_stack = np.zeros((self.Num_stackFrame, self.input_y))
        for stack_frame in range(self.Num_stackFrame):
            state_stack[(self.Num_stackFrame - 1) - stack_frame,
                        :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

        del self.state_set[0]

        return observation_stack, observation_set, state_stack, state_set

    def move(self, cmd=[0.0, 0.0]):
        env.step(cmd)

    def accelerate(self, accel):
        env.accel(accel)

    def update_path(self, path):
        self.path = path
        env.update_path(self.path)

    def init_sess(self):
        # Load the file if the saved file exists
        self.isTraining = True
        check_save = input('Load Model for Link 1? (1=yes/2=no): ')
        if check_save == 1:
            # Restore variables from disk.
            self.load_models()
            print("Link 1 model restored.")

            check_train = input(
                'Inference or Training? (1=Inference / 2=Training): ')
            if check_train == 1:
                self.isTraining = False
                self.Num_start_training = 0
                self.Num_training = 0

    def new_environment(self):

        # plt.scatter(self.episode, self.score, c='r')
        # plt.xlabel("Episode")
        # plt.ylabel("Reward")
        # plt.xlim(-1, ((self.episode/20 + 1)*20))
        # plt.ylim(-12, 3)
        # plt.pause(0.01)  # 0.05

        if self.progress != 'Observing':
            self.reward_list.append(self.score)
            self.reward_array = np.array(self.reward_list)
            # ------------------------------
            np.savetxt(self.save_location + '/qcar1_reward.txt',
                       self.reward_array, delimiter=',')
            if self.episode % 25 == 0:
                avg_score = np.mean(self.reward_list[-25:])
                self.avg_list.append(avg_score)
                print('___________Avgerage score is_____________', avg_score)
                plt.scatter(self.episode, avg_score, c='b')
                plt.xlabel("Episode")
                plt.ylabel("Average Reward")
                plt.xlim(-1, ((self.episode/20 + 1)*20))
                plt.ylim(-10, 3)
                plt.pause(0.05)
                self.avg_array = np.array(self.avg_list)
                np.savetxt(self.save_location + '/qcar1_25avg_score.txt',
                           self.avg_array, delimiter=',')
            self.episode += 1

        self.score = 0.0
        self.reward = 0.0
        env_info = env.get_env()
        self.observation_stack, self.observation_set, self.state_stack, self.state_set = self.input_initialization(
            env_info)

    def main_func(self):

        self.reward_list = []
        self.avg_list = []

        np.random.seed(1000)

        tf.random.set_seed(520)

        env_info = env.get_env()
        # env.info为4维，第1维为相机消息，第2维为agent robot的self state，第3维为terminal，第4维为reward
        self.observation_stack, self.observation_set, self.state_stack, self.state_set = self.input_initialization(
            env_info)

        self.step_for_newenv = 0

    def update_parameters(self, Num_start_training, Num_training, Num_test, learning_rate, gamma, MAXEPISODES):

        self.Num_start_training = Num_start_training
        self.Num_training = Num_training
        self.Num_test = Num_test
        self.learning_rate = learning_rate
        self.gamma = gamma

    def get_progress(self, step):

        if step <= self.Num_start_training:
            # Obsersvation
            progress = 'Observing'

        elif step <= self.Num_start_training + self.Num_training:
            # Training
            progress = 'Training'
            # self.actor.training = True
            # self.value.training = True
            # self.critic_1.training = True
            # self.critic_2.training = True

        elif step < self.Num_start_training + self.Num_training + self.Num_test:
            # Testing
            progress = 'Testing'
            # print('_________________start testing___________________')
            # self.actor.training = False
            # self.value.training = False
            # self.critic_1.training = False
            # self.critic_2.training = False

        else:
            # Finished
            progress = 'Finished'

        self.progress = progress

        return progress

    def save_fig(self):
        plt.savefig(self.save_location + '/qcar1_reward.png')
        plt.show()
# *****************************************Update information*********************************************************************

    def update_information(self):
        # Get information for update
        env_info = env.get_env()

        self.next_observation_stack, self.observation_set, self.next_state_stack, self.state_set = self.resize_input(
            env_info, self.observation_set, self.state_set)  # 调整输入信息
        terminal = env_info[-2]  # 获取terminal
        self.reward = env_info[-1]  # 获取reward

        if self.progress == 'Training' or self.progress == 'Observing':
            self.memory.store_transition(
                self.state_stack, self.action, self.reward, self.next_state_stack, terminal)

        if self.progress == 'Training':
            self.learn()

        # Update information
        self.step += 1
        #print('reward is',self.reward)
        self.score += self.reward
        self.observation_stack = self.next_observation_stack
        self.state_stack = self.next_state_stack
        self.step_for_newenv += 1

        return terminal
# *****************************************Select Action*********************************************************************

    def select_action(self, state_stack, progress):

        state = tf.convert_to_tensor([state_stack], dtype=tf.float32)
        #print('state_stack:__________',state_stack)
        actions, _ = self.actor.sample_normal(state,reparameterize=False)
        #print('action', actions)

        if progress == 'Training':
            actions += tf.random.normal(shape=[self.n_actions, 1],
                                        mean=0.0, stddev=self.noise)
        elif progress == "Observing":
            actions = tf.random.normal(shape=[self.n_actions, 1],
                                       mean=1.0, stddev=1.0)

        # note that if the environment has an action > 1, we have to multiply by max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        
        return actions[0][0].numpy()  # Exract the value

    def return_action(self):
        self.action = self.select_action(
            self.state_stack, self.progress)
        return self.action

    @tf.function
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = self.actor.sample_normal(states,
                                                                         reparameterize=False)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, current_policy_actions)
            q2_new_policy = self.critic_2(states, current_policy_actions)
            critic_value = tf.squeeze(
                tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss,
                                               self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(
            value_network_gradient, self.value.trainable_variables))

        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this so it's just the usual action.
            new_policy_actions, log_probs = self.actor.sample_normal(states,
                                                                     reparameterize=True)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(states, new_policy_actions)
            q2_new_policy = self.critic_2(states, new_policy_actions)
            critic_value = tf.squeeze(tf.math.minimum(
                q1_new_policy, q2_new_policy), 1)

            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss,
                                               self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(
            actor_network_gradient, self.actor.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
            q_hat = self.scale*reward + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1(states, actions), 1)
            q2_old_policy = tf.squeeze(self.critic_2(states, actions), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)

        critic_1_network_gradient = tape.gradient(critic_1_loss,
                                                  self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss,
                                                  self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(
            critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(
            critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()

    def print_information(self):
        print('[Link1-'+self.progress+'] step:'+str(self.step)+'/episode:' +
              str(self.episode)+'/path:'+self.path+'/score:' + str(self.score))

    def set_velocity(self, v2, v3, v4, s2, s3, s4):
        env.set_velocity(v2, v3, v4, s2, s3, s4)


# if __name__ == '__main__':
#     agent = SAC()
#     agent.main()
