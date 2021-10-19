#!/usr/bin/env python
# -*- coding: utf-8 -*-


from gazebo_env_two_qcar_link1 import envmodel1
import time
import matplotlib.pyplot as plt
import datetime
import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = '2' 
env = envmodel1()


gpus = tf.config.list_physical_devices('GPU')
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


env = envmodel1()


class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) *
            np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class ReplayBuffer:
    def __init__(self, max_size=100000, input_y=362, n_actions=1):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, input_y))
        self.new_state_memory = np.zeros((self.mem_size, input_y))
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


class DDPG:
    def __init__(self):

        self.input_y = 362  # 360 + 2 self state
        self.batch_size = 32
        self.n_actions = 1


        self.max_action = 1
        self.min_action = 0

        self.memory = ReplayBuffer(input_y=self.input_y,n_actions=self.n_actions)

        self.algorithm = 'DDPG'

        self.Number = 'test1'

        # Get parameters
        self.progress = ''
        self.load_path = '/home/sdcnlab/ROS_test/three_qcar/src/tf_pkg/scripts/saved_networks/two_qcar_links_2021-06-01_train1/qcar1/'
        self.step = 1
        self.score = 0
        self.episode = 0
        self.isTraining = True

        # date - hour - minute - second of training time
        self.date_time = str(datetime.date.today())

        self.save_location = 'saved_networks/' + 'two_qcar_links_' + \
            self.date_time + '_' + self.Number + '/qcar1'

        std_dev = 0.2

        self.ou_noise = OUActionNoise(mean=np.zeros(
                1), std_deviation=float(std_dev) * np.ones(1))

        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor(name='target_actor')
        self.target_critic = self.get_critic(name='target_critic')

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        self.critic_lr = 0.002
        self.actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)

        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        self.tau = 0.005

                # parameters for skipping and stacking
        self.Num_skipFrame = 1
        self.Num_stackFrame = 4


        self.init_sess()

    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # def input_initialization(self, env_info):
    #     state = env_info[0]  # laser info + self state
    #     state_set = []
    #     for i in range(self.Num_skipFrame * self.Num_stackFrame):
    #         state_set.append(state)
    #     state_stack = np.zeros((self.Num_stackFrame, self.input_y))
    #     for stack_frame in range(self.Num_stackFrame):
    #         state_stack[(self.Num_stackFrame - 1) - stack_frame,
    #                     :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

    #     observation = env_info[1]  # image info
    #     observation_set = []
    #     for i in range(self.Num_skipFrame * self.Num_stackFrame):
    #         observation_set.append(observation)
    #         # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
    #     observation_stack = np.zeros(
    #         (self.img_size, self.img_size, self.Num_stackFrame))
    #     # print("shape of observation stack={}".format(observation_stack.shape))
    #     for stack_frame in range(self.Num_stackFrame):
    #         observation_stack[:, :, stack_frame] = observation_set[-1 -
    #                                                                (self.Num_skipFrame * stack_frame)]
    #     observation_stack = np.uint8(observation_stack)

    #     return observation_stack, observation_set, state_stack, state_set

    # # Resize input information
    # def resize_input(self, env_info, observation_set, state_set):

    #     observation = env_info[1]
    #     observation_set.append(observation)
    #     # 使用观察组根据跳帧和堆叠帧的数量堆叠帧
    #     observation_stack = np.zeros(
    #         (self.img_size, self.img_size, self.Num_stackFrame))
    #     for stack_frame in range(self.Num_stackFrame):
    #         observation_stack[:, :, stack_frame] = observation_set[-1 -
    #                                                                (self.Num_skipFrame * stack_frame)]
    #     del observation_set[0]
    #     observation_stack = np.uint8(observation_stack)

    #     state = env_info[0]
    #     state_set.append(state)
    #     state_stack = np.zeros((self.Num_stackFrame, self.input_y))
    #     for stack_frame in range(self.Num_stackFrame):
    #         state_stack[(self.Num_stackFrame - 1) - stack_frame,
    #                     :] = state_set[-1 - (self.Num_skipFrame * stack_frame)]

    #     del self.state_set[0]

    #     return observation_stack, observation_set, state_stack, state_set

    def save_models(self):
        print('... saving models ...')
        self.actor_model.save_weights(self.actor_file)
        self.target_actor.save_weights(self.target_actor_file)
        
        self.critic_model.save_weights(self.critic_file)
        self.target_critic.save_weights(self.target_critic_file)

    def load_models(self):
        print('... loading models ...')
        self.actor_model.load_weights(self.actor_file)
        self.target_actor.load_weights(self.target_actor_file)
        self.critic_model.load_weights(self.critic_file)
        self.target_critic.load_weights(self.target_critic_file)

    def move(self, cmd=[0.0, 0.0]):
        env.step(cmd)

    def accelerate(self, accel):
        env.accel(accel)

    def update_path(self, path):
        self.path = path

        env.update_path(self.path)

    def init_sess(self):

        # Initialize variables

        os.makedirs(self.save_location)

        self.actor_file = os.path.join(self.save_location,'actor_ddpg.h5')
        self.target_actor_file = os.path.join(self.save_location,'target_actor_ddpg.h5')

        self.critic_file = os.path.join(self.save_location,'crtic_ddpg.h5')
        self.target_critic_file = os.path.join(self.save_location,'target_critic_ddpg.h5')

        # Load the file if the saved file exists
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



    def main_func(self):

        self.reward_list = []

        np.random.seed(1000)

        tf.random.set_seed(1234)

        env_info = env.get_env()

        # env.info为4维，第1维为相机消息，第2维为agent robot的self state，第3维为terminal，第4维为reward
        self.state_stack = env_info[0]

        self.step_for_newenv = 0

    def update_parameters(self, Num_start_training, Num_training, Num_test, learning_rate, Gamma, MAXEPISODES):

        self.Num_start_training = Num_start_training
        self.Num_training = Num_training
        self.Num_test = Num_test
        self.learning_rate = learning_rate
        self.Gamma = Gamma

    def get_progress(self, step):
        if self.isTraining == False:
            progress = 'Not Training'

        elif step <= self.Num_start_training:
            # Obsersvation
            progress = 'Observing'

        elif step <= self.Num_start_training + self.Num_training:
            # Training
            progress = 'Training'

        elif step < self.Num_start_training + self.Num_training + self.Num_test:
            # Testing
            progress = 'Testing'

        else:
            # Finished
            progress = 'Finished'

        self.progress = progress

        return progress

    def save_fig(self):
        plt.savefig(self.save_location + '/qcar1_reward.png')
        plt.show()


    def new_environment(self):

        plt.scatter(self.episode, self.score, c='r')

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.xlim(-1, ((self.episode/20 + 1)*20))
        plt.ylim(-12, 3)
        plt.pause(0.05)

        if self.progress == 'Training':
            self.reward_list.append(self.score)
            self.reward_array = np.array(self.reward_list)
            # ------------------------------
            np.savetxt(self.save_location + '/qcar1_reward.txt',
                       self.reward_array, delimiter=',')

        if self.progress != 'Observing':
            self.episode += 1

        self.score = 0

        env_info = env.get_env()

        self.state_stack = env_info[0]

    def update_information(self):
        # Get information for update
        env_info = env.get_env()

        self.next_state_stack = env_info[0]  # 调整输入信息
        terminal = env_info[-2]  # 获取terminal
        reward = env_info[-1]  # 获取reward

        self.memory.store_transition(
            self.state_stack, self.action, reward, self.next_state_stack, terminal)

        self.learn()

        # Update information
        self.step += 1
        self.score += reward
        self.state_stack = self.next_state_stack
        self.step_for_newenv += 1

        return terminal

    def policy(self, state,progress):


        # states = tf.convert_to_tensor(state, dtype=tf.float32)
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = self.ou_noise
        # Adding noise to action
        
        if progress == 'training':
            sampled_actions = sampled_actions.numpy() + noise 
            print(sampled_actions)
        elif progress == "Observing":
            sampled_actions = tf.random.normal(shape=[self.n_actions],
                                       mean=0.5, stddev=0.5)

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.min_action,self.max_action)

        return np.squeeze(legal_action)

    def return_action(self):
        # print(self.state_stack)
        state_stack = [float(x) for x in self.state_stack]
        state = tf.expand_dims(tf.convert_to_tensor(state_stack), 0)
        self.action = self.policy(state, self.progress)

        return self.action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)

        self.update(states, actions, rewards, states_)
        self.update_network_parameters(tau=self.tau)

    
    def update_network_parameters(self, tau=None):

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor_model.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic_model.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)




        
    def print_information(self):
        print('[Link1-'+self.progress+'] step:'+str(self.step)+'/episode:' +
              str(self.episode)+'/path:'+self.path+'/score:' + str(self.score))

    def get_actor(self,name='actor'):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(self.input_y,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="sigmoid",
                           kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs 
        model = tf.keras.Model(inputs, outputs)

        return model


    def get_critic(self,name='critic'):
    # State as input
        state_input = layers.Input(shape=(self.input_y,))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
        action_input = layers.Input(shape=(self.n_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model




# if __name__ == '__main__':
#     agent = DDPG()
#     agent.main()
