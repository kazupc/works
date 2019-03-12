import sys, random, copy, csv, time
import numpy as np
import gym
import scipy.misc as spm
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

from chainer import cuda, Function, Variable, optimizers, serializers, initializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L

ENV_NAME = 'Pong-v0'
EPISODES = 100000 #12000
FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
STATE_LENGTH = 4
INITIAL_EPSILON = 1.0 #1.0
FINAL_EPSILON = 0.1
EXPLORATION_STEPS = 1000000 #1000000
INITIAL_REPLAY_SIZE = 10000 #20000
MAX_REPLAY_SIZE = 100000 #400000
TARGET_UPDATE_INTERVAL = 10000 #10000
TRAIN_INTERVAL = 1 #4
ACTION_INTERVAL = 1 #4
SAVE_INTERVAL = 30000 #300000
EVALUATE_INTERVAL = 5
MINI_BATCH_SIZE = 32
DISCOUNT_RATE = 0.99
MODEL_SAVE_PATH = "/home/k_murakami/deeplearning/DQN_org/models_master/"
RESULT_SAVE_PATH = "/home/k_murakami/deeplearning/DQN_org/results_master/"
CSV_SAVE_PATH = "/home/k_murakami/deeplearning/DQN_org/csv_master/

start = 0.0

cuda.get_device(1).use()

class DQN(Chain):
    def __init__(self, num_actions):
        initializer = initializers.HeNormal()
        super(DQN, self).__init__()
        with self.init_scope():
            self.l1=L.Convolution2D(STATE_LENGTH, 32, ksize=8, stride=4, nobias=False, initialW=initializer)
            self.l2=L.Convolution2D(32, 64, ksize=4, stride=2, nobias=False, initialW=initializer)
            self.l3=L.Convolution2D(64, 64, ksize=3, stride=1, nobias=False, initialW=initializer)
            self.l4=L.Linear(3136, 512, initialW=initializer)
            self.q_value=L.Linear(512, num_actions,
                             initialW=np.zeros((num_actions, 512),
                             dtype=np.float32))


    def q_function(self, state):
        h1 = F.relu(self.l1(state/255.))
        h2 = F.relu(self.l2(h1))
        h3 = F.relu(self.l3(h2))
        h4 = F.relu(self.l4(h3))
        return self.q_value(h4)


class Agent:
    def __init__(self,num_actions):
        self.num_actions = num_actions
        self.q_model = DQN(num_actions).to_gpu()
        self.target_model = copy.deepcopy(self.q_model)
        self.optimizer = optimizers.RMSpropGraves(lr=0.00025, alpha=0.95, momentum=0.95, eps=0.01)
        self.optimizer.setup(self.q_model)
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.frames = 0
        self.num_experiences = 0
        self.repeated_action = 0
        self.eval_repeated_action = 0
        self.q_max = 0.0
        self.total_qmax = 0.0
        self.switch = False

        self.replay_buffer = [np.zeros((MAX_REPLAY_SIZE, STATE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8),
                  np.zeros(MAX_REPLAY_SIZE, dtype=np.uint8),
                  np.zeros(MAX_REPLAY_SIZE, dtype=np.float32),
                  np.zeros((MAX_REPLAY_SIZE, STATE_LENGTH, FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8),
                  np.zeros(MAX_REPLAY_SIZE, dtype=np.bool)]

    def load_model(self):
        self.frames = 18300000
        serializers.load_npz(MODEL_SAVE_PATH+"model_"+str(self.frames), self.q_model)
        serializers.load_npz(MODEL_SAVE_PATH+"target_model_"+str(self.frames),self.target_model)

    def loss_function(self,state, action, reward, next_state, terminal):
        s = Variable(cuda.to_gpu(state))
        s_dash = Variable(cuda.to_gpu(next_state))

        q = self.q_model.q_function(s)  # Get Q-value

        # Generate Target Signals
        tmp = self.target_model.q_function(s_dash)  # Q(s',*)
        tmp = list(map(np.max, tmp.data))  # max_a Q(s',a)
        max_q_prime = np.asanyarray(tmp, dtype=np.float32)
        target = np.asanyarray(copy.deepcopy(q.data.get()), dtype=np.float32)

        for i in range(MINI_BATCH_SIZE):
            if terminal[i]:
                tmp_ = reward[i]
            else:
                #  The sign of reward is used as the reward of DQN!
                tmp_ = reward[i] + DISCOUNT_RATE * max_q_prime[i]

            target[i, action[i]] = tmp_

        # TD-error clipping
        td = Variable(cuda.to_gpu(target)) - q  # TD error
        td_tmp = td.data + 1000.0 * (abs(td.data) <= 1)  # Avoid zero division
        td_clip = td * (abs(td.data) <= 1) + td/abs(td_tmp) * (abs(td.data) > 1)

        zero_val = Variable(cuda.to_gpu(np.zeros((MINI_BATCH_SIZE, self.num_actions), dtype=np.float32)))
        loss = F.mean_squared_error(td_clip, zero_val)
        return loss

    def preprocess(self,observation, last_observation, initial=False):
        img1 = np.dot(observation[...,:3], [0.299, 0.587, 0.114]) # Convert RGB to Grayscale
        obs_array1 = (spm.imresize(img1, (110, 84)))[110-84-8:110-8, :]  # Scaling

        if initial:
            return obs_array1

        img2 = np.dot(last_observation[...,:3], [0.299, 0.587, 0.114]) # Convert RGB to Grayscale
        obs_array2 = (spm.imresize(img2, (110, 84)))[110-84-8:110-8, :]  # Scaling

        obs_processed = np.maximum(obs_array1, obs_array2)  # Take maximum from two frames
        return obs_processed

    def get_action(self, state, episode):
        action = self.repeated_action

        if self.frames % ACTION_INTERVAL == 0:
            x = np.asanyarray(state.reshape(1, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=np.float32)
            s = Variable(cuda.to_gpu(x))
            action = self.q_model.q_function(s)
            q_max = np.max(action.data.get()[0])

            if self.epsilon >= random.random() or self.frames < INITIAL_REPLAY_SIZE:
                action = random.randrange(self.num_actions)
                print("dqn_org.py | EPISODE  %d  /  RANDOM SELECT  %d  /  EPSILON  %.6f  /  Q_max  %3f" % (episode, action, self.epsilon, q_max))
            else:
                action = np.argmax(action.data.get()[0])
                print("dqn_org.py | EPISODE  %d  /  AGENT  SELECT  %d  /  EPSILON  %.6f  /  Q_max  %3f" % (episode, action, self.epsilon, q_max))
            self.repeated_action = action

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.frames >= INITIAL_REPLAY_SIZE:
            if (self.epsilon - self.epsilon_step) < FINAL_EPSILON:
                self.epsilon = FINAL_EPSILON
            else:
                self.epsilon -= self.epsilon_step

        return action

    def eval_get_action(self, state, one_episode_reward, count):
        action = self.eval_repeated_action

        if count % ACTION_INTERVAL == 0:
            x = np.asanyarray(state.reshape(1, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=np.float32)
            s = Variable(cuda.to_gpu(x))
            action = self.q_model.q_function(s)
            q_max = np.max(action.data.get()[0])

            action = np.argmax(action.data.get()[0])
            print("dqn_org.py | EVALUATING  |  AGENT  SELECT  %d  /  TOTAL REWARD  %d  /  Q_max  %3f" % (action, one_episode_reward, q_max))
            self.total_qmax += q_max
            self.eval_repeated_action = action
        return action

    def replay_memory(self,state,action,reward,next_state,terminal):
        data_index = self.num_experiences % MAX_REPLAY_SIZE

        self.replay_buffer[0][data_index] = state
        self.replay_buffer[1][data_index] = action
        self.replay_buffer[2][data_index] = reward
        self.replay_buffer[3][data_index] = next_state
        self.replay_buffer[4][data_index] = terminal

        self.num_experiences += 1

        if INITIAL_REPLAY_SIZE < data_index and self.switch == False:
            print("")
            print("[Start Experience Replay]")
            self.switch = True

    def experience_replay(self):
        state_mb = np.ndarray(shape=(MINI_BATCH_SIZE, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=np.float32)
        action_mb = np.ndarray(shape=(MINI_BATCH_SIZE, 1), dtype=np.int8)
        reward_mb = np.ndarray(shape=(MINI_BATCH_SIZE, 1), dtype=np.float32)
        next_state_mb = np.ndarray(shape=(MINI_BATCH_SIZE, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=np.float32)
        terminal_mb = np.ndarray(shape=(MINI_BATCH_SIZE, 1), dtype=np.bool)

        if self.num_experiences < MAX_REPLAY_SIZE:
            sffindx = np.random.randint(0, self.num_experiences, MINI_BATCH_SIZE)
        else:
            sffindx = np.random.randint(0, MAX_REPLAY_SIZE, MINI_BATCH_SIZE)

        i = 0
        for j in sffindx:
            state_mb[i] = self.replay_buffer[0][j]
            action_mb[i] = self.replay_buffer[1][j]
            reward_mb[i] = self.replay_buffer[2][j]
            next_state_mb[i] = self.replay_buffer[3][j]
            terminal_mb[i] = self.replay_buffer[4][j]
            i += 1

        self.q_model.cleargrads()
        loss = self.loss_function(state_mb,action_mb,reward_mb,next_state_mb,terminal_mb)
        loss.backward()
        self.optimizer.update()

    def target_network(self):
        self.target_model = copy.deepcopy(self.q_model)

    def save_model(self):
        serializers.save_npz(MODEL_SAVE_PATH+"model_"+str(self.frames), self.q_model)
        serializers.save_npz(MODEL_SAVE_PATH+"target_model_"+str(self.frames),self.target_model)

class Evaluate:
    def __init__(self, env):
        self.f1 = open(CSV_SAVE_PATH+"score.csv","a")
        self.csvWriter1 = csv.writer(self.f1)
        self.f2 = open(CSV_SAVE_PATH+"average_q_max.csv","a")
        self.csvWriter2 = csv.writer(self.f2)
        self.f3 = open(CSV_SAVE_PATH+"time.csv","a")
        self.csvWriter3 = csv.writer(self.f3)

        self.env = env
        self.Score =[]
        self.Average_q_max =[]
        self.Frames = []

    def play(self, agent):
        global start
        agent.total_qmax = 0.0
        one_episode_reward = 0
        count = 0
        terminal = False
        state = np.zeros((STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT), dtype=np.uint8)
        observation = self.env.reset()
        observation, reward, terminal, _ = self.env.step(0)
        state[3] = agent.preprocess(observation, None, initial=True)
        
        while not terminal:
            last_observation = observation
            action = agent.eval_get_action(state, one_episode_reward, count)
            observation, reward, terminal, _ = self.env.step(action)
            #env.render()
            obs_processed = agent.preprocess(observation, last_observation)
            for i in range(STATE_LENGTH - 1):
                state[i] = state[i + 1].astype(np.uint8)
            state[STATE_LENGTH - 1] = obs_processed.astype(np.uint8)
            reward = np.sign(reward)
            one_episode_reward += reward
            count += 1

        average_q_max = agent.total_qmax / count

        write_csv1 = [agent.frames, one_episode_reward]
        self.csvWriter1.writerow(write_csv1)
        self.f1.flush()

        write_csv2 = [agent.frames, average_q_max]
        self.csvWriter2.writerow(write_csv2)
        self.f2.flush()

        elapsed_time = time.time() - start
        write_csv3 = [agent.frames, elapsed_time]
        self.csvWriter3.writerow(write_csv3)
        self.f3.flush()

        self.Score.append(one_episode_reward)
        self.Average_q_max.append(average_q_max)
        self.Frames.append(agent.frames)

        self.save_result()

    def save_result(self):
        plt.figure(figsize=(10,8))
        plt.ylabel("Score")
        plt.xlabel("Frames")
        plt.plot(self.Frames,self.Score, color="dodgerblue")
        #plt.grid(True)
        plt.title("Result of Deep Rainforcement Learning")
        plt.savefig(RESULT_SAVE_PATH+"score.png")

        plt.figure(figsize=(10,8))
        plt.ylabel("Average of Q_max per episode")
        plt.xlabel("Frames")
        plt.plot(self.Frames,self.Average_q_max, color="limegreen")
        #plt.grid(True)
        plt.title("Result of Deep Rainforcement Learning")
        plt.savefig(RESULT_SAVE_PATH+"average_q_max.png")


def main():
    env = gym.make(ENV_NAME)
    agent = Agent(num_actions=env.action_space.n)
    eva = Evaluate(env)
    #agent.load_model()

    for episode in range(1,EPISODES+1):
        terminal = False
        state = np.zeros((STATE_LENGTH, 84, 84), dtype=np.uint8)
        observation = env.reset()
        observation, reward, terminal, _ = env.step(0)
        state[3] = agent.preprocess(observation, None, initial=True)
        
        while not terminal:
            last_observation = observation
            action = agent.get_action(state, episode)
            observation, reward, terminal, _ = env.step(action)
            #env.render()
            agent.frames += 1
            obs_processed = agent.preprocess(observation, last_observation)
            next_state = copy.deepcopy(state)
            for i in range(STATE_LENGTH - 1):
                next_state[i] = next_state[i + 1].astype(np.uint8)
            next_state[STATE_LENGTH - 1] = obs_processed.astype(np.uint8)

            reward = np.sign(reward)
            agent.replay_memory(state,action,reward,next_state,terminal)

            if agent.frames % TRAIN_INTERVAL == 0 and agent.switch == True:
                agent.experience_replay()

            if agent.frames % TARGET_UPDATE_INTERVAL == 0 and agent.switch == True:
                agent.target_network()
            
            if agent.frames % SAVE_INTERVAL == 0:
                agent.save_model()
                print("[model is saved in "+str(agent.frames)+" frame]")
            

            state = copy.deepcopy(next_state)
        
        if episode % EVALUATE_INTERVAL == 0:
            eva.play(agent)
        

if __name__ == "__main__":
    start = time.time()
    main()
