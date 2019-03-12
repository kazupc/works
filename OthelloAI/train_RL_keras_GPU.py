# coding=utf-8
# 
# 0 : none
# 1 : black
# 2 : white

import sys
import numpy as np
import time
import random
import copy
import keras
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential, model_from_json, model_from_config
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import SGD
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from collections import deque
import tensorflow as tf

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        allow_growth=True # True->必要になったら確保, False->全部
    )
)
sess = tf.Session(config=config)


class Othello:
    Bwin = 0.0
    Wwin = 0.0
    draw = 0.0

    invalid_select = 0
    action_count = 0

    White_model_weights = np.array([None]*100)

    model = model_from_json(open('keras_reversi_model.json').read())
    model.load_weights('keras_reversi_weights.h5')

    White_model_weights[0] = model.get_weights()
        
    def __init__(self):
        self.board = [0]*64
        self.board[27] = 2
        self.board[28] = 1
        self.board[35] = 1
        self.board[36] = 2

    def get_valid_put(self,color):
        validputlist = []
        borderU = [0,1,2,3,4,5,6,7]
        borderUR = [0,1,2,3,4,5,6,7,15,23,31,39,47,55,63]
        borderR = [7,15,23,31,39,47,55,63]
        borderDR = [7,15,23,31,39,47,55,63,62,61,60,59,58,57,56]
        borderD = [56,57,58,59,60,61,62,63]
        borderDL = [0,8,16,24,32,40,48,56,57,58,59,60,61,62,63]
        borderL = [0,8,16,24,32,40,48,56]
        borderUL = [7,6,5,4,3,2,1,0,8,16,24,32,40,48,56]

        if color == 1:
            contrary_color = 2
        else:
            contrary_color = 1

        for x in range(0,64):
            Uc = URc = Rc = DRc = Dc = DLc = Lc = ULc = 0
            U = UR = R = DR = D = DL = L = UL = x
            
            if self.board[x] != 0:
                continue

            if U not in borderU:
                while True:
                    U -= 8
                    if self.board[U] != contrary_color or U in borderU:
                        break
                    Uc += 1
                if Uc > 0 and self.board[U] == color:
                    validputlist.append(x)
                    continue

            if UR not in borderUR:
                while True:
                    UR -= 7
                    if self.board[UR] != contrary_color or UR in borderUR:
                        break
                    URc += 1
                if URc > 0 and self.board[UR] == color:
                    validputlist.append(x)
                    continue

            if R not in borderR:
                while True:
                    R += 1 
                    if self.board[R] != contrary_color or R in borderR:
                        break
                    Rc += 1
                if Rc > 0 and self.board[R] == color:
                    validputlist.append(x)
                    continue

            if DR not in borderDR:
                while True:
                    DR += 9
                    if self.board[DR] != contrary_color or DR in borderDR:
                        break
                    DRc += 1
                if DRc > 0 and self.board[DR] == color:
                    validputlist.append(x)
                    continue

            if D not in borderD:
                while True:
                    D += 8
                    if self.board[D] != contrary_color or D in borderD:
                        break
                    Dc += 1
                if Dc > 0 and self.board[D] == color:
                    validputlist.append(x)
                    continue

            if DL not in borderDL:
                while True:
                    DL += 7
                    if self.board[DL] != contrary_color or DL in borderDL:
                        break
                    DLc += 1
                if DLc > 0 and self.board[DL] == color:
                    validputlist.append(x)
                    continue

            if L not in borderL:
                while True:
                    L -= 1
                    if self.board[L] != contrary_color or L in borderL:
                        break
                    Lc += 1
                if Lc > 0 and self.board[L] == color:
                    validputlist.append(x)
                    continue

            if UL not in borderUL:
                while True:
                    UL -= 9
                    if self.board[UL] != contrary_color or UL in borderUL:
                        break
                    ULc += 1
                if ULc > 0 and self.board[UL] == color:
                    validputlist.append(x)
                    continue

        if len(validputlist) == 0:
            validputlist.append(64)

        return validputlist

    def check_valid_put(self,x,color):
        U = UR = R = DR = D = DL = L = UL = x
        Uc = URc = Rc = DRc = Dc = DLc = Lc = ULc = 0
        reverselist = []
        borderU = [0,1,2,3,4,5,6,7]
        borderUR = [0,1,2,3,4,5,6,7,15,23,31,39,47,55,63]
        borderR = [7,15,23,31,39,47,55,63]
        borderDR = [7,15,23,31,39,47,55,63,62,61,60,59,58,57,56]
        borderD = [56,57,58,59,60,61,62,63]
        borderDL = [0,8,16,24,32,40,48,56,57,58,59,60,61,62,63]
        borderL = [0,8,16,24,32,40,48,56]
        borderUL = [7,6,5,4,3,2,1,0,8,16,24,32,40,48,56]

        if color == 1:
            contrary_color = 2
        else:
            contrary_color = 1
        
        if self.board[x] != 0:
            return reverselist

        if U not in borderU:
            while True:
                U -= 8
                if self.board[U] != contrary_color or U in borderU:
                    break
                Uc += 1
            if Uc > 0 and self.board[U] == color:
                reverselist.append("U")

        if UR not in borderUR:
            while True:
                UR -= 7
                if self.board[UR] != contrary_color or UR in borderUR:
                    break
                URc += 1
            if URc > 0 and self.board[UR] == color:
                reverselist.append("UR")

        if R not in borderR:
            while True:
                R += 1 
                if self.board[R] != contrary_color or R in borderR:
                    break
                Rc += 1
            if Rc > 0 and self.board[R] == color:
                reverselist.append("R")

        if DR not in borderDR:
            while True:
                DR += 9
                if self.board[DR] != contrary_color or DR in borderDR:
                    break
                DRc += 1
            if DRc > 0 and self.board[DR] == color:
                reverselist.append("DR")

        if D not in borderD:
            while True:
                D += 8
                if self.board[D] != contrary_color or D in borderD:
                    break
                Dc += 1
            if Dc > 0 and self.board[D] == color:
                reverselist.append("D")

        if DL not in borderDL:
            while True:
                DL += 7
                if self.board[DL] != contrary_color or DL in borderDL:
                    break
                DLc += 1
            if DLc > 0 and self.board[DL] == color:
                reverselist.append("DL")

        if L not in borderL:
            while True:
                L -= 1
                if self.board[L] != contrary_color or L in borderL:
                    break
                Lc += 1
            if Lc > 0 and self.board[L] == color:
                reverselist.append("L")

        if UL not in borderUL:
            while True:
                UL -= 9
                if self.board[UL] != contrary_color or UL in borderUL:
                    break
                ULc += 1
            if ULc > 0 and self.board[UL] == color:
                reverselist.append("UL")

        return reverselist

    def reverse(self,x,color,reverselist):
        U = UR = R = DR = D = DL = L = UL = x

        self.board[x] = color
        
        for P in reverselist:
            if P == "U":
                while True:
                    U -= 8
                    if self.board[U] == color:
                        break
                    self.board[U] = color

            if P == "UR":
                while True:
                    UR -= 7
                    if self.board[UR] == color:
                        break
                    self.board[UR] = color

            if P == "R":
                while True:
                    R += 1
                    if self.board[R] == color:
                        break
                    self.board[R] = color

            if P == "DR":
                while True:
                    DR += 9
                    if self.board[DR] == color:
                        break
                    self.board[DR] = color

            if P == "D":
                while True:
                    D += 8
                    if self.board[D] == color:
                        break
                    self.board[D] = color

            if P == "DL":
                while True:
                    DL += 7
                    if self.board[DL] == color:
                        break
                    self.board[DL] = color

            if P == "L":
                while True:
                    L -= 1
                    if self.board[L] == color:
                        break
                    self.board[L] = color

            if P == "UL":
                while True:
                    UL -= 9
                    if self.board[UL] == color:
                        break
                    self.board[UL] = color

    def check_pass(self,color):
        for i in range(0,64):
            if len(self.check_valid_put(i,color)) != 0:
                return False
        else:
            return True

    def dead_lock(self):
        if self.check_pass(1) and self.check_pass(2):
            return True
        else:
            return False

    def which_is_winner(self):
        score_B = 0
        score_W = 0
        for i in range(0,64):
            if self.board[i] == 1:
                score_B += 1
            elif self.board[i] == 2:
                score_W += 1

        if score_B > score_W:
            Othello.Bwin += 1
            reward = 1 
        elif score_B < score_W:
            Othello.Wwin += 1
            reward = -1
        else:
            Othello.draw += 1
            reward = 0
        
        return reward

    def check_terminal(self):
        if 0 not in self.board or self.dead_lock():
                reward = self.which_is_winner()
                terminal = True
                return reward,terminal
        else:
            reward = 0
            terminal = False
            return reward,terminal

    def print_board(self):
        for i in range(1,65):
            if i % 8 == 0:
                print(" " + str(self.board[i-1]))
            else:
                print(" " + str(self.board[i-1]),end="")

        print("")

    def makeinputdata(self,color):
        inputdata = [[[0 for i in range(8)] for j in range(8)] for k in range(3)]
        a = 0
        b = 0
        for i in self.board:
            if i == 0:
                inputdata[2][a][b] = 1
            elif i == color:
                inputdata[0][a][b] = 1
            else:
                inputdata[1][a][b] = 1

            b += 1
            if b == 8:
                a += 1
                b = 0

            if a == 8:
                a = 0

        return inputdata

    def White_turn(self):
        reward,terminal = self.check_terminal()
        if terminal:
            return reward,terminal

        X = np.array(self.makeinputdata(2)).reshape(1,8,8,3)
        y = Othello.model.predict(X,1,0)
        while True:
            put = np.argmax(y)
            if put > 63:
                if self.check_pass(2):
                    break
                else:
                    y[0][put] = -100
                    continue
                
            reverselist = self.check_valid_put(put,2)
            if len(reverselist) == 0:
                y[0][put] = -100
                continue
            else:
                self.reverse(put,2,reverselist)
                break

        reward,terminal = self.check_terminal()

        return reward,terminal

    def select_model(self,agent):
        index = random.randint(0,agent.i - 1)
        print("Agent VS Model_No."+str(index))
        self.model.set_weights(self.White_model_weights[index])


class Agent:
    def __init__(self):
        self.minibatch_size = 32
        self.replay_memory_size = 2000
        self.start_train_size = 1000
        self.target_update_frequency = 100
        self.D = deque(maxlen=self.replay_memory_size)
        self.discount_rate = 0.95
        self.epsilon = 1.0
        self.Ex_switch = False
        self.Load_Model()
        self.othello = None
        self.i = 1

    def Load_Model(self):
        self.q_model = model_from_json(open('keras_reversi_model.json').read())
        self.q_model.load_weights('keras_reversi_weights.h5')

        self.target_model = model_from_json(open('keras_reversi_model.json').read())
        self.target_model.load_weights('keras_reversi_weights.h5')

        #self.target_model = copy.copy(self.q_model)

        self.q_model.compile(loss='categorical_crossentropy',
                        optimizer=SGD(),
                        metrics=['accuracy'])

    def Save_Model(self):
        model_json_str = self.target_model.to_json()
        open('keras_reversi_model_RL_6.json', 'w').write(model_json_str)
        self.target_model.save_weights('keras_reversi_weights_RL_6.h5')
        
    def Replay_Memory(self,state,action,reward,next_state,terminal):
        self.D.append((state,action,reward,next_state,terminal))

        if self.Ex_switch:
            return

        if self.start_train_size < len(self.D):
            print("")
            print("[Start Experience Replay]")
            self.Ex_switch = True

    def Experience_Replay(self):
        xtrain = []
        ytrain = []

        sffindx = np.random.randint(0, len(self.D), self.minibatch_size)
        for j in sffindx:
            state,action,reward,next_state,terminal = self.D[j]

            t = self.Q_value(state)

            if terminal:
                t[action] = reward
            else:
                t[action] = reward + self.discount_rate * np.max(self.Q_value(next_state,target=True))

            xtrain.append(state)
            ytrain.append(t)

        hist = self.q_model.fit(np.array(xtrain).reshape(self.minibatch_size,8,8,3),
                                np.array(ytrain), 
                                batch_size=self.minibatch_size, 
                                epochs=1, 
                                shuffle=False,
                                verbose=0)
    
    def Q_value(self,inputdata,target=False):
        model = self.target_model if target else self.q_model
        X = np.array(inputdata).reshape(1,8,8,3)
        y = model.predict(X,1,0)
        return y[0]
        
    def Target_Network(self,custom_objects={}):
        """
        config = {
            'class_name': self.q_model.__class__.__name__,
            'config': self.q_model.get_config(),
        }
        self.target_model = model_from_config(config, custom_objects=custom_objects)
        """
        self.target_model.set_weights(self.q_model.get_weights())

        if self.i < 100:
            Othello.White_model_weights[self.i] = self.q_model.get_weights()
            self.i += 1
        else:
            Othello.White_model_weights[random.randint(0,99)] = self.q_model.get_weights()

    #def loss_function(self):

    def epsilon_greedy(self,othello):
        action_list = self.othello.get_valid_put(1)
        if random.random() > self.epsilon:
            #action_list = self.othello.get_valid_put(1)
            action = action_list[random.randint(0,len(action_list)-1)]
        else:
            #action = np.argmax(self.Q_value(self.othello.makeinputdata(1)))
            action = self.Q_value(self.othello.makeinputdata(1))
            action = np.argmax(action[action_list[0:]])
            action = action_list[action]
        return action
        

def main():
    train = 40000 #10000回で約10時間 25000回で約17時間 30000回で約27時間半 40000回で約75時間
    Winning_percentage = []
    invalid_select_percentage = []

    agent = Agent()

    for episode in range(1,train+1):
        print("Episode: "+str(episode)+" / "+str(train)+"  ",end="")
        othello = Othello()
        agent.othello = othello
        terminal = False
        reward = 0

        while not terminal:
            action = agent.epsilon_greedy(othello)
            Othello.action_count += 1
            state = othello.makeinputdata(1)
            if action > 63:
                if othello.check_pass(1):
                    reward,terminal = othello.White_turn()
                    next_state = othello.makeinputdata(1)
                    agent.Replay_Memory(state,action,reward,next_state,terminal)
                else:
                    print("invalid_select!!")
                    othello.print_board()
                    print("action : "+str(action))
                    sys.exit()
                    reward = -1
                    terminal = True
                    next_state = None
                    agent.Replay_Memory(state,action,reward,next_state,terminal)
                    Othello.invalid_select += 1
            else:
                reverselist = othello.check_valid_put(action,1)
                if len(reverselist) != 0:
                    othello.reverse(action,1,reverselist)
                    reward,terminal = othello.White_turn()
                    next_state = othello.makeinputdata(1)
                    agent.Replay_Memory(state,action,reward,next_state,terminal)
                else:
                    print("invalid_select!!")
                    othello.print_board()
                    print("action : "+str(action))
                    sys.exit()
                    reward = -1
                    terminal = True
                    next_state = None
                    agent.Replay_Memory(state,action,reward,next_state,terminal)
                    Othello.invalid_select += 1

            if agent.Ex_switch:
                agent.Experience_Replay()

        """
        if (episode % agent.target_update_frequency) == 0:
            agent.Target_Network()
            log1 = Othello.Bwin*100 / agent.target_update_frequency
            log2 = Othello.invalid_select*100 / Othello.action_count
            Winning_percentage.append(log1)
            invalid_select_percentage.append(log2)
            print("Winning_percentage: "+"{0:.1f}".format(log1)+"%")
            print("invalid_select_percentage: "+"{0:.1f}".format(log2)+"%")
            print("")
            Othello.Bwin = 0.0
            Othello.Wwin = 0.0
            Othello.draw = 0.0
            Othello.invalid_select = 0
            Othello.action_count = 0
            othello.select_model(agent)
        """

        if (episode % agent.target_update_frequency) == 0:
            agent.Target_Network()

        if reward == 1:
            print("WIN")
            print("")
            agent.Target_Network()
            Winning_percentage.append(reward)
            othello.select_model(agent)
        elif reward == 0:
            print("DRAW")
            Winning_percentage.append(reward)
        else:
            print("LOSE")
            Winning_percentage.append(reward)


    agent.Save_Model()

    fig,axis1 = plt.subplots()
    #axis2 = axis1.twinx()
    axis1.set_ylabel("Winning_percentage")
    #axis2.set_ylabel("invalid_select_percentage")
    plt.xlabel("epoch")
    #axis2.set_ylim(0.0,10.0)
    axis1.plot(Winning_percentage,label = "Winning_percentage")
    #axis2.plot(invalid_select_percentage, label = "invalid_select_percentage",color = "g")
    plt.grid(True)
    plt.title("result of Rainforcement Learning")
    plt.savefig("./result_RL_6.png")
                

if __name__ == "__main__":
    print("[start training]")
    start = time.time()
    main()
    print("[finish_training]")
    elapsed_time = time.time() - start
    print("elapsed time: " + str(elapsed_time) + " [sec]")