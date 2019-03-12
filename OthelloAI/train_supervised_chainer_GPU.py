# coding=utf-8
# 
# 0 : none
# 1 : black
# 2 : white

import sys
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from matplotlib import pyplot as plt
import random
import time

#definition model
class MyChain(Chain):
    def __init__(self):
        super(MyChain, self).__init__(
            conv1 = L.Convolution2D(4,192,5,pad=2),
            conv2 = L.Convolution2D(192,192,3,pad=1),
            conv3 = L.Convolution2D(192,192,3,pad=1),
            conv4 = L.Convolution2D(192,192,3,pad=1),
            conv5 = L.Convolution2D(192,192,3,pad=1),
            conv6 = L.Convolution2D(192,192,3,pad=1),
            conv7 = L.Convolution2D(192,192,3,pad=1),
            conv8 = L.Convolution2D(192,192,3,pad=1),
            conv9 = L.Convolution2D(192,192,3,pad=1),
            conv10 = L.Convolution2D(192,192,3,pad=1),
            conv11 = L.Convolution2D(192,192,3,pad=1),
            conv12 = L.Convolution2D(192,192,3,pad=1),
            conv13 = L.Convolution2D(192,1,1),
            l1 = L.Linear(64,64)
        )

    def __call__(self,x,y):
        yhat = self.forward(x)
        loss = F.softmax_cross_entropy(yhat,y)
        accuracy = F.accuracy(yhat,y)
        return loss,accuracy

    def forward(self,x):
        h1 = F.relu(self.conv1(x))
        h2 = F.relu(self.conv2(h1))
        h3 = F.relu(self.conv3(h2))
        h4 = F.relu(self.conv4(h3))
        h5 = F.relu(self.conv5(h4))
        h6 = F.relu(self.conv6(h5))
        h7 = F.relu(self.conv7(h6))
        h8 = F.relu(self.conv8(h7))
        h9 = F.relu(self.conv9(h8))
        h10 = F.relu(self.conv10(h9))
        h11 = F.relu(self.conv11(h10))
        h12 = F.relu(self.conv12(h11))
        h13 = F.relu(self.conv13(h12))
        yhat = self.l1(h13)
        return yhat


class MakeRecordList():
    def __init__(self):
        self.f1 = open("inputdata_Bfull3.txt","r")
        self.f2 = open("correctanswer_Bfull3.txt","r")
        self.plane4 = [[1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1],
                       [1,1,1,1,1,1,1,1]]#plane4: filled with 1

    def makelist(self,numdata):
        print("[Loading training data set ...]")
        #record_x = []
        #record_y = []
        global xtrain,ytrain

        for line in self.f1:
            plane1 = []
            plane2 = []
            plane3 = []
            tmp1 = []
            tmp2 = []
            tmp3 = []
            oneset = []

            line = list(line)
            line.pop()
            line = list(map(int,line))
            for i in line:
                if i == 0:
                    tmp1.append(0)#plane1: black stone is 0 or 1
                    tmp2.append(0)#plane2: white stone is 0 or 1
                    tmp3.append(1)#plane3: none is 0 or 1
                elif i == 1:
                    tmp1.append(1)
                    tmp2.append(0)
                    tmp3.append(0)
                else:
                    tmp1.append(0)
                    tmp2.append(1)
                    tmp3.append(0)

                if len(tmp1) == 8:
                    plane1.append(tmp1)
                    plane2.append(tmp2)
                    plane3.append(tmp3)
                    tmp1 = []
                    tmp2 = []
                    tmp3 = []

            oneset.append(plane1)
            oneset.append(plane2)
            oneset.append(plane3)
            oneset.append(self.plane4)

            xtrain.append(oneset)

            ytrain.append(int(self.f2.readline().rstrip("\n")))

            if len(ytrain) == numdata:
                break

        print("[finished loading data set]")

        xtrain = np.array(xtrain).astype(np.float32)
        ytrain = np.array(ytrain).astype(np.int32)

    
start = time.time()

#training
model = MyChain()
chainer.cuda.get_device(0).use()  # Make a specified GPU current
model.to_gpu()  # Copy the model to the GPU
optimizer = optimizers.SGD()
optimizer.setup(model)

mrl = MakeRecordList()
bs = 16 # VRAM usage is MAX 1500
train = 10
n = 10 # 3634474
train_losses = []
train_accuracy = []
xtrain = []
ytrain = []
mrl.makelist(n)

print("[start training]")
for T in range(train):
    sffindx = np.random.permutation(n)
    for i in range(0,n,bs):
        #reshape x,y and translate gpu_array and copy the array to gpu
        #y,int32, this program is classifier problem
        x = cuda.to_gpu(xtrain[sffindx[i:(i+bs) if (i+bs) < n else n]])
        y = cuda.to_gpu(ytrain[sffindx[i:(i+bs) if (i+bs) < n else n]])

        model.zerograds()
        loss,accuracy = model(x,y)
        
        loss.backward()
        optimizer.update()

    train_losses.append(loss.data)
    train_accuracy.append(accuracy.data)

    print("loss: "+str(loss.data)+"  accuracy: "+str(accuracy.data)+"  progress: "+str(T+1)+"/"+str(train))
    print("")



#save model
serializers.save_npz('chainer_reversi_model_train.npz', model)

print("[finish_training]")
elapsed_time = time.time() - start
print("elapsed time: " + str(elapsed_time) + " [sec]")

fig,axis1 = plt.subplots()
axis2 = axis1.twinx()
axis1.set_ylabel("loss")
axis2.set_ylabel("accuracy")
plt.xlabel("epoch")
axis2.set_ylim(0.0,1.0)
axis1.plot(train_losses,label = "train_loss")
axis2.plot(train_accuracy, label = "train_accuracy",color = "g")
plt.grid(True)
plt.title("result of training")
plt.savefig("./result.png")
