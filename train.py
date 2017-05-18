# coding: utf-8
from torch.autograd import Variable
from torch import nn
from model import CNNReg
from model import makeBatch
import pickle
import torch


with open("data.pkl", "rb") as f:
    index2voca = pickle.load(f)
    voca2index = pickle.load(f)
    train_X = pickle.load(f)
    train_y = pickle.load(f)
    test_X = pickle.load(f)
    test_y = pickle.load(f)

vocaNum = len(index2voca)
trainNum = len(train_y)
testNum = len(test_y)


embedding_dim = 200
epoch = 10
batch_size = 32
num_iter = int(trainNum/batch_size)
print_iter = 2000

reg = CNNReg(vocaNum, embedding_dim)
reg.cuda()

criterion = nn.MSELoss()
opt = torch.optim.Adam(reg.parameters())

for e in range(epoch):
    for i in range(num_iter):
        opt.zero_grad()
        batch_X = train_X[i*batch_size:(i+1)*batch_size]
        batch_y = train_y[i*batch_size:(i+1)*batch_size]
        batch_X = makeBatch(batch_X)
        batch_y = torch.FloatTensor(batch_y)

        batch_X = Variable(batch_X).cuda()
        batch_y = Variable(batch_y).cuda()

        predict = reg(batch_X)
        loss = criterion(predict, batch_y)
        loss.backward()
        opt.step()

        if i % print_iter == 0 or i == num_iter-1:
            print("batch: {}, iteration: {}, loss: {}\n".format(e, i, loss.data.mean()))

torch.save(reg.state_dict(), 'cnn_regression.pkl')
