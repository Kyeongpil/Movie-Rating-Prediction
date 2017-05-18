# coding: utf-8
from konlpy.tag import Twitter
from torch.autograd import Variable
from torch import nn
from model import CNNReg
from model import makeBatch
import pickle
import torch


twitter = Twitter()

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
batch_size = 32
num_iter = int(trainNum/batch_size)

reg = CNNReg(vocaNum, embedding_dim)
reg.load_state_dict(torch.load("cnn_regression.pkl"))
reg.eval()

# test with testset
criterion = nn.L1Loss()
num_iter = int(testNum/batch_size)
average_loss = 0
for i in range(num_iter):
    batch_X = test_X[i*batch_size:(i+1)*batch_size]
    batch_y = test_y[i*batch_size:(i+1)*batch_size]
    batch_X = makeBatch(batch_X)
    batch_y = torch.FloatTensor(batch_y)
    batch_X = Variable(batch_X)
    batch_y = Variable(batch_y)

    predict = reg(batch_X)
    loss = criterion(predict, batch_y)
    average_loss += loss.data.mean()

average_loss /= num_iter
print("test error(MAE): {}".format(average_loss))

test_text = u"이 영화 겁나 재미있다 ㅋㅋㅋㅋ 겁나 웃겨"
test = twitter.morphs(test_text)
test = [[voca2index[word] if word in voca2index else voca2index['<UNK>'] for word in test]]
test = makeBatch(test)
test = Variable(test)
predict = reg(test).data.tolist()[0][0]
print("{} - predicted score: {}".format(test_text, predict))
