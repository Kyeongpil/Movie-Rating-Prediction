# coding: utf-8

from konlpy.tag import Twitter
from glob import glob
from collections import Counter
from collections import defaultdict
import ujson as json
import pickle
from random import shuffle

files = glob("data/crawledResult/*")
twitter = Twitter()

with open(files[0]) as f:
    movies = json.load(f)

max_length = 100
ratingList = defaultdict(list)
i = 0
for file in files:
    with open(file) as f:
        movies = json.load(f)
    for movie in movies:
        ratings = movie['ratings']
        for rating in ratings:
            i += 1
            if i % 10000 == 0:
                print(i)

            reple = rating['reple']
            score = rating['score']/10
            words = twitter.morphs(reple)
            if len(words) <= max_length and len(words) > 1:
                ratingList[score].append(words)

minNum = min([len(l) for l in ratingList.values()])

voca = Counter()
dataset = []
for score, reples in ratingList.items():
    for reple in reples[:minNum]:
        voca.update(reple)
        dataset.append((score, reple))

voca_num = 20000
index2voca = [v[0] for v in voca.most_common(voca_num-2)]
index2voca.append('<UNK>')
index2voca.insert(0, '<PAD>')
voca2index = {v: i for i, v in enumerate(index2voca)}
del ratingList

shuffle(dataset)

dataset_X = []
dataset_y = []
for data in dataset:
    dataset_y.append(data[0])
    indices = []
    for word in data[1]:
        try:
            indices.append(voca2index[word])
        except:
            indices.append(voca2index['<UNK>'])
    dataset_X.append(indices)

num_data = len(dataset)
num_train = int(num_data*0.9)
num_test = num_data-num_train
print("num_train: {}, num_test: {}".format(num_train, num_test))
del dataset

train_X = dataset_X[:num_train]
train_y = dataset_y[:num_train]
test_X = dataset_X[num_train:]
test_y = dataset_y[num_train:]

with open("data.pkl", "wb") as f:
    pickle.dump(index2voca, f)
    pickle.dump(voca2index, f)
    pickle.dump(train_X, f)
    pickle.dump(train_y, f)
    pickle.dump(test_X, f)
    pickle.dump(test_y, f)
