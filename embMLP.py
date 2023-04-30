import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model
import keras
from keras.metrics import RootMeanSquaredError,MeanAbsoluteError

# this part is for GridCV to find a suitabel parameters
#
ratings = pd.read_csv('./dataset/u.data', sep='\t', header=None,
                         names=['user_id', 'movie_id', 'rating', 'timestamp'])

# encoder
movie_encoder = LabelEncoder()
user_encoder = LabelEncoder()
ratings['movie_id'] = movie_encoder.fit_transform(ratings['movie_id'])
ratings['user_id'] = user_encoder.fit_transform(ratings['user_id'])


# embedding layer
num_users = ratings['user_id'].nunique()
num_movies = ratings['movie_id'].nunique()
embedding_size = 32

user_input = Input(shape=(1,))
user_embedding = Embedding(num_users, embedding_size)(user_input)
user_embedding = Flatten()(user_embedding)

movie_input = Input(shape=(1,))
movie_embedding = Embedding(num_movies, embedding_size)(movie_input)
movie_embedding = Flatten()(movie_embedding)

# concate
merged = keras.layers.Concatenate()([user_embedding, movie_embedding])

# MLP layer
dense1 = Dense(64, activation='relu')(merged)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(1)(dense2)

# build model
model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(loss='mse', optimizer='adam',metrics=[RootMeanSquaredError(),MeanAbsoluteError()])

# data spliting
train_size = int(len(ratings) * 0.8)
train_ratings = ratings[:train_size]
test_ratings = ratings[train_size:]

result = {}
# find parameters
for i in [20,40,60]:
    for j in [64,128,256]:
        # train
        history = model.fit([train_ratings['user_id'], train_ratings['movie_id']], train_ratings['rating'],
                            batch_size=j, epochs=i, verbose=1, validation_split=0.1,
                            callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)])

        # # evaluate
        # mse = model.evaluate([test_ratings['user_id'], test_ratings['movie_id']], test_ratings['rating'])
        result[history.history['loss'][-1]] = [i,j]

keys = list(result.keys())
keys.sort()
min_key = keys[-1]
print('best para:{}'.format(result[min_key]))



# The following part is for 5-fold cross validation
#
import os

# path to dataset folder
files_dir = os.path.expanduser("./dataset/fold_data/")

# folds_files is a list of tuples containing file paths:
# [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
train_file = files_dir + "u%d.base"
test_file = files_dir + "u%d.test"
folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

rmse_list = []
mae_list = []

import time
startT = time.time()

from keras.metrics import RootMeanSquaredError, MeanAbsoluteError

predictlist = []

for (trainset_dir, testset_dir) in folds_files:
    trainset = pd.read_csv(trainset_dir, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])
    testset = pd.read_csv(testset_dir, sep='\t', names=['user_id', 'movie_id', 'rating', 'timestamp'])

    movie_encoder = LabelEncoder()
    user_encoder = LabelEncoder()
    trainset['movie_id'] = movie_encoder.fit_transform(trainset['movie_id'])
    trainset['user_id'] = user_encoder.fit_transform(trainset['user_id'])

    num_users = trainset['user_id'].nunique() + testset['user_id'].nunique()
    num_movies = trainset['movie_id'].nunique() + testset['movie_id'].nunique()
    embedding_size = 32

    user_input = Input(shape=(1,))
    user_embedding = Embedding(num_users, embedding_size)(user_input)
    user_embedding = Flatten()(user_embedding)

    movie_input = Input(shape=(1,))
    movie_embedding = Embedding(num_movies, embedding_size)(movie_input)
    movie_embedding = Flatten()(movie_embedding)

    merged = keras.layers.Concatenate()([user_embedding, movie_embedding])

    dense1 = Dense(64, activation='relu')(merged)
    dense2 = Dense(32, activation='relu')(dense1)
    output = Dense(1)(dense2)

    model = Model(inputs=[user_input, movie_input], outputs=output)
    model.compile(loss='mse', optimizer='adam', metrics=[RootMeanSquaredError(), MeanAbsoluteError()])

    history = model.fit([trainset['user_id'], trainset['movie_id']], trainset['rating'],
                        batch_size=64, epochs=20, verbose=1, validation_split=0.1,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)])

    mse = model.evaluate([testset['user_id'], testset['movie_id']], testset['rating'])

    rmse_list.append(mse[1])
    mae_list.append(mse[2])

    predicts = model.predict([testset['user_id'], testset['movie_id']])

    predictlist.append(predicts)

endT = time.time()
import numpy as np
print(rmse_list)
print(mae_list)
print('Mean rmse:{}'.format(np.mean(rmse_list)))
print('Mean mae:{}'.format(np.mean(mae_list)))
print('Running time:{} s'.format(endT-startT))

# out put the last fold which is u5.test 's prediction rating out to a csv file for last compare
u5_predict = pd.DataFrame(predictlist[-1])
u5_predict.to_csv('emb_predictions.csv', index=False)