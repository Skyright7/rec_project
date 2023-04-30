import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 读取评分数据
ratings = pd.read_csv('./dataset/u.data', sep='\t', header=None,
                         names=['user_id', 'movie_id', 'rating', 'timestamp'])

# 编码电影和用户ID
movie_encoder = LabelEncoder()
user_encoder = LabelEncoder()
ratings['movie_id'] = movie_encoder.fit_transform(ratings['movie_id'])
ratings['user_id'] = user_encoder.fit_transform(ratings['user_id'])

from keras.layers import Input, Embedding, Flatten, Dense
from keras.models import Model
import keras

# 创建嵌入层
num_users = ratings['user_id'].nunique()
num_movies = ratings['movie_id'].nunique()
embedding_size = 32

user_input = Input(shape=(1,))
user_embedding = Embedding(num_users, embedding_size)(user_input)
user_embedding = Flatten()(user_embedding)

movie_input = Input(shape=(1,))
movie_embedding = Embedding(num_movies, embedding_size)(movie_input)
movie_embedding = Flatten()(movie_embedding)

# 将用户和电影嵌入向量进行拼接
merged = keras.layers.Concatenate()([user_embedding, movie_embedding])

# 定义MLP
dense1 = Dense(64, activation='relu')(merged)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(1)(dense2)

from keras.metrics import RootMeanSquaredError,MeanAbsoluteError
# 构建模型
model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(loss='mse', optimizer='adam',metrics=[RootMeanSquaredError(),MeanAbsoluteError()])

# 分割训练和测试集
train_size = int(len(ratings) * 0.8)
train_ratings = ratings[:train_size]
test_ratings = ratings[train_size:]

# 训练模型
history = model.fit([train_ratings['user_id'], train_ratings['movie_id']], train_ratings['rating'],
                    batch_size=64, epochs=10, verbose=1, validation_split=0.1)

# 评估模型
mse = model.evaluate([test_ratings['user_id'], test_ratings['movie_id']], test_ratings['rating'])

print('Mean Squared Error: ', mse)


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

for (trainset_dir, testset_dir) in folds_files:
    trainset = pd.read_csv(trainset_dir, sep='\t', names=["UserID","MovieID","Rating","Timestamp"])
    testset = pd.read_csv(testset_dir, sep='\t', names=["UserID","MovieID","Rating","Timestamp"])
    train_user = trainset["UserID"].values
    train_movie = trainset["MovieID"].values
    cross_x_train = np.dstack((train_user, train_movie))[0]
    cross_y_train = trainset["Rating"].values
    x_train_input = np.transpose(cross_x_train)

    test_user = testset["UserID"].values
    test_movie = testset["MovieID"].values
    cross_x_test = np.dstack((test_user, test_movie))[0]
    cross_y_test = testset["Rating"].values

    x_valid_input = np.transpose(cross_x_test)

    num_user_train = np.max(trainset["UserID"])
    num_movie_train = np.max(trainset["MovieID"])

    num_user_test = np.max(testset["UserID"])
    num_movie_test = np.max(testset["MovieID"])

    num_user = max(num_user_train,num_user_test)
    num_movie = max(num_movie_train,num_movie_test)

    # train and test algorithm.
    # 编译模型
    best_model = Recmand_model(num_user, num_movie, 50)

    # 训练模型
    best_model.fit(cross_x_train, cross_y_train, batch_size=1000, epochs=10,
                   callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)],
                   validation_data=(cross_x_test, cross_y_test))

    # 评估模型
    scores = best_model.evaluate(cross_x_test, cross_y_test, verbose=0)

    rmse_list.append(scores[1])
    mae_list.append(scores[2])

endT = time.time()
import numpy as np
print(rmse_list)
print(mae_list)
print('Mean rmse:{}'.format(np.mean(rmse_list)))
print('Mean mae:{}'.format(np.mean(mae_list)))
print('Running time:{} s'.format(endT-startT))
