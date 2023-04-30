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
