import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Model
import keras.backend as K
import keras

# Load Movielens 100k dataset
data = pd.read_csv('dataset/u.data', sep='\t', names=["UserID","MovieID","Rating","Timestamp"])

# Split dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

num_user = np.max(data["UserID"])
num_movie = np.max(data["MovieID"])

train_user = train_data["UserID"].values
train_movie = train_data["MovieID"].values
train_x = [train_user,train_movie]
train_y = train_data["Rating"].values

valid_user = test_data["UserID"].values
valid_movie = test_data["MovieID"].values
valid_x = [valid_user,valid_movie]
valid_y = test_data["Rating"].values

K.clear_session()

from keras.layers import Embedding, Reshape, Input, Dot, Flatten
from keras.layers import Input, Embedding, Concatenate, Dense
from keras.metrics import RootMeanSquaredError,MeanAbsoluteError

def neural_cf_model(num_users, num_movies, embedding_size, hidden_size):
    input_tensor = Input(shape=[None,],dtype="int32")
    input_uer = input_tensor[:,0]
    print(input_uer)
    model_uer = Embedding(num_users + 1, embedding_size, input_length=1)(input_uer)
    model_uer = Reshape((embedding_size,))(model_uer)
    user_embed = Flatten()(model_uer)

    input_movie = input_tensor[:,1]
    model_movie = Embedding(num_movies+ 1, embedding_size, input_length=1)(input_movie)
    model_movie = Reshape((embedding_size,))(model_movie)
    movie_embed = Flatten()(model_movie)

    concat = Concatenate()([user_embed, movie_embed])

    for i in range(3):
        concat = Dense(hidden_size, activation='relu')(concat)

    output = Dense(1, activation='sigmoid')(concat)
    model = Model(inputs=[input_tensor], outputs=output)
    model.compile(loss='mse', optimizer='Adam',metrics=[RootMeanSquaredError(),MeanAbsoluteError()])
    model.summary()
    return model

model = neural_cf_model(num_user, num_movie, 100,10)
history=model.fit(train_x,train_y,batch_size = 100,epochs =20,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)],validation_data=(valid_x,valid_y))
print(history.history['loss'][-1])

# 没活了，手动调吧

# result = {}
# # model = Recmand_model(num_user,num_movie,100)
# for m in [100,200,500]:
#     for k in [10,50,100]:
#         for i in [50,500,1000]:
#             for j in [10,20,50]:
#                 model = neural_cf_model(num_user, num_movie, m,k)
#                 history=model.fit(train_x,train_y,batch_size = i,epochs =j,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)],validation_data=(valid_x,valid_y))
#                 result[history.history['loss'][-1]] = [k,i,j]
#
# keys = list(result.keys())
# keys.sort()
# min_key = keys[-1]
# print('best para:{}'.format(result[min_key]))

