import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras import Model
import keras.backend as K
from keras.layers import Embedding,Reshape,Input,Dot
import keras

# # Load Movielens 100k dataset
# data = pd.read_csv('dataset/u.data', sep='\t', names=["UserID","MovieID","Rating","Timestamp"])
#
# # Split dataset into training and testing sets
# train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
#
# num_user = np.max(data["UserID"])
# num_movie = np.max(data["MovieID"])
#
# train_user = train_data["UserID"].values
# train_movie = train_data["MovieID"].values
# train_x = [train_user,train_movie]
# train_y = train_data["Rating"].values
#
# valid_user = test_data["UserID"].values
# valid_movie = test_data["MovieID"].values
# valid_x = [valid_user,valid_movie]
# valid_y = test_data["Rating"].values
#
K.clear_session()

from keras.metrics import RootMeanSquaredError,MeanAbsoluteError
def Recmand_model(num_user, num_movie, k):
    input_tensor = Input(shape=[None,],dtype="int32")
    input_uer = input_tensor[:,0]
    print(input_uer)
    model_uer = Embedding(num_user + 1, k, input_length=1)(input_uer)
    model_uer = Reshape((k,))(model_uer)

    input_movie = input_tensor[:,1]
    model_movie = Embedding(num_movie + 1, k, input_length=1)(input_movie)
    model_movie = Reshape((k,))(model_movie)

    out = Dot(1)([model_uer, model_movie])
    model = Model(inputs=[input_tensor], outputs=out)
    model.compile(loss='mse', optimizer='Adam',metrics=[RootMeanSquaredError(),MeanAbsoluteError()])
    model.summary()
    return model


# 没活了，手动调吧

# import scikeras.wrappers
#
# model = Recmand_model(num_user,num_movie,100)
# models = scikeras.wrappers.KerasClassifier(model)
# grid = GridSearchCV(estimator=models, param_grid=params, scoring='accuracy')
# grid.fit(train_x,train_y)

# result = {}
# # model = Recmand_model(num_user,num_movie,100)
# for k in [50,100,150]:
#     for i in [50,500,1000]:
#         for j in [10,20,50]:
#             model = Recmand_model(num_user, num_movie, k)
#             history=model.fit(train_x,train_y,batch_size = i,epochs =j,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)],validation_data=(valid_x,valid_y))
#             result[history.history['loss'][-1]] = [k,i,j]
#
# keys = list(result.keys())
# keys.sort()
# min_key = keys[-1]
# print('best para:{}'.format(result[min_key]))
#
# from sklearn.model_selection import KFold
# kf = KFold(n_splits=5,shuffle=True,random_state=42)
#
# data_user = data["UserID"].values
# data_movie = data["MovieID"].values
# cross_x = np.dstack((data_user,data_movie))[0]
# cross_y = data["Rating"].values
#
# fold = 1
# score_list = []
# for train_index, val_index in kf.split(cross_x):
#     # 准备数据
#     x_train_fold = cross_x[train_index]
#     y_train_fold = cross_y[train_index]
#     x_val_fold = cross_x[val_index]
#     y_val_fold = cross_y[val_index]
#     x_train_input = np.transpose(x_train_fold)
#     x_valid_input = np.transpose(x_val_fold)
#
#     # 编译模型
#     best_model = Recmand_model(num_user, num_movie, 50)
#
#     # 训练模型
#     best_model.fit(x_train_fold, y_train_fold, batch_size=1000, epochs=10,
#                    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)],
#                    validation_data=(x_val_fold, y_val_fold))
#
#     # 评估模型
#     scores = best_model.evaluate(x_val_fold, y_val_fold, verbose=0)
#     print(f"Fold {fold} : {scores}")
#     score_list.append(scores)
#     fold += 1
#
# # 5. 计算平均结果
# # scores 包含每个折叠的评分
# print(f"Mean score: {np.mean(score_list)}")
# # 这个是MAE的值

# best_model = Recmand_model(num_user, num_movie, 50)
# history=best_model.fit(train_x,train_y,batch_size = 1000,epochs =10,callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=0)],validation_data=(valid_x,valid_y))

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

predictlist = []
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


    # 预测结果
    predicts = best_model.predict(cross_x_test)

    predictlist.append(predicts)

endT = time.time()
import numpy as np
print(rmse_list)
print(mae_list)
print('Mean rmse:{}'.format(np.mean(rmse_list)))
print('Mean mae:{}'.format(np.mean(mae_list)))
print('Running time:{} s'.format(endT-startT))

u5_predict = pd.DataFrame(predictlist[-1])
u5_predict.to_csv('MF_predictions.csv', index=False)