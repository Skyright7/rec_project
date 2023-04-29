import pandas as pd
from sklearn.model_selection import train_test_split

names = ['userId','movieId','rating','timestamp']
data = pd.read_csv("./dataset/u.data",sep='\t',names=names)

data = data.drop(columns=['timestamp'])

import scipy.sparse as sp
#
# # 创建稀疏矩阵
# train_sparse = sp.csr_matrix((train['rating'], (train['userId'], train['movieId'])))
# test_sparse = sp.csr_matrix((test['rating'], (test['userId'], test['movieId'])))
#
from sklearn.decomposition import NMF
#
# # 设置模型参数
# model = NMF(n_components=10, init='random', random_state=0)
#
# # 将训练数据传递给模型并拟合它
# model.fit(train_sparse)
#
# # 使用模型预测测试数据
# test_predictions = model.transform(test_sparse)

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# 定义K-Fold交叉验证迭代器
kf = KFold(n_splits=5)

# 遍历K-Fold交叉验证迭代器
for train_index, test_index in kf.split(data):
    # 获取训练集和测试集数据
    train, test = data[train_index], data[test_index]

    # 创建稀疏矩阵
    train_sparse = sp.csr_matrix((train['rating'], (train['userId'], train['movieId'])))
    test_sparse = sp.csr_matrix((test['rating'], (test['userId'], test['movieId'])))

    model = NMF(n_components=10, init='random', random_state=0)
    # 拟合模型
    model.fit(train_sparse)

    # 预测测试数据
    test_predictions = model.transform(test_sparse)

    # 计算均方误差
    mse = mean_squared_error(test_sparse, test_predictions)

    # 打印均方误差
    print("Mean Squared Error: ", mse)

