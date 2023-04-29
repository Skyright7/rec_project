import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares

from surprise import Dataset
from surprise.model_selection import train_test_split
names = ['user_id','item_id','rate','timestamp']
data = pd.read_csv("./dataset/u.data",sep='\t',names=names)

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(data.head())

# 将训练数据变为User_item矩阵的形式(这个矩阵应该现在是稀疏的)
n_user = train_data['user_id'].unique()
n_item = test_data['item_id'].unique()

user_items = pd.DataFrame(index=n_user,columns=n_item)

for i in range(len(data.columns)):
    row_i = data.iloc[i]
    user_id = row_i['user_id']
    item_id = row_i['item_id']
    rating = row_i['rate']
    user_items.loc[user_id,item_id] = rating

print(user_items.head())

als = AlternatingLeastSquares()
als.fit(user_items=user_items)
user_vecs = als.user_factors
item_vecs = als.item_factors