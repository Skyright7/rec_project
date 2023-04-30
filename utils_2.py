from surprise import Dataset
from surprise.model_selection import train_test_split

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# 数据集划分
trainset, testset = train_test_split(data, test_size=0.2,random_state=40)

import numpy as np
np.savetxt("train_split",trainset)