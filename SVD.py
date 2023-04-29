from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# 数据集划分
trainset, testset = train_test_split(data, test_size=0.2)

# 设置参数选取范围
param_grid = {'n_epochs':[n for n in range(10,25,5)],"lr_all":[n/1000 for n in range(2,5)],"reg_all":[n/100 for n in range(1,5)]}

gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)

gs.fit(data)

# best RMSE score
print(gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print(gs.best_params["rmse"])
# # Use the famous SVD algorithm.
# algo = SVD()
#
# # Run 5-fold cross-validation and print results.
# cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)