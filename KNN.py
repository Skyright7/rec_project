from surprise import KNNBasic
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

# 数据集划分
trainset, testset = train_test_split(data, test_size=0.2,random_state=40)

# 设置参数选取范围
param_grid = {'k':[n for n in range(5,20)]}

gs = GridSearchCV(KNNBasic, param_grid, measures=["rmse", "mae"], cv=5)

gs.fit(data)

# best RMSE score
print(gs.best_score["rmse"])

# combination of parameters that gave the best RMSE score
print(gs.best_params["rmse"])

print(gs.best_score["mae"])

print(gs.best_params["mae"])

# 使用GridSearch找到的最优参数训练模型并储存，后续的crossValidation可以几个模型一起做
algo = KNNBasic(k=gs.best_params["rmse"]['k'])
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)

