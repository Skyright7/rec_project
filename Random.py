from surprise import NormalPredictor
from surprise import Dataset
from surprise.model_selection import cross_validate

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')

algo = NormalPredictor()
cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)