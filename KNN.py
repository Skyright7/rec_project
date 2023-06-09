from surprise import KNNBasic
from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
import os
from surprise import accuracy, Dataset, Reader
from surprise.model_selection import PredefinedKFold
import pandas as pd

def KNN_param():
    # Load the movielens-100k dataset (download it if needed).
    data = Dataset.load_builtin('ml-100k')

    trainset, testset = train_test_split(data, test_size=0.2,random_state=40)

    param_grid = {'k':[n for n in range(5,20)]}

    gs = GridSearchCV(KNNBasic, param_grid, measures=["rmse", "mae"], cv=5)

    gs.fit(data)

    # best RMSE score
    print(gs.best_score["rmse"])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params["rmse"])

    print(gs.best_score["mae"])

    print(gs.best_params["mae"])

    return gs.best_params["rmse"] # best result is k=19


def KNN_5_cv(best_k):
    # path to dataset folder
    files_dir = os.path.expanduser("./dataset/fold_data/")

    # This time, we'll use the built-in reader.
    reader = Reader("ml-100k")

    # folds_files is a list of tuples containing file paths:
    # [(u1.base, u1.test), (u2.base, u2.test), ... (u5.base, u5.test)]
    train_file = files_dir + "u%d.base"
    test_file = files_dir + "u%d.test"
    folds_files = [(train_file % i, test_file % i) for i in (1, 2, 3, 4, 5)]

    data = Dataset.load_from_folds(folds_files, reader=reader)
    pkf = PredefinedKFold()

    algo = KNNBasic(k=best_k)

    rmse_list = []
    mae_list = []

    import time
    startT = time.time()

    predictlist = []
    for trainset, testset in pkf.split(data):

        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)
        predictlist.append([pred.est for pred in predictions])
        # Compute and print Root Mean Squared Error
        rmse_list.append(accuracy.rmse(predictions, verbose=True))
        mae_list.append(accuracy.mae(predictions, verbose=True))

    endT = time.time()
    import numpy as np
    print('Mean rmse:{}'.format(np.mean(rmse_list)))
    print('Mean mae:{}'.format(np.mean(mae_list)))
    print('Running time:{} s'.format(endT-startT))

    # out put the last fold which is u5.test 's prediction rating out to a csv file for last compare
    u5_predict = pd.DataFrame(predictlist[-1])
    u5_predict.to_csv('KNN_predictions.csv', index=False)

if __name__ == '__main__':
    # KNN_param()
    # k=19 best
    KNN_5_cv(19)