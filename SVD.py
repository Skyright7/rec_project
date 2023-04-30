from surprise.model_selection import train_test_split
from surprise.model_selection import GridSearchCV
import os
from surprise import accuracy, Dataset, Reader, SVD
from surprise.model_selection import PredefinedKFold
import time
import pandas as pd
def SVD_param():
    # Load the movielens-100k dataset (download it if needed).
    data = Dataset.load_builtin('ml-100k')

    trainset, testset = train_test_split(data, test_size=0.2,random_state=40)

    param_grid = {'n_epochs':[n for n in range(10,25,5)],
                  "lr_all":[n/1000 for n in range(2,5)],
                  "reg_all":[n/100 for n in range(1,5)],
                  "n_factors":[50,100,200]}

    gs = GridSearchCV(SVD, param_grid, measures=["rmse", "mae"], cv=5)

    gs.fit(data)

    # best RMSE score
    print(gs.best_score["rmse"])

    # combination of parameters that gave the best RMSE score
    print(gs.best_params["rmse"])
    return gs.best_params["rmse"]

def SVD_5_cv(param:dict):

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

    algo = SVD(n_epochs=param['n_epochs'],lr_all=param['lr_all'],
               reg_all=param['reg_all'],n_factors=param['n_factors'])

    rmse_list = []
    mae_list = []

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
    u5_predict = pd.DataFrame(predictlist[-1])
    u5_predict.to_csv('SVD_predictions.csv', index=False)


if __name__ == '__main__':
    param_result = {'n_epochs': 20, 'lr_all': 0.004, 'reg_all': 0.04, 'n_factors': 100} # run SVD_param() get this
    SVD_5_cv(param_result)