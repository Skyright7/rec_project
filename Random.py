from surprise import NormalPredictor
import os
from surprise import accuracy, Dataset, Reader
from surprise.model_selection import PredefinedKFold
import pandas as pd

def random_5_cv():
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

    algo = NormalPredictor()

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

    u5_predict = pd.DataFrame(predictlist[-1])
    u5_predict.to_csv('rand_predictions.csv', index=False)

if __name__ == '__main__':
    random_5_cv()