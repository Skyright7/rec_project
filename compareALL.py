import pandas as pd
import keras

import numpy as np

def rmse(actual, predicted):
    # actual = np.array(actual)
    # predicted = np.array(predicted)

    # 计算差值并求平方
    errors = (actual - predicted) ** 2

    # 求平方和并计算RMSE
    return np.sqrt(np.mean(errors))


if __name__ == '__main__':
    df_knn = pd.read_csv('KNN_predictions.csv',header=None)
    knn_arr = np.asarray(df_knn)

    df_rand = pd.read_csv('rand_predictions.csv',header=None)
    rand_arr = np.asarray(df_rand)

    df_svd = pd.read_csv('SVD_predictions.csv',header=None)
    svd_arr = np.asarray(df_svd)

    df_mf = pd.read_csv('MF_predictions.csv',header=None)
    mf_arr = np.asarray(df_mf)

    df_embMlp = pd.read_csv('emb_predictions.csv',header=None)
    embMlp_arr = np.asarray(df_embMlp)

    arrlist = [knn_arr,rand_arr,svd_arr,mf_arr,embMlp_arr]

    n = len(arrlist)
    output = []

    for i in range(n):
        temp = []
        for j in range(n):
            if i == j:
                temp.append(round(0.0,4))
            else:
                val = rmse(arrlist[i],arrlist[j])
                temp.append(round(val,4))
        output.append(temp)

    output = np.asarray(output)
    name = ['KNN','Random','SVD','MF','EmbMLP']
    df_out = pd.DataFrame(output,index=name,columns=name)
    print(df_out)


