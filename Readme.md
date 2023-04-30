# Recommend system demo
## Environment
if you want to run the code by yourself. You need some module for install
```python
import surprise
import keras
import pandas
import numpy
import sklearn
```
Just use pip install command to install the module you do not have
special reminder: if your python version is 3.10. you may face problem in pip install surprise,because the 
python 3.10 wheel does not available now.
my python version is 3.9, but for this code, i think python 3.7 above should all be work.

## about Dataset：
The data set is Movielens 100k. And only use part of it.
For normal using the whole rating data u.data. But, at 5-fold 
cross validation part. This data set already have prepared data for 5-fold cross validation.
This can make all the model I implement have the same source data at the validation part.

## about Running:
You can just find the python file named as the model name like: KNN.py, embMLP.py
,you can just using python command to run it for re-run my code.

## about parameter choice:
All the parameter are choice the best one by GridSearch.
## SVD
result:
```shell
parameter:
{'n_epochs': 20, 'lr_all': 0.004, 'reg_all': 0.04, 'n_factors': 100}
5-fold:
RMSE: 0.9540
MAE:  0.7540
RMSE: 0.9410
MAE:  0.7419
RMSE: 0.9336
MAE:  0.7369
RMSE: 0.9314
MAE:  0.7366
RMSE: 0.9342
MAE:  0.7421
Mean rmse:0.9388394767892786
Mean mae:0.7423030318093865
Running time:2.388345718383789 s
```
## KNN
result:
```shell
parameter:
{'k': 19}
RMSE: 0.9880
MAE:  0.7812
RMSE: 0.9845
MAE:  0.7728
RMSE: 0.9760
MAE:  0.7691
RMSE: 0.9713
MAE:  0.7675
RMSE: 0.9795
MAE:  0.7756
Mean rmse:0.9798738334709366
Mean mae:0.7732423842029357
Running time:4.765560865402222 s
```
## Random
for the random model are base model, do not have hyperparameter.
So, just do the 5-fold cross validatiom:
```shell
RMSE: 1.5318
MAE:  1.2303
RMSE: 1.5300
MAE:  1.2280
RMSE: 1.5194
MAE:  1.2226
RMSE: 1.5081
MAE:  1.2120
RMSE: 1.5187
MAE:  1.2158
Mean rmse:1.521602026777218
Mean mae:1.2217613877804665
Running time:0.9276118278503418 s
```
## MF
Matrix factorization with regularization:
```shell
bast param: k = 50, batchsize = 1000, epoch = 10.
RMSE:
[0.9845523238182068, 0.9721872806549072, 0.9641396403312683, 0.9637541174888611, 0.9709964990615845]
MAE:
[0.7726186513900757, 0.7632729411125183, 0.7580324411392212, 0.7586801052093506, 0.7679036259651184]
Mean rmse:0.9711259722709655
Mean mae:0.7641015529632569
Running time:5.521119117736816 s
```
## Embedding + MLP
A famous and traditional neural network.
Which is embedding + Mlp architecture.
```shell
best para:[20, 64] epoch,batchsize
[1.3913956880569458, 1.3910679817199707, 1.3422080278396606, 1.3794997930526733, 1.342484951019287]
[1.1163194179534912, 1.1110049486160278, 1.0742131471633911, 1.109380841255188, 1.0804023742675781]
Mean rmse:1.3693312883377076
Mean mae:1.0982641458511353
Running time:51.18940997123718 s
```

## directly compare all these different algorithms (round to four decimal places)(RMSE)
```shell
           KNN  Random     SVD      MF  EmbMLP
KNN     0.0000  1.1793  0.3809  0.4355  0.9971
Random  1.1793  0.0000  1.1807  1.2344  1.3636
SVD     0.3809  1.1807  0.0000  0.3372  1.0055
MF      0.4355  1.2344  0.3372  0.0000  1.0161
EmbMLP  0.9971  1.3636  1.0055  1.0161  0.0000
```
The method to do this is just letting all the model train on same data.
At here is u5.base.
Then using the trained model to do the prediction on the related test data.
Here is u5.test.
Then, for each model we get a "rating" result on test data.
We store all this rating data in the csv file by pandas model by model.
(you can see the data like KNN_predictions.csv in root)
Then, use pandas and numpy to calculate all this rating data, by one model's rating as "actual",
another as "predict". To calculate the ralated RMSE. Finally, draw this table.

