# 推荐系统尝试
## 关于cross validation数据划分：
老师提供的movielens 100k数据集中已经random有划分好的专门用于5-fold cross validation 的数据
因此就直接使用就好了。（生成的命令是mku.sh,数据集自带）
## SVD
gridCV后参数选择及按这个参数做cross validation结果：
```shell
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
gridCV后参数选择及按这个参数做cross validation结果：
```shell
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
Random 不需要做Grid因为无参可调，只需要做一下交叉验证
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
MF,因为gridCV运行稍长，代码现在是暂时将其注释掉的，如果需要运行请将那部分代码取消注释
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
经典的神经网络在推荐系统上的应用，通过embedding层跟MLP实现推荐
```shell
best para:[20, 64] epoch,batchsize
[1.3913956880569458, 1.3910679817199707, 1.3422080278396606, 1.3794997930526733, 1.342484951019287]
[1.1163194179534912, 1.1110049486160278, 1.0742131471633911, 1.109380841255188, 1.0804023742675781]
Mean rmse:1.3693312883377076
Mean mae:1.0982641458511353
Running time:51.18940997123718 s
```
