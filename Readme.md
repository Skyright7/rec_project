# 推荐系统尝试
## SVD
gridCV后参数选择及按这个参数做cross validation结果：
```shell
0.9361994312022865
{'n_epochs': 20, 'lr_all': 0.004, 'reg_all': 0.04}
Evaluating RMSE, MAE of algorithm SVD on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.9316  0.9353  0.9342  0.9376  0.9431  0.9364  0.0039  
MAE (testset)     0.7374  0.7388  0.7409  0.7404  0.7462  0.7407  0.0030  
Fit time          0.32    0.32    0.32    0.33    0.32    0.32    0.00    
Test time         0.06    0.06    0.06    0.06    0.06    0.06    0.00   
```
## KNN
gridCV后参数选择及按这个参数做cross validation结果：
```shell
0.9763488871120598
{'k': 19}
Evaluating RMSE, MAE of algorithm KNNBasic on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    0.9769  0.9831  0.9686  0.9829  0.9732  0.9769  0.0056  
MAE (testset)     0.7716  0.7759  0.7618  0.7770  0.7652  0.7703  0.0059  
Fit time          0.07    0.07    0.07    0.07    0.07    0.07    0.00    
Test time         0.78    0.78    0.78    0.78    0.78    0.78    0.00 
```
## Random
Random 不需要做Grid因为无参可调，只需要做一下交叉验证
```shell
Evaluating RMSE, MAE of algorithm NormalPredictor on 5 split(s).

                  Fold 1  Fold 2  Fold 3  Fold 4  Fold 5  Mean    Std     
RMSE (testset)    1.5278  1.5156  1.5216  1.5171  1.5187  1.5201  0.0043  
MAE (testset)     1.2288  1.2145  1.2211  1.2181  1.2223  1.2209  0.0048  
Fit time          0.03    0.03    0.03    0.03    0.03    0.03    0.00    
Test time         0.04    0.04    0.04    0.04    0.04    0.04    0.00    
```
