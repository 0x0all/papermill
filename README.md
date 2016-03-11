Papermill
=====

My personal implementation of xgboost-style gradient boosting trees.


Requirements
-----

* Python 3.4.3
* Cython 0.23.4
* numpy 1.10.4


Installation
-----

```
pip install git+https://github.com/khyh/papermill.git --user
```


Example
-----

```py
from papermill import Papermill


ppm = Papermill(seed = 0, loss_type = "log_loss", eta = 0.01,
    subsample = 0.8, colsample_bytree = 0.8, num_round = 300)


# simple fit
ppm.fit(data_train, label_train)

pred_test = ppm.predict(data_test)


# early stopping
ppm.fit(
    data_train, label_train, # train data
    data_valid, label_valid, # additional data for validation
    "roc_auc_score",         # metrics to monitor. "rmse", "log_loss", "roc_auc_score" is supported
    10)                      # validation error needs to decrease at every 10 rounds

pred_test = ppm.predict(data_test) # best round is used by default
```


Parameters
-----

* num_threads [default=-1]
  - number of threads to be used.
  - maximum number of threads are used by default.
* seed [default=0]
  - random seed.
* silent [default=False]
  - be silent or not.
* loss_type ["mse" or "log_loss", default="mse"]
  - loss to minimize. "mse" or "log_loss" is supported for now.
  - aka objective.
* eta [default=0.3]
  - shrinkage parameter.
* max_depth [default=6]
  - maximum depth of trees.
* min_child_weight [default=1.0]
  - minimum sum of instance hessian needed in a child. 
* lambda_ [default=1.0]
  - L2 regularization term on weights.
* gamma [default=0.0]
  - minimum gain to make a node split while pruning a tree.
  - pruning is not performed when gamma is smaller than gamma_zero.
* subsample [default=1.0]
  - sampling ratio of training samples.
* colsample_bytree [default=1.0]
  - sampling ratio of columns used by a single tree.
* normalize_target [True or False, default=False]
  - set bias to (sum of gradient) / (sum of hessian).
  - when False, bias is set to 0.0.
* gamma_zero [default=1.0e-5]
  - minimum gain to make a node split while growing a tree.
* num_round [default=100]
  - number of maximum trees to train.


References
-----

* [dmlc/xgboost](https://github.com/dmlc/xgboost) by Distributed (Deep) Machine Learning Community
* [Parallel Gradient Boosting Decision Trees](http://zhanpengfang.github.io/418home.html) by Zhanpeng Fang
