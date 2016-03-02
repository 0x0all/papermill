Papermill
=====

My personal implementation of xgboost-style gradient boosting trees.


Requirements
-----

* Python 3.4.3
* Cython 0.23.4
* numpy 1.10.4

`make` to build. Append the root directory to `sys.path` to use.


Parameters
-----

* num_threads [default=-1]
  - number of threads to be used.
  - maximum number of threads are used by default.
* seed [default=0]
  - random seed.
* loss_type ["mse" or "log_loss", default="mse"]
  - loss to minimize. "mse" or "log_loss" is supported for now.
  - aka objective.
* eta [default=0.3]
  - shrinkage parameter.
* max_depth [default=6]
  - maximum depth of trees.
* lambda [default=1.0]
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


Example
-----

```py

import sys

sys.path.append('papermill/')
from papermill import Papermill


ppm = Papermill(seed = 0, loss_type = "log_loss", eta = 0.01,
    subsample = 0.8, colsample_bytree = 0.8, n_estimators = 300)


# simple fit
ppm.fit(data_train, label_train)

pred_test = ppm.predict(data_test)


# early stopping
from sklearn import metrics

ppm.fit_with_early_stopping(
    data_train, label_train,
    data_valid, label_valid,       # additional data for validation
    eval_func = metrics.log_loss,  # metrics to monitor
    maximize = False,              # direction of the above metrics
    early_stopping_rounds = 10)    # validation error needs to decrease at every 10 rounds

pred_test = ppm.predict(data_test)
```


References
-----

* [dmlc/xgboost](https://github.com/dmlc/xgboost)
