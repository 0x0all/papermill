###############################################################################
# "Papermill"
###############################################################################

from gbm import pyMatrix, pyGradientBoostingMachine

import numpy as np
import math

import sys
import numbers
import time

class Papermill (object):
    def __init__(
        self,

        # base parameters
        num_threads = -1,
        seed = 0,

        loss_type = "mse", # "mse" or "log_loss" 
        eta = 0.3,
        max_depth = 6,
        min_child_weight = 1.0,
        lambda_ = 1.0,
        gamma = 0.0,
        subsample = 1.0,
        colsample_bytree = 1.0,

        normalize_target = False,
        gamma_zero = 1.0e-5,

        # wrapper parameters
        n_estimators = 100,
        ):

        # 
        self.num_threads = num_threads
        self.seed = seed

        self.loss_type = loss_type
        self.eta = eta
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree

        self.normalize_target = normalize_target
        self.gamma_zero = gamma_zero

        #
        self.n_estimators = n_estimators

        # internal objects
        self.num_rows = 0
        self.num_cols = 0
        self._gbm = None

        # internal states
        self.early_stopping_rounds = 0
        self.maximize = False

        self.best_iter = -1 # if best_iter > 0, it uses first n trees


    def _init(self):
        # check parameters & init internal object

        num_threads = self.num_threads
        seed = self.seed

        if self.loss_type == "mse":
            loss_type = 0
        elif self.loss_type == "log_loss":
            loss_type = 1
        elif self.loss_type == "user_defined":
            loss_type = 99
        else:
            raise ValueError( "Invalid value for loss_type.")

        eta = self.eta

        max_depth = self.max_depth

        if not self.min_child_weight > 0.0:
            raise ValueError("Invalid value for min_child_weight.")
        min_child_weight = self.min_child_weight
        
        lambda_ = self.lambda_

        gamma = self.gamma

        subsample = self.subsample

        colsample_bytree = self.colsample_bytree


        normalize_target = 1 if self.normalize_target else 0

        gamma_zero = self.gamma_zero

        # 
        self._gbm = pyGradientBoostingMachine(
            num_threads,
            seed,

            loss_type,
            eta,
            max_depth,
            min_child_weight,
            lambda_,
            gamma,
            subsample,
            colsample_bytree,

            normalize_target,
            gamma_zero
            )


    def _check_data(self, data):
        if not isinstance(data, np.ndarray):
            raise ValueError("Invalid data input. Only numpy.ndarray is allowed.")
        if not data.ndim == 2:
            raise ValueError("Invalid data input. Only 2d array is allowed.")


    def _check_label(self, label):
        if not isinstance(label, np.ndarray):
            raise ValueError("Invalid data input. Only numpy.ndarray is allowed.")
        if not label.ndim == 1:
            raise ValueError("Invalid data input. Only 1d array is allowed.")


    def fit(self, data_train, label_train):
        self._check_data(data_train)
        self._check_label(label_train)
        self._init()

        matrix_train = pyMatrix(data_train)
        self._gbm._set_data(matrix_train, label_train)

        # 
        n = self.n_estimators

        for i in range(n):
            self._gbm._boost_one_iter()

        self.best_iter = n - 1 # == last round of i


    def fit_with_early_stopping(self,data_train, label_train, data_valid, label_valid,
        eval_func, maximize=False, early_stopping_rounds=0):

        # ealry_stopping_rounds == 0 -> stop training whenever score_valid does not improve

        self.early_stopping_rounds = early_stopping_rounds
        self.maximize = maximize

        self._check_data(data_train)
        self._check_label(label_train)
        self._check_data(data_valid)
        self._check_label(label_valid)
        self._init()

        matrix_train = pyMatrix(data_train)
        matrix_valid = pyMatrix(data_valid)

        self._gbm._set_data(matrix_train, label_train)
        b = self._gbm._get_bias() # bias can be obtained after setting data

        pred_train = np.ones(label_train.shape[0], dtype=np.float32) * b
        pred_valid = np.ones(label_valid.shape[0], dtype=np.float32) * b

        # 
        n = self.n_estimators
        rl = early_stopping_rounds

        time_start = time.time()

        for i in range(n):
            self._gbm._boost_one_iter()

            pred_train += np.array(self._gbm._predict_last(matrix_train), dtype=np.float32)
            pred_valid += np.array(self._gbm._predict_last(matrix_valid), dtype=np.float32)

            if self.loss_type == "log_loss":
                score_train = eval_func(label_train, 1.0 / (1.0 + np.exp(-pred_train)))
                score_valid = eval_func(label_valid, 1.0 / (1.0 + np.exp(-pred_valid)))
            else:
                score_train = eval_func(label_train, pred_train)
                score_valid = eval_func(label_valid, pred_valid)

            if i == 0:
                best_score_train = score_train
                best_score_valid = score_valid
                best_iter = i

            if ((self.maximize and score_valid > best_score_valid) or
                (not self.maximize and score_valid < best_score_valid)):

                best_score_train = score_train
                best_score_valid = score_valid
                best_iter = i

                rl = early_stopping_rounds
            else:
                rl -= 1


            # print
            t = (time.time() - time_start) / (i + 1) # mean iter/second
            s = "[{: >4d}]  train: {:0.6f}  valid: {:0.6f}  (best: {:0.6f}, {:})" \
                .format(i, score_train, score_valid, best_score_valid, best_iter)
            if (best_iter == i):
                pass
            print(s, file=sys.stderr)


            # early stopping?
            if i == 0:
                continue
            elif (rl <= 0 and early_stopping_rounds > 0):
                print("early stopping ...", file=sys.stderr)
                break


        if (i == n-1):
            print("reached max n_estimators ...", file=sys.stderr)

        print("best iteration:", file=sys.stderr)
        print("[{: >4d}]  train: {:0.6f}  valid: {:0.6f}" \
            .format(best_iter, best_score_train, best_score_valid), file=sys.stderr)


        self.best_iter = best_iter


    def predict(self, data_test):
        self._check_data(data_test)

        matrix_test = pyMatrix(data_test)

        label_test = self._gbm.predict(matrix_test, self.best_iter + 1) # num of trees == iter + 1
        label_test = np.array(label_test, dtype=np.float32)

        if (self.loss_type == "log_loss"):
            label_test = 1.0 / (1.0 + np.exp(-label_test))

        return label_test
