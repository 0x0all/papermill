###############################################################################
# User Interface
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
        silent = False,

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
        num_round = 100,
        ):

        # 
        self.num_threads = num_threads
        self.seed = seed
        self.silent = silent

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
        self.num_round = num_round

        # internal objects
        self.num_rows = 0
        self.num_cols = 0
        self._gbm = None

        # internal states
        self.early_stopping_rounds = 0
        self.eval_metric = 0

        self.best_round = 0


    def _init(self):
        # check parameters & init internal object

        num_threads = self.num_threads
        seed = self.seed

        silent = 1 if self.silent else 0

        if self.loss_type == "mse":
            loss_type = 0
        elif self.loss_type == "log_loss":
            loss_type = 3
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


        num_round = self.num_round
        # 
        self._gbm = pyGradientBoostingMachine(
            num_threads,
            seed,
            silent,

            loss_type,
            eta,
            max_depth,
            min_child_weight,
            lambda_,
            gamma,
            subsample,
            colsample_bytree,

            normalize_target,
            gamma_zero,

            num_round
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

    def _is_none(self, data):
        if not isinstance(data, np.ndarray):
            return True
        

    def fit(self, data_train, label_train,
        data_valid=None, label_valid=None, eval_metric="rmse", early_stopping_rounds=5):


        if (self._is_none(data_valid) or self._is_none(label_valid)):
            train_mode = 0 # just train, don't monitor

            self._check_data(data_train)
            self._check_label(label_train)
            self._init()

            matrix_train = pyMatrix(data_train)
            matrix_train._init()

            self._gbm._set_data(matrix_train, label_train)
            self._gbm._set_train_mode(train_mode)

            # 
            n = self.num_round
            for i in range(n):
                self._gbm._boost_one_iter()

            #self.best_round = n-1
            self.best_round = self._gbm._get_best_round()

        else:
            train_mode = 2 # fit with early stopping

            self._check_data(data_train)
            self._check_label(label_train)
            self._check_data(data_valid)
            self._check_label(label_valid)

            self._init()

            # init for early stopping
            if eval_metric == "mse":
                #eval_metric = 0
                raise NotImplementedError
            elif eval_metric == "rmse":
                eval_metric = 1
            elif eval_metric  == "mae":
                #eval_metric = 2
                raise NotImplementedError
            elif eval_metric  == "log_loss":
                eval_metric = 3
            elif eval_metric == "roc_auc_score":
                eval_metric = 4
            elif eval_metric == "user_defined":
                eval_metric = 99
            else:
                raise ValueError( "Invalid value for eval_metric.")

            if (early_stopping_rounds == 0):
                early_stopping_rounds = self.num_round + 1

            matrix_train = pyMatrix(data_train)
            matrix_train._init() 
            matrix_valid = pyMatrix(data_valid)

            self._gbm._set_data(matrix_train, label_train)
            self._gbm._set_data_valid(matrix_valid, label_valid)
            self._gbm._set_train_mode(train_mode)
            self._gbm._set_eval_metric(eval_metric)
            self._gbm._set_early_stopping_rounds(early_stopping_rounds)

            #
            n = self.num_round
            for i in range(n):
                round_status = self._gbm._boost_one_iter()
                if (round_status <= 0):
                    break

            self.best_round = self._gbm._get_best_round()

    def predict(self, data_test):
        self._check_data(data_test)

        matrix_test = pyMatrix(data_test)

        label_test = self._gbm._predict(matrix_test, self.best_round)
        label_test = np.array(label_test, dtype=np.float32)

        return label_test
