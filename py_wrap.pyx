from libcpp.vector cimport vector

cdef extern from "wrap.hpp":
    cdef cppclass Matrix
    cdef cppclass GradientBoostingMachine

    # Martrix
    Matrix* NewMatrix(
        vector[vector[float]] values,
        int num_rows,
        int num_cols
        )
    void DeleteMatrix(Matrix *matrix_ptr)

    void MatrixInit(Matrix *matrix_ptr)

    # GradientBoostingMachine
    void* NewGradientBoostingMachine(
        int num_threads,
        int seed,
        int silent,

        int loss_type,
        float eta,
        int max_depth,
        float min_child_weight,
        float lambda_,
        float gamma,
        float subsample,
        float colsample_bytree,

        int normalize_target,
        float gamma_zero,
        
        int num_round
        )
    void DeleteGradientBoostingMachine(GradientBoostingMachine *gbm_ptr)

    void GradientBoostingMachineSetData(
        GradientBoostingMachine *grm_ptr,
        Matrix *matrix_ptr,
        vector[float] label
        )
    void GradientBoostingMachineSetDataValid(
        GradientBoostingMachine *grm_ptr,
        Matrix *matrix_ptr,
        vector[float] label
        )
    void GradientBoostingMachineSetTrainMode(
        GradientBoostingMachine* gbm_ptr,
        int train_mode
        )
    void GradientBoostingMachineSetEvalMetric(
        GradientBoostingMachine* gbm_ptr,
        int eval_metric
        )
    void GradientBoostingMachineSetEarlyStoppingRounds(
        GradientBoostingMachine* gbm_ptr,
        int early_stopping_rounds
        )
    vector[float] GradientBoostingMachineGetScores(
        GradientBoostingMachine *grm_ptr,
        int score_type
        )
    int GradientBoostingMachineGetBestRound(
        GradientBoostingMachine *grm_ptr
        )

    int GradientBoostingMachineBoostOneIter(
        GradientBoostingMachine* gbm_ptr,
        int n
        )
    vector[float] GradientBoostingMachinePredict(
        GradientBoostingMachine *grm_ptr,
        Matrix *matrix_ptr,
        int r
        )


cdef class pyMatrix:
    # 
    cdef Matrix* thisptr

    def __cinit__(self, values):
        (num_rows, num_cols) = values.shape
        self.thisptr = <Matrix*> NewMatrix(values, num_rows, num_cols)

    def __dealloc__(self):
        DeleteMatrix(self.thisptr)

    def _init(self):
        MatrixInit(self.thisptr)


cdef class pyGradientBoostingMachine:
    # 
    cdef GradientBoostingMachine* thisptr

    def __cinit__(
        self,

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
        ):

        self.thisptr = <GradientBoostingMachine*> NewGradientBoostingMachine(
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

    def __dealloc__(self):
        DeleteGradientBoostingMachine(self.thisptr)

    def _set_data(self, pyMatrix matrix, vector[float] label):
        GradientBoostingMachineSetData(self.thisptr, <Matrix*> matrix.thisptr, label)

    def _set_data_valid(self, pyMatrix matrix, vector[float] label):
        GradientBoostingMachineSetDataValid(self.thisptr, <Matrix*> matrix.thisptr, label)

    def _set_train_mode(self, int train_mode):
        GradientBoostingMachineSetTrainMode(self.thisptr, train_mode)

    def _set_eval_metric(self, int eval_metric):
        GradientBoostingMachineSetEvalMetric(self.thisptr, eval_metric)

    def _set_early_stopping_rounds(self, int early_stopping_rounds):
        GradientBoostingMachineSetEarlyStoppingRounds(self.thisptr, early_stopping_rounds)

    def _get_scores(self, int score_type):
        return GradientBoostingMachineGetScores(self.thisptr, score_type)

    def _get_best_round(self):
        return GradientBoostingMachineGetBestRound(self.thisptr)

    def _boost_one_iter(self, int n = 1):
        return GradientBoostingMachineBoostOneIter(self.thisptr, n)

    def predict(self, pyMatrix matrix, int r = -1):
        return GradientBoostingMachinePredict(self.thisptr, <Matrix*> matrix.thisptr, r)
