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

    # GradientBoostingMachine
    void* NewGradientBoostingMachine(
        int num_threads,
        int seed,

        int loss_type,
        float eta,
        int max_depth,
        float min_child_weight,
        float lambda_,
        float gamma,
        float subsample,
        float colsample_bytree,

        int normalize_target,
        float gamma_zero
        )
    void DeleteGradientBoostingMachine(GradientBoostingMachine *gbm_ptr)

    void GradientBoostingMachineSetData(
        GradientBoostingMachine *grm_ptr,
        Matrix *matrix_ptr,
        vector[float] label
        )
    void GradientBoostingMachineBoostOneIter(
        GradientBoostingMachine* gbm_ptr,
        int n
        )
    float GradientBoostingMachineGetBias(
        GradientBoostingMachine *grm_ptr
        )
    vector[float] GradientBoostingMachinePredictLast(
        GradientBoostingMachine *grm_ptr,
        Matrix *matrix_ptr
        )
    vector[float] GradientBoostingMachinePredict(
        GradientBoostingMachine *grm_ptr,
        Matrix *matrix_ptr,
        int l
        )


cdef class pyMatrix:
    # 
    cdef Matrix* thisptr

    def __cinit__(self, values):
        (num_rows, num_cols) = values.shape
        self.thisptr = <Matrix*> NewMatrix(values, num_rows, num_cols)

    def __dealloc__(self):
        DeleteMatrix(self.thisptr)



cdef class pyGradientBoostingMachine:
    # 
    cdef GradientBoostingMachine* thisptr

    def __cinit__(
        self,

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
        ):

        self.thisptr = <GradientBoostingMachine*> NewGradientBoostingMachine(
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

    def __dealloc__(self):
        DeleteGradientBoostingMachine(self.thisptr)

    def _set_data(self, pyMatrix matrix, vector[float] label):
        GradientBoostingMachineSetData(self.thisptr, <Matrix*> matrix.thisptr, label)

    def _boost_one_iter(self, int n = 1):
        GradientBoostingMachineBoostOneIter(self.thisptr, n)

    def _get_bias(self):
        return GradientBoostingMachineGetBias(self.thisptr)

    def _predict_last(self, pyMatrix matrix):
        return GradientBoostingMachinePredictLast(self.thisptr, <Matrix*> matrix.thisptr)

    def predict(self, pyMatrix matrix, int l = -1):
        return GradientBoostingMachinePredict(self.thisptr, <Matrix*> matrix.thisptr, l)

