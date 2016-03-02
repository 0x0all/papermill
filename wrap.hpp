// ----------------------------------------------------------------------------
// wrapper for cython
// ----------------------------------------------------------------------------


#include <vector>

#include "matrix.hpp"
#include "gbm.hpp"

// ----------------------------------------------------------------------------
// matrix/array
// ----------------------------------------------------------------------------

Matrix* NewMatrix(std::vector< std::vector<float> > values, int num_rows, int num_cols) {
    Matrix *matrix_ptr = new Matrix(values, num_rows, num_cols);
    return matrix_ptr;
};

void DeleteMatrix(Matrix *matrix_ptr) {
    delete matrix_ptr;
};


// ----------------------------------------------------------------------------
// gbm
// ----------------------------------------------------------------------------

GradientBoostingMachine* NewGradientBoostingMachine(
    int num_threads,
    int seed,

    int loss_type,
    float eta,
    int max_depth,
    float min_child_weight,
    float lambda_, // for python namespace
    float gamma,
    float subsample,
    float colsample_bytree,

    int normalize_target, 
    float gamma_zero

    ) {

    GradientBoostingMachine *gbm_ptr = new GradientBoostingMachine(
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
        );

    return gbm_ptr;
};


void DeleteGradientBoostingMachine(GradientBoostingMachine *gbm_ptr) {
    delete gbm_ptr;
};


void GradientBoostingMachineSetData(
    GradientBoostingMachine *gbm_ptr,
    Matrix *matrix_ptr,
    std::vector<float> label
    ) {
    gbm_ptr->set_data(*matrix_ptr, label);
};


void GradientBoostingMachineBoostOneIter(
    GradientBoostingMachine *gbm_ptr,
    int n
    ) {
    gbm_ptr->boost_one_iter(n);
}


float GradientBoostingMachineGetBias(
    GradientBoostingMachine *gbm_ptr
    ) {

    return gbm_ptr->get_bias();
};


std::vector<float> GradientBoostingMachinePredictLast(
    GradientBoostingMachine *gbm_ptr,
    Matrix *matrix_ptr
    ) {

    std::vector<float> p;
    gbm_ptr->predict_last(*matrix_ptr, p);

    return p;
};


std::vector<float> GradientBoostingMachinePredict(
    GradientBoostingMachine *gbm_ptr,
    Matrix *matrix_ptr,
    int l
    ) {

    std::vector<float> p;
    gbm_ptr->predict(*matrix_ptr, p, l);

    return p;
};
