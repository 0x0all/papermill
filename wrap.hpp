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

void MatrixInit(
    Matrix *matrix_ptr
    ) {
    matrix_ptr->init();
};


// ----------------------------------------------------------------------------
// gbm
// ----------------------------------------------------------------------------

GradientBoostingMachine* NewGradientBoostingMachine(
    int num_threads,
    int seed,
    int silent,

    int loss_type,
    float eta,
    int max_depth,
    float min_child_weight,
    float lambda_, // for python namespace
    float gamma,
    float subsample,
    float colsample_bytree,

    int normalize_target, 
    float gamma_zero,

    int num_round
    ) {

    GradientBoostingMachine *gbm_ptr = new GradientBoostingMachine(
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


void GradientBoostingMachineSetDataValid(
    GradientBoostingMachine *gbm_ptr,
    Matrix *matrix_ptr,
    std::vector<float> label
    ) {
    gbm_ptr->set_data_valid(*matrix_ptr, label);
};


void GradientBoostingMachineSetTrainMode(
    GradientBoostingMachine *gbm_ptr,
    int train_mode
    ) {
    gbm_ptr->set_train_mode(train_mode);
};


void GradientBoostingMachineSetEvalMetric(
    GradientBoostingMachine *gbm_ptr,
    int eval_metric
    ) {
    gbm_ptr->set_eval_metric(eval_metric);
};


void GradientBoostingMachineSetEarlyStoppingRounds(
    GradientBoostingMachine *gbm_ptr,
    int early_stopping_rounds
    ) {
    gbm_ptr->set_early_stopping_rounds(early_stopping_rounds);
};


std::vector<float> GradientBoostingMachineGetScores(
    GradientBoostingMachine *gbm_ptr,
    int score_type
    ) {

    std::vector<float> s;
    gbm_ptr->get_scores(s, score_type);

    return s;
};


int GradientBoostingMachineGetBestRound(
    GradientBoostingMachine *gbm_ptr
    ) {
    return gbm_ptr->get_best_round();
};


int GradientBoostingMachineBoostOneIter(
    GradientBoostingMachine *gbm_ptr,
    int n
    ) {
    return gbm_ptr->boost_one_iter(n);
};


std::vector<float> GradientBoostingMachinePredict(
    GradientBoostingMachine *gbm_ptr,
    Matrix *matrix_ptr,
    int r
    ) {

    std::vector<float> p;
    gbm_ptr->predict(*matrix_ptr, p, r);

    return p;
};
