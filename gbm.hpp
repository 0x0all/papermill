#ifndef _DECISION_TREE
#define _DECISION_TREE

// ------------------------------------------------------------
// base tree / gradient boosting machine
// ------------------------------------------------------------

#include <vector>
#include <iostream>
#include <algorithm>
#include <random>

#include <cassert>
#include <cmath>

#include <omp.h>

#include "matrix.hpp"


// EPSI * 2.0 for finding split
// EPSI for minimal loss change
const float EPSI = 1.0e-5f;

// ------------------------------------------------------------
// auxiliary functions
// ------------------------------------------------------------

inline float calc_gain(float g, float h, float l) {
    return g * g / (h + l);
}

inline float calc_weight(float g, float h, float l) {
    return - g / (h + l);
}

// ------------------------------------------------------------
// internal structures for base tree
// ------------------------------------------------------------

class Split {
    public:
        Split() {};

        inline void update(
            int col_index,
            float feat_value,
            float gain,
            float sum_grad,
            float sum_hess
            ) {
            this->col_index = col_index;
            this->feat_value = feat_value;
            this->gain = gain;
            this->sum_grad = sum_grad;
            this->sum_hess = sum_hess;
        };

        int col_index; // index to split
        float feat_value; // split value

        float gain;
        float sum_grad;
        float sum_hess;
};


class Stat {
    public:
        float sum_grad;
        float sum_hess;

        float prev_value;
};


class Node {
    public:
        Node() : is_leaf(true) {};

        Node(
            float sum_grad,
            float sum_hess,
            float root_gain,
            float weight,
            int parent_index
            ) :
            is_leaf(true),
            sum_grad(sum_grad), 
            sum_hess(sum_hess), 
            root_gain(root_gain),
            weight(weight),
            parent_index(parent_index)
            {};

        inline void update_child (
            float sum_grad,
            float sum_hess,
            float root_gain,
            float weight,
            int parent_index
            ) {
            this->is_leaf = true;

            this->sum_grad = sum_grad;
            this->sum_hess = sum_hess;
            this->root_gain = root_gain;
            this->weight = weight;
            this->parent_index = parent_index;
        };

        inline void update_parent (
            bool is_leaf,
            int col_index,
            float feat_value,
            int child_index_0,
            int child_index_1
            ) {
            this->is_leaf = false;

            this->col_index = col_index;
            this->feat_value = feat_value;
            this->child_index[0] = child_index_0;
            this->child_index[1] = child_index_1;
        };

        inline void update_gain (
            float gain
            ) {
            this->gain = gain;
        };

        bool is_leaf;

        float sum_grad;
        float sum_hess;
        float root_gain;
        float gain;

        int col_index;
        float feat_value;
        float weight;

        int parent_index;
        int child_index[2];
};


// ------------------------------------------------------------
// base tree
// ------------------------------------------------------------

class BaseTree {
    private:
        // parameters given by gbm
        int num_rows;
        int num_cols;

        int num_threads;
        int seed;

        float eta;
        int max_depth;
        float min_child_weight;
        float lambda;
        float gamma;
        float subsample;
        float colsample_bytree;

        float gamma_zero;


        // internal state
        std::mt19937 random_state;

        std::vector<int> positions;
        std::vector<int> queues;

        std::vector<Node> nodes;

        std::vector<Split> splits;

        std::vector< std::vector<Split> > tmp_splits;
        std::vector< std::vector<Stat> > tmp_stats;

        //
        std::vector<int> child_nodes;
        std::vector<int> new_queues;
        int offset;
        int num_newnode;
        int new_offset;

        // for sampling
        int num_rows_sampled;
        int num_cols_sampled;
        std::vector<int> row_index_list;  //    0,           2,    3, ...
        std::vector<bool> row_index_mask; // true, false, true, true, ...
        std::vector<int> col_index_list;
        std::vector<bool> col_index_mask;

        // tree statistics
        int num_pruned;

    public:
        BaseTree(
            int num_rows,
            int num_cols,

            int num_threads,
            int seed,

            float eta,
            int max_depth,
            float min_child_weight,
            float lambda,
            float gamma,
            float subsample,
            float colsample_bytree,

            float gamma_zero
            ) {
            this->num_rows = num_rows;
            this->num_cols = num_cols;

            this->num_threads = num_threads;
            this->seed = seed;

            this->eta = eta;
            this->max_depth = max_depth;
            this->min_child_weight = min_child_weight;
            this->lambda = lambda;
            this->gamma = gamma;
            this->subsample = subsample;
            this->colsample_bytree = colsample_bytree;

            this->gamma_zero = gamma_zero;


            queues.reserve(      2048 );
            new_queues.reserve(  2048 );
            nodes.reserve(       2048 );
            splits.reserve(      2048 );
            child_nodes.reserve( 2048 );

            this->random_state = std::mt19937(this->seed);
        };

        ~BaseTree() {
            // !!! delete objects
        };

        inline void fit(const Matrix &matrix_full, const std::vector<float> &grad, const std::vector<float> &hess) {

            init(grad);
            stump(grad, hess);

            SubMatrix *matrix_sub = NULL;

            // Level 1 reduction
            matrix_sub = subMatrix(row_index_mask, num_rows_sampled,
                col_index_mask, col_index_list, num_cols_sampled,
                matrix_full, grad, hess);


            for (int cur_depth = 0; cur_depth < max_depth; ++cur_depth) {
                // Level 2 reduction 

                update(matrix_full, (*matrix_sub), grad, hess);

                if (queues.size() == 0) { break; }
            }

            if (gamma > gamma_zero) {
                num_pruned = prune();
            } else {
                num_pruned = 0;
            }

            delete matrix_sub;
            matrix_sub = NULL;

        }

        inline void clear() {
            positions.clear();
            queues.clear();
            new_queues.clear();
            splits.clear();
            tmp_splits.clear();
            tmp_stats.clear();
            col_index_list.clear();
            col_index_mask.clear();
            row_index_list.clear();
            row_index_mask.clear();

            positions.shrink_to_fit();
            queues.shrink_to_fit();
            new_queues.shrink_to_fit();
            splits.shrink_to_fit();
            tmp_splits.shrink_to_fit();
            tmp_stats.shrink_to_fit();
            col_index_list.shrink_to_fit();
            col_index_mask.shrink_to_fit();
            row_index_list.shrink_to_fit();
            row_index_mask.shrink_to_fit();
        };


        inline void init(const std::vector<float> &grad) {

            std::uniform_real_distribution<float> p_uniform(0.0f, 1.0f);

            // ------------------------------------------------------------
            // column samping 
            // ------------------------------------------------------------

            col_index_mask.resize(num_cols);

            while (true) {
                col_index_list.clear();
                num_cols_sampled = 0;

                //#pragma omp parallel for schedule( static )
                for (int col_index = 0; col_index < num_cols; ++col_index) {
                    if (p_uniform(random_state) > (1.0f - colsample_bytree))  {
                        col_index_list.push_back(col_index);
                        col_index_mask[col_index] = true;
                        num_cols_sampled += 1;
                    } else {
                        col_index_mask[col_index] = false;
                    }
                }

                if (num_cols_sampled > 0) {
                    break;
                }
            }

            // ------------------------------------------------------------
            // row samping 
            // ------------------------------------------------------------

            positions.resize(num_rows);
            row_index_mask.resize(num_rows);

            while (true) {
                row_index_list.clear();
                num_rows_sampled = 0;

                //#pragma omp parallel for schedule( static )
                for (int row_index = 0; row_index < num_rows; ++row_index) {
                    if (p_uniform(random_state) > (1.0f - subsample)) {
                        positions[row_index] = 0;
                        row_index_mask[row_index] = true;
                        row_index_list.push_back(row_index);
                        num_rows_sampled += 1;
                    } else {
                        positions[row_index] = -1;
                        row_index_mask[row_index] = false;
                    }
                }

                if (num_rows_sampled > 0) {
                    break;
                }
            }

            std::shuffle(col_index_list.begin(), col_index_list.end(), random_state);
        };

        inline void stump(const std::vector<float> &grad, const std::vector<float> &hess) {
            // init stump

            float sum_grad = 0.0f;
            float sum_hess = 0.0f;

            // dont' do this since it can harm reproducibility?
            //#pragma omp parallel for reduction(+:sum_grad,sum_hess,count) schedule(static)
            for (int i = 0; i < num_rows_sampled; ++i) {
                int row_index = row_index_list[i];
                sum_grad += grad[row_index];
                sum_hess += hess[row_index];
            }


            //
            Node t = Node(
                sum_grad,
                sum_hess,
                calc_gain(sum_grad, sum_hess, lambda), // root_gain
                calc_weight(sum_grad, sum_hess, lambda),
                -1 // parent_index
                );

            //
            nodes.push_back(t);

            //
            queues.clear();
            queues.push_back(0);

            offset = 0;
            num_newnode = 1;
        };


        inline void update(const Matrix &matrix_full, const SubMatrix &matrix_sub,
            const std::vector<float> &grad, const std::vector<float> &hess) {

            // (1)
            tmp_splits.resize( num_threads );
            tmp_stats.resize( num_threads );

            const int num_queues = static_cast<int>(queues.size());

            std::vector<int> queues_to_grow;
            int num_queues_to_grow;
            
            int num_nodes = static_cast<int>(nodes.size());

            // splits share index with queues
            // ex. if queue[2] is 30, splits[2]'s node_index is 30 
            splits.resize(num_queues);

            
            #pragma omp parallel
            {

            // (2) "fast touching"?  (some slides of open mp mention this)
            #pragma omp for schedule( static ) nowait
            for (int i = 0; i < num_threads; ++i) {
                const int i_thread = omp_get_thread_num();

                tmp_splits[i_thread].clear();
                tmp_splits[i_thread].shrink_to_fit();
                tmp_splits[i_thread].resize( num_newnode );

                tmp_stats[i_thread].clear();
                tmp_stats[i_thread].shrink_to_fit();
                tmp_stats[i_thread].resize( num_newnode );

                //tmp_splits[i_thread].resize( num_newnode );
                //tmp_stats[i_thread].resize( num_newnode );

                for (int i_newnode = 0; i_newnode < num_newnode; ++i_newnode) {
                    tmp_splits[i_thread][i_newnode].gain = gamma_zero;
                }
            }

            // (3)
            #pragma omp for schedule(dynamic, 1)
            for (int i_col_index = 0; i_col_index < num_cols_sampled; ++i_col_index) {
                int col_index = col_index_list[i_col_index];

                const int i_thread = omp_get_thread_num();

                for (int i_newnode = 0; i_newnode < num_newnode; ++i_newnode) {
                    tmp_stats[i_thread][i_newnode].sum_grad = 0.0f;
                    tmp_stats[i_thread][i_newnode].sum_hess = 0.0f;
                }

                //
                float gain, csum_grad, csum_hess;

                sub_col_iter itr = matrix_sub.getSubColEntryIterBegin(col_index);
                sub_col_iter itr_end = matrix_sub.getSubColEntryIterEnd(col_index);

                for (; itr != itr_end; ++itr) {

                    const int position = positions[itr->row_index];

                    if (position - offset < 0) { continue; }

                    Stat *t = &tmp_stats[i_thread][position - offset];

                    if (itr->value - t->prev_value > EPSI * 2.0f) {
                        if (t->sum_hess > min_child_weight) {
                            Node* n = &nodes[position];
                            csum_hess = n->sum_hess - t->sum_hess;

                            if (csum_hess > min_child_weight) {
                                csum_grad = n->sum_grad - t->sum_grad;

                                // do not use min_split_loss here
                                gain = calc_gain(t->sum_grad, t->sum_hess, lambda) +
                                    calc_gain(csum_grad, csum_hess, lambda) - n->root_gain;

                                if (gain > gamma_zero) {
                                    Split *u = &tmp_splits[i_thread][position - offset];

                                    // !!! sometimes the same gain can appear
                                    // so col_index must be also used for repdoducibility
                                    if ((gain > u->gain) || (gain >= u->gain && col_index < u->col_index))  {
                                        
                                        u->update(
                                            col_index,
                                            (itr->value + t->prev_value) * 0.5f, // feat_value
                                            gain,
                                            t->sum_grad,
                                            t->sum_hess
                                            );
                                    }
                                }
                            }
                        }
                    }

                    t->sum_grad += itr->grad;
                    t->sum_hess += itr->hess;
                    t->prev_value = itr->value;
                }
            }


            // (4) gather splits
            #pragma omp for schedule( static )
            for (int i_queue = 0; i_queue < num_queues; ++i_queue) {
                const int position = queues[i_queue];

                Split *s = &splits[i_queue];
                s->gain = gamma_zero;

                for (int i_thread = 0; i_thread < num_threads; ++i_thread) {
                    Split *u = &tmp_splits[i_thread][position - offset];

                    if ((u->gain > s->gain) || (u->gain >= s->gain && u->col_index < s->col_index)){
                        s->update(
                            u->col_index,
                            u->feat_value,
                            u->gain,
                            u->sum_grad,
                            u->sum_hess
                            );  
                    }
                }
            }


            // (5) preserve new node
            #pragma omp single
            {
                num_newnode = 0;
                child_nodes.resize(num_queues);

                queues_to_grow.clear();

                for (int i_queue = 0; i_queue < num_queues; ++i_queue) {
                    child_nodes[i_queue] = num_nodes + num_newnode;

                    Split *s = &splits[i_queue];
                    Node *t = &nodes[queues[i_queue]];
                    t->update_gain(s->gain);

                    if (s->gain > gamma_zero) {
                        num_newnode += 2;
                        queues_to_grow.push_back(i_queue);
                    }
                }

                new_queues.clear();
                for (int i = 0; i < num_newnode; ++i) {
                    new_queues.push_back(num_nodes + i);
                }

                new_offset = num_nodes;
                nodes.resize(num_nodes + num_newnode);

                num_queues_to_grow = static_cast<int>(queues_to_grow.size());
            }


            // (6) add new node
            #pragma omp for schedule(static)
            for (int i = 0; i < num_queues_to_grow; ++i) {
                int i_queue = queues_to_grow[i];

                Split *s = &splits[i_queue];      
                Node *t = &nodes[queues[i_queue]];

                t->update_parent(
                    false,
                    s->col_index,
                    s->feat_value,
                    child_nodes[i_queue], 
                    child_nodes[i_queue] + 1
                    );
                    
                // add new
                float g, h; // sum_grad, sum_hess

                for (int j = 0; j < 2; ++j) {
                    if (j == 0) {
                        g = s->sum_grad;
                        h = s->sum_hess;
                    } else {
                        g = (t->sum_grad - s->sum_grad);
                        h = (t->sum_hess - s->sum_hess);
                    }

                    nodes[child_nodes[i_queue] + j].update_child(
                        g, // sum_grad
                        h, // sum_hess
                        calc_gain(g, h, lambda), // root gain
                        calc_weight(g, h, lambda), // weight
                        queues[i_queue] // parent_index
                        );
                }

            }


            // (7) move samples's positions
            #pragma omp for schedule(static)
            for (int row_index = 0; row_index < num_rows; ++row_index) {
                const int position = positions[row_index];
                if (position - offset < 0) { continue; }

                Node *n = &nodes[position];

                if (!n->is_leaf) {
                    if (matrix_full.getValue(row_index, n->col_index) < n->feat_value) {
                        positions[row_index] = n->child_index[0];
                    } else {
                        positions[row_index] = n->child_index[1];
                    }
                };
            }

            }
        
            queues = new_queues;
            offset = new_offset;
        };


        inline int prune() {
            int num_nodes = static_cast<int>(nodes.size());
            int num_pruned = 0;

            for(int node_index = 0; node_index < num_nodes; ++node_index) {
                if (!nodes[node_index].is_leaf) {
                    num_pruned += prune_node(node_index, 0);
                }
            }

            return num_pruned;
        };


        inline bool is_pruned() {
            if (num_pruned > 0) { return true; } else { return false; }
        };


        inline int prune_node(int node_index, int np) {
            if (node_index < 0) { return np; }

            Node *t = &nodes[node_index];

            if (t->is_leaf) { return np; }
            if (t->gain >= gamma) { return np; }

            if (nodes[t->child_index[0]].is_leaf && nodes[t->child_index[1]].is_leaf) {
                t->is_leaf = true;
                return prune_node(t->parent_index, np+1);
            } else {
                return np;
            }
        };

        inline void predict(const Matrix &matrix, std::vector<float> &pred) {
            int num_rows = matrix.getNumRows();

            #pragma omp parallel for schedule(dynamic, 1)
            for (int row_index = 0; row_index < num_rows; ++row_index) {
                int node_index = 0;

                Node* n;

                while (true) {
                    n = &nodes[node_index];

                    if (n->is_leaf) {
                        pred[row_index] += eta * n->weight;
                        break;
                    }

                    if (matrix.getValue(row_index, n->col_index) < n->feat_value) {
                        node_index = n->child_index[0];
                    } else {
                        node_index = n->child_index[1];
                    }
                }
            }
        };


        inline void predict_cache(const Matrix &matrix, std::vector<float> &pred) {
            int num_rows = matrix.getNumRows();

            #pragma omp parallel for schedule(dynamic, 1)
            for (int row_index = 0; row_index < num_rows; ++row_index) {
                if (positions[row_index] > 0) {
                    pred[row_index] += eta * nodes[positions[row_index]].weight;
                    continue;
                }

                int node_index = 0;
                Node* n;

                while (true) {
                    n = &nodes[node_index];

                    if (n->is_leaf) {
                        pred[row_index] += eta * n->weight;
                        break;
                    }

                    if (matrix.getValue(row_index, n->col_index) < n->feat_value) {
                        node_index = n->child_index[0];
                    } else {
                        node_index = n->child_index[1];
                    }
                }
            }
        };

};

// ------------------------------------------------------------
// for calculating loss
// ------------------------------------------------------------

inline void calc_grad_hess(
    const std::vector<float> &label,
    const std::vector<float> &pred,
    std::vector<float> &grad, std::vector<float> &hess,
    const int loss_type
    ) {
    // see gbm's manual for more loss functions

    int n = static_cast<int>(label.size());

    if (loss_type == 0) {
        // mse
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            grad[i] = 2.0f * (pred[i] - label[i]);
            hess[i] = 2.0f;
        }

    } else if (loss_type == 3) {
        // log_loss
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            const float t = 1.0f / (1.0f + expf(-pred[i]));
            grad[i] = t - label[i];
            hess[i] = t * (1.0f - t);
        }

    } else if (loss_type == 99) {
        // user_defined
    }
}


// ------------------------------------------------------------
// for calculating score
// ------------------------------------------------------------


inline static bool compare_first(const std::pair<float, float> &a, const std::pair<float, float> &b) {
    // auxiliary function for roc_auc_score
    return a.first < b.first;
};


inline float calc_score(
    const std::vector<float> &label, // aka true
    const std::vector<float> &pred, // aka score
    const int eval_metric,
    const int loss_type
    ) {

    int n = static_cast<int>(label.size());

    if (eval_metric == 0)  {
        // mse
        return 0.0f;

    } else if (eval_metric == 1) {
        // rmse
        if (loss_type == 3) {
            float sum = 0.0f;

            //#pragma omp parallel for reduction(+:sum) schedule(static)
            for (int i = 0; i < n; ++i) {
                const float p = 1.0 / (1.0 + expf(-pred[i]));
                const float y = label[i];
                sum += ((p - y) * (p - y));
            }
            return sqrtf(sum / static_cast<float>(n));
        } else {
            float sum = 0.0f;

            //#pragma omp parallel for reduction(+:sum) schedule(static)
            for (int i = 0; i < n; ++i) {
                const float p = pred[i];
                const float y = label[i];
                sum += ((p - y) * (p - y));
            }
            return sqrtf(sum / static_cast<float>(n));
        }

    } else if (eval_metric == 2) {
        // mae
        return 0.0f;

    } else if (eval_metric == 3) {
        // log loss
        if (loss_type == 3) {
            const float eps = 1.0e-6f;
            float sum = 0.0f;

            //#pragma omp parallel for reduction(+:sum) schedule(static)
            for (int i = 0; i < n; ++i) {
                const float p = 1.0f / (1.0f + expf(-pred[i])); // loss_type == los_loss?
                const float py = p;
                const float pn = 1.0f - p;
                const float y  = label[i];

                if (py < eps) {
                    sum += - y * std::log(eps) - (1.0f - y) * std::log(1.0f - eps);
                } else if (pn < eps) {
                    sum += - y * std::log(1.0f - eps) - (1.0f - y) * std::log(eps);
                } else {
                    sum += - y * std::log(py) - (1.0 - y) * std::log(pn);
                }
            }

            return (sum / static_cast<float>(n));
        } else {
            return 0.0f;
        }

    } else if (eval_metric == 4) {
        // auc
        // taken from xgboost's code and modified a little bit

        std::vector< std::pair<float, float> > rec;
        const int n = static_cast<int>(pred.size());
        rec.resize(n);

        bool a = true;

        for (int i = 0; i < static_cast<int>(pred.size()); i++) {
            rec[i].first = pred[i];
            rec[i].second = label[i];

            if (a) { if ((pred[i] > 0.0f) != (label[i] > 0.5f)) { a = false; } }
        }
        if (a) { return 1.0f; }

        std::sort(rec.begin(), rec.end(), compare_first);

        double sum_auc = 0.0;
        double sum_negpair = 0.0;
        double sum_npos = 0.0;
        double sum_nneg = 0.0;
        double buf_pos = 0.0; // number of pos samples
        double buf_neg = 0.0; // number of neg samples

        for (int j = 0; j < static_cast<int>(rec.size()); ++j){
            // keep bucketing predictions in same bucket
            if (j != 0 && rec[j].first != rec[j - 1].first){ // first->pred
                sum_negpair += buf_pos * (sum_nneg + buf_neg * 0.5); // 0.5 for diagonal
                sum_npos += buf_pos;
                sum_nneg += buf_neg;
                buf_pos = 0.0;
                buf_neg = 0.0;
            }
            buf_pos += (rec[j].second);
            buf_neg += (1.0 - rec[j].second); // second->label
        }
        sum_negpair += buf_pos * (sum_nneg + buf_neg * 0.5);
        sum_npos += buf_pos;
        sum_nneg += buf_neg;

        //
        if (sum_npos > 0.0 && sum_nneg > 0.0) {
            sum_auc += sum_negpair / (sum_npos*sum_nneg);
            return static_cast<float>(sum_auc);
        } else {
            return 0.0f;
        }

    } else if (eval_metric == 99) {
        // user defined, check eval_maximize
        return 0.0f;

    } else {
        return 0.0f;

    }
};


class GradientBoostingMachine {
    public:
        std::vector<BaseTree*> tree_ptrs;

    private:
        int num_threads;
        int seed;
        int silent;

        int loss_type;
        float eta;
        int max_depth;
        float min_child_weight;
        float lambda;
        float gamma; // aka min_split_loss
        float subsample;
        float colsample_bytree;

        int normalize_target;
        float gamma_zero;

        float bias;

        Matrix *matrix_ptr;
        std::vector<float> label;
        std::vector<float> pred;

        int num_rows;
        int num_cols;

        // internal structures
        std::vector<float> grad;
        std::vector<float> hess;

        //
        Matrix *matrix_valid_ptr;
        std::vector<float> label_valid;
        std::vector<float> pred_valid;

        int num_rows_valid;

        //
        std::vector<float> scores_train;
        std::vector<float> scores_valid;

        //
        int train_mode; // 0: no monitor, 1: train, 2: train & valid

        int  eval_metric; //
        bool eval_maximize;

        int early_stopping_rounds;
        int early_stopping_count; 
        float best_score_train;
        float best_score_valid;
        int  best_round;

        int num_round;


    private:
        inline void init_num_threads(int num_threads) {
            if (num_threads < 0) {
                #pragma omp parallel
                { this->num_threads = omp_get_num_threads(); }
            } else {
                this->num_threads = num_threads;
            }

            omp_set_num_threads(this->num_threads);
            fprintf(stderr, "omp_set_num_threads: %d\n", this->num_threads);
        };


    public:
        GradientBoostingMachine(
            // master list
            int num_threads,
            int seed,
            int silent,

            int loss_type,
            float eta,
            int max_depth,
            float min_child_weight,
            float lambda,
            float gamma,
            float subsample,
            float colsample_bytree,

            int normalize_target,
            float gamma_zero,

            int num_round
            ) {
            //
            this->num_threads = num_threads;
            this->seed = seed;
            this->silent = silent;

            this->loss_type = loss_type;
            this->eta = eta;
            this->max_depth = max_depth;
            this->min_child_weight = min_child_weight;
            this->lambda = lambda;
            this->gamma = gamma;
            this->subsample = subsample;
            this->colsample_bytree = colsample_bytree;

            this->normalize_target = normalize_target;
            this->gamma_zero = gamma_zero;

            this->num_round = num_round;

            //
            init_num_threads(num_threads);
        };

        ~GradientBoostingMachine() {
            int num_trees = static_cast<int>(tree_ptrs.size());

            for (int tree_index = 0; tree_index < num_trees; ++tree_index) {
                delete tree_ptrs[tree_index];
            }
        };


        inline void set_data(Matrix &matrix, std::vector<float> &label) {
            this->matrix_ptr = &matrix;
            this->label = label;

            this->num_rows = matrix.getNumRows();
            this->num_cols = matrix.getNumCols();

            grad.resize(num_rows);
            hess.resize(num_rows);

            pred.resize(num_rows);

            //#pragma omp parallel for schedule(static)
            for (int row_index = 0; row_index < num_rows; ++row_index) {
                pred[row_index] = 0.0f;
            }

            if (this->normalize_target == 1) {
                calc_grad_hess(label, pred, grad, hess, loss_type);

                float sum_grad = 0.0f;
                float sum_hess = 0.0f;
                for (int row_index = 0; row_index < num_rows; ++row_index) {
                    sum_grad += grad[row_index];
                    sum_hess += hess[row_index];
                }

                if (sum_hess > 0.0f) {
                    bias = sum_grad / sum_hess;
                } else {
                    bias = 0.0f;
                }

            } else {
                bias = 0.0f;
            }

            //
            for (int row_index = 0; row_index < num_rows; ++row_index) {
                pred[row_index] = bias;
            }
        };


        inline void set_data_valid(Matrix &matrix_valid, std::vector<float> &label_valid) {

            this->matrix_valid_ptr = &matrix_valid;
            this->label_valid = label_valid;
            this->num_rows_valid = matrix_valid.getNumRows();

            pred_valid.resize(num_rows_valid);

            //
            for (int row_index = 0; row_index < num_rows_valid; ++row_index) {
                pred_valid[row_index] = bias;
            }
        };


        inline void set_train_mode(int train_mode) {
            this->train_mode = train_mode;

            if (train_mode > 0) {
                scores_train.resize(2048);
                scores_train.clear();
            }
            if (train_mode > 1) {
                scores_valid.resize(2048);
                scores_valid.clear();
            }
        };

        inline void set_eval_metric(int eval_metric) {
            this->eval_metric = eval_metric;

            if (eval_metric == 4) { // auc
                this->eval_maximize = true;
            } else {
                this->eval_maximize = false;
            }
        };

        inline void set_early_stopping_rounds(int early_stopping_rounds) {
            this->early_stopping_rounds = early_stopping_rounds;

            // init for early_stopping
            this->early_stopping_count = early_stopping_rounds;
        };

        inline int check_early_stopping(const int round, const float score_train, const float score_valid) {
            // dont use count. just compare round & best_round must be better?
            if (((score_valid != best_score_valid) && ((score_valid > best_score_valid) == eval_maximize)) || round == 0) {
                best_score_train = score_train;
                best_score_valid = score_valid;
                best_round = round;
                early_stopping_count = early_stopping_rounds;
                return 2;
            } else {
                early_stopping_count -= 1;
                if (early_stopping_count > 0) {
                    return 1;
                } else {
                    return 0;
                }
            }
        };

        inline void get_scores(std::vector<float> &scores, int score_type) {
            if (score_type == 0) {
                scores = scores_train;
            } else if (score_type == 1) {
                scores = scores_valid;
            }
        };

        inline int boost_one_iter(int n) {
            // for now, n is just ignored and run just one iter

            int round = static_cast<int>(tree_ptrs.size());

            calc_grad_hess(label, pred, grad, hess, loss_type);

            BaseTree *t = new BaseTree(
                num_rows,
                num_cols,

                num_threads,
                seed + round,

                eta,
                max_depth,
                min_child_weight,
                lambda,
                gamma,
                subsample, 
                colsample_bytree,

                gamma_zero
                );


            t->fit((*matrix_ptr), grad, hess);

            // train
            if (t->is_pruned()) {
                t->predict((*matrix_ptr), pred);
            } else {
                t->predict_cache((*matrix_ptr), pred);
            }


            int round_status = 0; // 0: stop, 1,2: continue

            if (train_mode == 0) {
                if (round < num_round-1) {
                    round_status= 1;
                }
                best_round = round;

            } else if (train_mode == 2) {
                float score_train, score_valid;

                t->predict((*matrix_valid_ptr), pred_valid);
                

                #pragma omp parallel sections
                {
                    #pragma omp section
                    {
                        score_train = calc_score(label, pred, eval_metric, loss_type);
                    }
                    #pragma omp section
                    {
                        score_valid = calc_score(label_valid, pred_valid, eval_metric, loss_type);
                    }
                }

                scores_train.push_back(score_train);
                scores_valid.push_back(score_valid);

                round_status = check_early_stopping(round, score_train, score_valid);

                // print
                if (silent != 1) {
                    // see termcolor (python package) for color printing
                    fprintf(stderr, "[%4d]  train: %0.6f  valid: %0.6f  (best: %0.6f, %d)\n",
                        round, score_train, score_valid, best_score_valid, best_round); // 

                    if (round == num_round-1) {
                        //fprintf(stderr, "max num_round.\n\n");
                        fprintf(stderr, "\033[31mreached max num_round.\033[0m\n\n"); // red
                    } else if (round_status == 0) {
                        //fprintf(stderr, "early stopping.\n\n");
                        fprintf(stderr, "\033[32mearly stopping.\033[0m\n\n"); // green
                    }
                }
            }

            //
            t->clear();
            tree_ptrs.push_back(t);

            return round_status; 
        };

        inline int get_best_round() {
            return best_round;
        };

        inline void predict(const Matrix &matrix, std::vector<float> &pred, int round) {
            // predict by all trees
            int num_rows = matrix.getNumRows();
            int num_trees = tree_ptrs.size();
            
            pred.resize(num_rows);

            for (int row_index = 0; row_index < num_rows; ++row_index) {
                pred[row_index] = bias;
            }

            if (round > -1) {
                num_trees = round + 1;
            } else {
                num_trees = num_trees;
            }

            for (int tree_index = 0; tree_index < num_trees; ++tree_index) {
                tree_ptrs[tree_index]->predict(matrix, pred);
            }

            if (loss_type == 3) {
                const float eps = 1.0e-6f;
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < num_rows; ++i) {
                    float p = 1.0f / (1.0f + expf(-pred[i])); 
                    p = p < eps ? eps : p;
                    p = p > 1.0f-eps ? 1.0f-eps : p;
                    pred[i] = p;
                }
            }
        };

};

#endif
