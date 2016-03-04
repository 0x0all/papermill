#ifndef _MATRIX
#define _MATRIX

#include <iostream>
#include <vector>
#include <algorithm>

#include <cassert>
#include <omp.h>


// ----------------------------------------------------------------------------
// entry of matrices
// ----------------------------------------------------------------------------

class ColEntry {
    public:
        int row_index;
        float value;

        ColEntry() {};
        ~ColEntry() {};

        ColEntry(const ColEntry &src) {
            this->row_index = src.row_index;
            this->value = src.value;
        };

        inline bool operator < (const ColEntry& a) const {
            return value < a.value;
        };
};


// ----------------------------------------------------------------------------
// Matrix
// For now, only dense matrices are supported.
// ----------------------------------------------------------------------------

class Matrix {
    public:
        int num_rows;
        int num_cols;

        std::vector< std::vector<ColEntry> > ColEntries; // col -> row, sorted
        std::vector< std::vector<float> > values;

    public:
        Matrix() {};

        Matrix( std::vector< std::vector<float> > &values, int num_rows, int num_cols) {
            //
            this->num_rows = num_rows;
            this->num_cols = num_cols;

            this->values = values;
        };


        ~Matrix() {
        };


        inline void init() {
            // init is needed for using train
            ColEntries.resize(num_cols);

            for (int col_index = 0; col_index < num_cols; ++col_index) {
                ColEntries[col_index].resize(num_rows);
            }

            //#pragma omp parallel for schedule( dynamic, 1 )
            #pragma omp parallel for schedule( static )
            for (int col_index = 0; col_index < num_cols; ++col_index) {
                ColEntry* ce_begin = &ColEntries[col_index][0];
                ColEntry* ce = ce_begin;

                for (int row_index = 0; row_index < num_rows; ++row_index) {
                    float value = values[row_index][col_index];

                    ce->row_index = row_index;
                    ce->value = value;
                    ce++;
                }

                std::sort(ce_begin, ce);
            }
        };

        inline int getNumRows() const {
            return this->num_rows;
        }
        inline int getNumCols() const {
            return this->num_cols;
        }

        //
        inline float getValue(const int row_index, const int col_index) const {
            return values[row_index][col_index];
        }
};


// ----------------------------------------------------------------------------
// SubMatrix
// sub matrices that only used for training
// ----------------------------------------------------------------------------

class SubColEntry {
    public:
        int row_index;
        float value;
        float grad;
        float hess;

        SubColEntry() {};
        ~SubColEntry() {};

        inline void update(
            const int row_index,
            const float value,
            const float grad,
            const float hess
            ) {
            this->row_index = row_index;
            this->value = value;
            this->grad = grad;
            this->hess = hess;
        };
};

typedef std::vector<SubColEntry>::const_iterator sub_col_iter;

class SubMatrix {
    public:
        int num_rows;
        int num_cols;

        std::vector< std::vector<SubColEntry> > SubColEntries; // col -> row, sorted

    public:
        SubMatrix() {};
        ~SubMatrix() {};

        inline sub_col_iter getSubColEntryIterBegin(int col_index) const {
            return SubColEntries[col_index].begin();
        };
        inline sub_col_iter getSubColEntryIterEnd(int col_index) const {
            return SubColEntries[col_index].end();
        };
};


// subsamping matrix from full matrix
inline SubMatrix* subMatrix(
    std::vector<bool> &row_index_mask, int num_rows_sampled,
    std::vector<bool> &col_index_mask, std::vector<int> col_index_list, int num_cols_sampled,
    const Matrix &src_matrix,
    const std::vector<float> &grad, const std::vector<float> &hess) {

    SubMatrix *dst_matrix = new SubMatrix();
    dst_matrix->num_rows = num_rows_sampled;
    dst_matrix->num_cols = src_matrix.num_cols; // not num_cols_sampled

    dst_matrix->SubColEntries.resize(src_matrix.num_cols);


    for (int col_index = 0; col_index < src_matrix.num_cols; ++col_index) {
        if (!col_index_mask[col_index]) {
            dst_matrix->SubColEntries[col_index].resize(0);
        } else {
            dst_matrix->SubColEntries[col_index].resize(num_rows_sampled);
        }
    }


    #pragma omp parallel for schedule( static )
    for (int i = 0; i < num_cols_sampled; ++i) {
        int col_index = col_index_list[i];

        const ColEntry *s = &(src_matrix.ColEntries[col_index][0]);
        SubColEntry *d = &(dst_matrix->SubColEntries[col_index][0]);

        int ri;

        for (int i = 0; i < src_matrix.num_rows; ++i) {
            if (row_index_mask[s->row_index]) {
                ri = s->row_index;
                d->update(ri, s->value, grad[ri], hess[ri]);
                d++;
            }

            s++;
        }
    }

    return dst_matrix;
};

#endif
