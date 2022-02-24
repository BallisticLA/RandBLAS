#include "rblas/sjlts.hh"
#include "Random123/threefry.h"

namespace rblas::sjlts {

void fill_colwise(SJLT sjl, uint64_t seed_key, uint64_t seed_ctr) {
    // Use Fisher-Yates

    // Load shorter names into the workspace
    uint64_t k = sjl.vec_nnz;
    uint64_t n_rows = sjl.n_rows;
    uint64_t n_cols = sjl.n_cols; 
    double *vals = sjl.vals; // point to array of length nnz 
    uint64_t *cols = sjl.cols; // point to array of length nnz.
    uint64_t *rows = sjl.rows; // point to array of length nnz.

    // Define variables needed in the main loop
    uint64_t i, j, ell, swap, offset;
    uint64_t row_work[n_rows];
    for (j = 0; j < n_rows; ++j) {
        row_work[j] = j;
    }
    uint64_t pivots[k];
    typedef r123::Threefry2x64 CBRNG;
	CBRNG::key_type key = {{seed_key}};
	CBRNG::ctr_type ctr = {{seed_ctr, 0}};
    CBRNG::ctr_type randpair;
	CBRNG g;

    for (i = 0; i < n_cols; ++i)
    {
        offset = i * k;
        for (j = 0; j < k; ++j)
        {
            // one step of Fisher-Yates shuffling
            ctr[0] = seed_ctr + offset + j;
            randpair = g(ctr, key);
            ell = j + randpair.v[0] % (n_rows - j);            
            pivots[j] = ell;
            swap = row_work[ell];
            row_work[ell] = row_work[j];
            row_work[j] = swap;
                   
            // update (rows, cols, vals)
            rows[j + offset] = swap;             
            if (randpair.v[1] % 2 == 0)
            {
                vals[j + offset] = 1.0;
            }
            else 
            {
                vals[j + offset] = -1.0;
            }
            cols[j + offset] = i;
        }
        // restore row_work for next iteration of Fisher-Yates
        for (j = 1; j <= k; ++j)
        {
            int jj = k - j;
            swap = rows[jj + offset];
            ell = pivots[jj];
            row_work[jj] = row_work[ell];
            row_work[ell] = swap;
        }
    }
    return;
}

void print_sjlt(SJLT sjl)
{
    std::cout << "SJLT information" << std::endl;
    std::cout << "\tn_rows = " << sjl.n_rows << std::endl;
    std::cout << "\tn_cols = " << sjl.n_cols << std::endl;
    uint64_t nnz;
    if (sjl.ori == ColumnWise)
    {
        std::cout << "\torientation: ColumnWise" << std::endl;
        nnz = sjl.vec_nnz * sjl.n_cols;
    }
    else
    {
        std::cout << "\tOrientation: RowWise" << std::endl;
        nnz = sjl.vec_nnz *  sjl.n_rows;
    }
    std::cout << "\tvector of row indices\n\t\t";
    for (uint64_t i = 0; i < nnz; ++i) {
        std::cout << sjl.rows[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of column indices\n\t\t";
    for (uint64_t i = 0; i < nnz; ++i) {
        std::cout << sjl.cols[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "\tvector of values\n\t\t";
    for (uint64_t i = 0; i < nnz; ++i) {
        std::cout << sjl.vals[i] << ", ";
    }
    std::cout << std::endl;
}

} // end namespace rblas::sjlts
