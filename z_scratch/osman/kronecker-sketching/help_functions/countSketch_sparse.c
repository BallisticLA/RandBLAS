/*
 *Calling sequence:
 * Pt = countSketch_sparse( At, indx_map, mSmall, s )
 * 
 * A is a mBig x N matrix and At is its transpose
 * P is a mSmall x N matrix, and Pt is its transpose
 * indx_map     is a int64 vector of length mBig where each entry
 *      is an integer in [1,mSmall] (i.e., 1-based indexing, Matlab-style)
 * s is a row or column vector containing the sign function. This should be
 * 		a double array. It should be of length mBig.
 *
 * Implements the "CountSketch"
 * as known in the data streaming literature
 * (e.g., 7] Moses Charikar, Kevin Chen, and Martin Farach-Colton. 
 * "Finding frequent items in data streams". Theor.
 *  Comput. Sci., 312(1):3–15, 2004 )
 *
 * In the compressed least squares literature, this was analyzed in
 *
 * "Low Rank Approximation and Regression in Input Sparsity Time"
 * Kenneth L. Clarkson, David P. Woodruff
 * http://arxiv.org/abs/1207.6365
 * STOC '13, co-winner of best paper award
 *
 * Computational complexity is nnz(A)
 *
 * */

/*  Implementation details
 * Compile with just:  mex -largeArrayDims -O countSketch_sparse.c
 *
 * For efficiency, since Matlab uses column-major order,
 * the input should be At ( = A') and NOT A
 * Likewise, the output is Pt ( = P' = (Sketch(A))' )
 *
 * This version of the code works with sparse or dense input matrices
 * "A". Output is always dense regardless of input sparsity.
 *
 * Warning: the code does not do error checks, so it can easily crash.
 * Make sure that the "indx_map" is of type int64
 *
 * Stephen Becker, stephen.becker@colorado.edu, June 21 2015
 * The use of CountSketch was suggested by Haim Avron
 **/
#if defined(__GNUC__) && !(defined(__clang__)) 
#include <uchar.h>
#endif
#include <math.h>
#include "mex.h"

/* NOTE 1: This code uses int64_T for indx_map to ensure that the size 
 * is 8 bytes on all systems.
**/

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *At,*Pt; 
	int64_T *indx_map;
    mwSize mBig,mSmall,n, i,j,k;
    mwIndex *ir, *jc;
    double *a, *s;
    
    /* Check for proper number of arguments */
    if (nrhs != 4) { 
	    mexErrMsgIdAndTxt( "MATLAB:countSketch_sparse:invalidNumInputs",
                "Four input arguments required."); 
    } else if (nlhs > 1) {
	    mexErrMsgIdAndTxt( "MATLAB:countSketch_sparse:maxlhs",
                "Too many output arguments."); 
    } 
    if ( !(mxIsInt64(prhs[1])) ) {
        mexErrMsgTxt("2nd input must be of type int64");
	}
	indx_map      = (int64_T *)mxGetData( prhs[1] );
    mSmall        = mxGetScalar( prhs[2] );
	s			  = mxGetPr(prhs[3]);
    
    At  = mxGetPr(prhs[0] );
    n   = mxGetM( prhs[0] );
    mBig= mxGetN( prhs[0] );
    /* Create a matrix for the return argument */
    plhs[0] = mxCreateDoubleMatrix( (mwSize)n, (mwSize)mSmall, mxREAL);
    Pt      = mxGetPr( plhs[0] );
    
    if (mxIsComplex(prhs[0])) {
        mexErrMsgTxt("Cannot handle complex data yet");
    }
	
    if (mxIsSparse(prhs[0])) {
        
        ir = mxGetIr(prhs[0]);      /* Row indexing      */
        jc = mxGetJc(prhs[0]);      /* Column count      */
        a  = mxGetPr(prhs[0]);      /* Non-zero elements */
        
        /* Loop through columns of At */
        
        for ( i=0; i < mBig; i++ ) {
            k   = indx_map[i]-1; /* 0-based */
            /* copy Pt(:,k) <-- At(:,i)
             * e.g. since height of Pt is N,
             * P + k*n <-- At + i*n  */
            
            for ( j=jc[i]; j<jc[i+1]; j++ ) {
                Pt[k*n+ ir[j] ] += s[i]*a[j];
			}
        }
        
    } else {
        
        
        /* And the actual computation:
         * Copy columns of At to Pt */
        for ( i=0; i < mBig; i++ ) {
            k   = indx_map[i]-1; /* 0-based */
            /* copy Pt(:,k) <-- At(:,i)
             * e.g. since height of Pt is N,
             * P + k*n <-- At + i*n  */
            
            for ( j=0; j<n; j++ ) {
                Pt[k*n+j] += s[i]*At[i*n+j];
			}
		}
        
    }

    return;
}
