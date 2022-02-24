/*
 *Calling sequence:
 * Pt = countSketch( At, indx_map, mSmall, s )
 * 
 * A is a mBig x N matrix and At is its transpose
 * P is a mSmall x N matrix, and Pt is its transpose
 * indx_map     is a int64 vector of length mBig where each entry
 *      is an integer in [1,mSmall] (i.e., 1-based indexing, Matlab-style)
 * s is a row or column vector containing the sign function. This should be
 * 		a double array.
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
 * Compile with just:  mex countSketch.c
 *
 * For efficiency, since Matlab uses column-major order,
 * the input should be At ( = A') and NOT A
 * Likewise, the output is Pt ( = P' = (Sketch(A))' )
 *
 * In theory, this can be applied to sparse matrices
 * It would only be efficient if they are stored in csr order
 * (Matlab uses csc), or if we have the transpose of a csc matrix
 * (i.e., do the exact same transpose trick we do for sparse
 *  matrices )
 *
 * For now, does NOT do sparse matrices
 *
 * Warning: the code does not do error checks, so it can easily crash.
 * Make sure that the "indx_map" is of type int64
 *
 * Stephen Becker, srbecker@us.ibm.com, June 5 2014
 * The use of CountSketch was suggested by Haim Avron
 **/
#if defined(__GNUC__) && !(defined(__clang__)) 
#include <uchar.h>
#endif
#include <math.h>
#include "mex.h"

#ifdef USE_BLAS
#include "blas.h"
#endif

/* NOTE 1: This code uses int64_T for indx_map to ensure that the size 
 * is 8 bytes on all systems.
 * 
 * NOTE 2: I have changed this code so that a sign function can be passed in
 * instead of a plusMinusSwitchIndex.
**/

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *At,*Pt; 
    double *A, *P;
    double alpha;	
	int64_T *indx_map;
	double *s;

    mwSize mBig,mSmall,n, i,j,k;
    int     DO_TRANSPOSE=1;
#ifdef USE_BLAS
    ptrdiff_t   size, stride, strideBig;
#endif
    
    /* Check for proper number of arguments */
    if (nrhs != 5) { 
	    mexErrMsgIdAndTxt( "MATLAB:countSketch:invalidNumInputs",
                "Five input arguments required."); 
    } else if (nlhs > 1) {
	    mexErrMsgIdAndTxt( "MATLAB:countSketch:maxlhs",
                "Too many output arguments."); 
    } 
    if ( !(mxIsInt64(prhs[1])) )
        mexErrMsgTxt("2nd input must be of type int64");
	indx_map      = (int64_T *)mxGetData( prhs[1] );
    mSmall        = mxGetScalar( prhs[2] );
    s 			  = mxGetPr(prhs[3]);
    DO_TRANSPOSE  = (int)mxGetScalar( prhs[4] );
    
    if (mxIsSparse(prhs[0]))
        mexErrMsgTxt("Cannot handle sparse 'A' matrix, try countSketch_sparse.c instead");
    
    if ( DO_TRANSPOSE == 1 ) {
        At  = mxGetPr(prhs[0] );
        n   = mxGetM( prhs[0] );
        mBig= mxGetN( prhs[0] );
        /* Create a matrix for the return argument */
        plhs[0] = mxCreateDoubleMatrix( (mwSize)n, (mwSize)mSmall, mxREAL);
        Pt      = mxGetPr( plhs[0] );
        P       = NULL; /* try to prevent bugs */
        A       = NULL;
#ifdef USE_BLAS
        size    = (ptrdiff_t)n;
        stride  = (ptrdiff_t)1;
#endif
        /* And the actual computation:
         * Copy columns of At to Pt */
		for(i = 0; i < mBig; ++i) {
			alpha = s[i];
			k 	= indx_map[i] - 1;
#ifdef USE_BLAS
			daxpy(&size,&alpha,At+i*n,&stride,Pt+k*n,&stride);
#else
			for ( j=0; j<n; j++ )
                Pt[k*n+j] += alpha*At[i*n+j];
#endif 
		}
		
    } else if ( DO_TRANSPOSE == 0 ) {
        A   = mxGetPr(prhs[0] );
        n   = mxGetN( prhs[0] );
        mBig= mxGetM( prhs[0] );
        /* Create a matrix for the return argument */
        plhs[0] = mxCreateDoubleMatrix( (mwSize)mSmall, (mwSize)n, mxREAL);
        P       = mxGetPr( plhs[0] );
        Pt      = NULL;
        At      = NULL;
#ifdef USE_BLAS
        size    = (ptrdiff_t)n;
        stride  = (ptrdiff_t)mSmall;
        strideBig  = (ptrdiff_t)mBig;
#endif
        
		for(i = 0; i < mBig; ++i) {
			alpha = s[i];
			k = indx_map[i] - 1;
#ifdef USE_BLAS
			daxpy(&size,&alpha,A+i,&strideBig,P+k,&stride);
#else
			for ( j=0; j<n; j++ )
                P[k+j*mSmall] += alpha*A[i+j*mBig];
#endif
		}
	        
    } else {
        mexErrMsgTxt("4th input must 0 (no transpose) or 1 (transpose)");
    }

    return;
}
