/** 
 * Compile with the following in matlab:
 * mex CXXFLAGS='$CXXFLAGS -std=c++1z -O3 -march=native -Wall -Wextra -Wpedantic -Wshadow' acf_est.cpp
 */

#include <thread>

// Matlab mex headers
#include "matrix.h"
#include "mex.h"

// Agner Fog's vectorclass library headers
#include "vectorclass/vectorclass.h"

#include <iostream>

/** Number of threads to spawn */
const unsigned int THREADS = 4;

// TODO list:
// * Add support for floats, ints ...?
// * Remove vectorclass in favour of VcDevel/Vc?
// * Detect maximum vectorization level (AVX, AVX2...). See above.
// * Autodetect number of cores
// * Add another estimation function?
// * Handle exceptions gracefully when creating threads etc
// * Mex ID:s are wrong in mexErrMsgIdAndText

/** Parameters to spawned threads */
struct ThreadParams
{
    unsigned int index; /**< Thread index (0 <= index < THREADS) */
    mwSize N;           /**< Number of rows in matrix (Size of each ACF estimation) */
    mwSize C;           /**< Number of columns in matrix (Number of ACFs to estimate) */
    mxDouble* x;        /**< Input matrix [N, C] */
    mxDouble* y;        /**< Output matrix [(2*N-1), C] */
};

/**
 * Caluclate Batlett's estimate
 * 
 * \param p Thread parameters
 * \return Nothing
 * 
 */
void* calculate(const ThreadParams& p)
{
    // Iterate through each column
    for (mwSize c = 0; c < p.C; ++c)
    {
        // Iterate through each input index
        for (mwSize k = (c + p.index) % THREADS; k < p.N; k += THREADS)
        {
            double s = 0.0; /**< Current sum */
#if 1 // Enable for correct but slow implementation
            int lim = (int)p.N - (int)k - 3; /**< Iteration limit */
            int n;
            for (n = 0; n < lim; n += 4)
            {
                // Vectorized multiplication and summation
                Vec4d v1 = Vec4d().load(p.x + c*p.N + n);
                Vec4d v2 = Vec4d().load(p.x + c*p.N + n + k);
                Vec4d prod = v1*v2;
                s += horizontal_add(prod);
            }

            // Don't forget the last elements that didn't fit into a vector
            for (; n < p.N - k; ++n)
                s += p.x[c*p.N + n]*p.x[c*p.N + n + k];
#else
            for (size_t n = 0; n < N - k; ++n)
                s += x[n]*x[n + k];
#endif
            
            // Divide by N and write twice, due to ACF symmetry
            double d = s/p.N;
            p.y[c*p.N + k] = d;
        }
    }
    
    return nullptr;
}

/**
 * Spawn worker threads
 * 
 * \param vIn Input array
 * \return Output array
 */
mxArray* spawnThreads(const mxArray* vIn)
{
    // Get matrix dimensions
    const mwSize* dims = mxGetDimensions(vIn);
    mwSize N = dims[0];
    mwSize C = dims[1];

    // Create output
    mxArray* vOut = mxCreateDoubleMatrix(N, C, mxREAL);

    // Ensure that the first non-singular dimension is handled 
    if (N == 1 && C != 1)
        std::swap(C, N);
    
    // Allocate threads and their parameters
    std::thread threads[THREADS];
    ThreadParams params[THREADS];
    
    // Set parameters for each thread
    for (unsigned int i = 0; i < THREADS; ++i)
    {
        params[i].index = i;
        params[i].N = N;
        params[i].C = C;
        params[i].x = (mxDouble*) mxGetData(vIn);
        params[i].y = (mxDouble*) mxGetData(vOut);
    }
    
    // Start all threads
    for (unsigned int i = 0; i < THREADS; ++i)
        threads[i] = std::thread(calculate, params[i]);

    // Wait for all threads to finish
    for (unsigned int i = 0; i < THREADS; ++i)
        threads[i].join();
    
    return vOut;
}

/**
 * Check that arguments in are valid
 * 
 * \param nlhs Number of left hand parameters
 * \param plhs Left hand parameters [nlhs]
 * \param nrhs Number of right hand parameters
 * \param prhs Right hand parameters [nrhs]
 * 
 */
void checkArguments(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
    (void)(plhs);
    
    if (nrhs != 1)
        mexErrMsgIdAndTxt("acf_est", "One input required");

    if (!mxIsDouble(prhs[0]))
        mexErrMsgIdAndTxt("acf_est", "Input matrix must be of type double");
    
    if (mxGetNumberOfDimensions(prhs[0]) >= 4)
        mexErrMsgIdAndTxt("acf_est", "cannot handle 4-dimensional matrices or greater");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("acf_est", "One or zero outputs required");
}

/**
 * Matlab mex entry function. Calculate Bartlett estimate of auto correlation function efficiently.
 * 
 * \param nlhs Number of left hand parameters
 * \param plhs Left hand parameters [nlhs]
 * \param nrhs Number of right hand parameters
 * \param prhs Right hand parameters [nrhs]
 * 
 */
void mexFunction(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs)
{
    checkArguments(nlhs, plhs, nrhs, prhs);
    mxArray* y = spawnThreads(prhs[0]);
    if (nrhs >= 1)
        plhs[0] = y;
}
