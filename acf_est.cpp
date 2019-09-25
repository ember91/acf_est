/** 
 * Compile with the following in matlab:
 * mex CXXFLAGS='$CXXFLAGS -std=c++1z -O3 -march=native -Wall -Wextra -Wpedantic -Wshadow' acf_est.cpp
 */

#include <thread>
#include <vector>

// Matlab mex headers
#include "matrix.h"
#include "mex.h"

// Agner Fog's vectorclass library headers
#include "vectorclass/vectorclass.h"

// TODO list:
// * Add support for floats, ints ...?
// * Remove vectorclass in favour of VcDevel/Vc?
// * Detect maximum vectorization level (AVX, AVX2...). See above.
// * Handle exceptions gracefully when creating threads etc
// * Add README
// * Change formatter defaults

/** Parameters to spawned threads */
struct ThreadParams
{
    unsigned n_threads; /**< Total number of worker threads */
    unsigned index;     /**< Thread index (0 <= index < n_threads) */
    mwSize N;           /**< Number of rows in matrix (Size of each ACF estimation) */
    mwSize C;           /**< Number of columns in matrix (Number of ACFs to estimate) */
    mxDouble *x;        /**< Input matrix [N, C] */
    mxDouble *y;        /**< Output matrix [(2*N-1), C] */
};

/**
 * Caluclate Batlett's estimate
 * 
 * \param p Thread parameters
 * \return Nothing
 * 
 */
void *calculate(const ThreadParams &p)
{
    // Iterate through each column
    for (mwSize c = 0; c < p.C; ++c)
    {
        // Iterate through each input index
        for (mwSize k = (c + p.index) % p.n_threads; k < p.N; k += p.n_threads)
        {
            double s = 0.0;                  /**< Current sum */
#if 1                                        // Enable for correct but slow implementation
            int lim = (int)p.N - (int)k - 3; /**< Iteration limit */
            int n;
            for (n = 0; n < lim; n += 4)
            {
                // Vectorized multiplication and summation
                Vec4d v1 = Vec4d().load(p.x + c * p.N + n);
                Vec4d v2 = Vec4d().load(p.x + c * p.N + n + k);
                Vec4d prod = v1 * v2;
                s += horizontal_add(prod);
            }

            // Don't forget the last elements that didn't fit into a vector
            for (; n < p.N - k; ++n)
                s += p.x[c * p.N + n] * p.x[c * p.N + n + k];
#else
            for (size_t n = 0; n < N - k; ++n)
                s += x[n] * x[n + k];
#endif

            // Divide by N and write twice, due to ACF symmetry
            double d = s / p.N;
            p.y[c * p.N + k] = d;
        }
    }

    return nullptr;
}

/**
 * Detect number of CPU cores
 * 
 * \return Number of CPU cores
 */
unsigned detectNumberOfCores()
{
    /** Number of threads to spawn. Assume SMT for now. */
    unsigned n = std::thread::hardware_concurrency();

    // Sanity check
    if (n == 0)
        n = 1;

    // Assume SMT if even number of threads
    if (n % 2 == 0)
        n /= 2;

    return n;
}

/**
 * Spawn worker threads
 * 
 * \param vIn Input array
 * \return Output array
 */
mxArray *spawnThreads(const mxArray *vIn)
{
    // Get matrix dimensions
    const mwSize *dims = mxGetDimensions(vIn);
    mwSize N = dims[0];
    mwSize C = dims[1];

    // Create output matrix
    mxArray *vOut = mxCreateDoubleMatrix(N, C, mxREAL);

    // Ensure that the first non-singular dimension is handled
    if (N == 1 && C != 1)
        std::swap(C, N);

    unsigned n_threads = detectNumberOfCores();

    // Allocate threads and their parameters
    std::vector<std::thread> threads(n_threads);
    std::vector<ThreadParams> params(n_threads);

    // Set parameters for each thread
    for (unsigned i = 0; i < n_threads; ++i)
    {
        params[i].n_threads = n_threads;
        params[i].index = i;
        params[i].N = N;
        params[i].C = C;
        params[i].x = (mxDouble *)mxGetData(vIn);
        params[i].y = (mxDouble *)mxGetData(vOut);
    }

    // Start all threads
    for (unsigned i = 0; i < n_threads; ++i)
        threads[i] = std::move(std::thread(calculate, params[i]));

    // Wait for all threads to finish
    for (unsigned i = 0; i < n_threads; ++i)
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
void checkArguments(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    (void)(plhs);

    if (nrhs != 1)
        mexErrMsgIdAndTxt("acf_est:checkArguments", "One input required");

    if (!mxIsDouble(prhs[0]))
        mexErrMsgIdAndTxt("acf_est:checkArguments", "Input matrix must be of type double");

    if (mxIsComplex(prhs[0]))
        mexErrMsgIdAndTxt("acf_est:checkArguments", "Input matrix cannot be complex");

    if (mxGetNumberOfDimensions(prhs[0]) >= 4)
        mexErrMsgIdAndTxt("acf_est:checkArguments", "Cannot handle 4-dimensional matrices or greater");

    if (nlhs > 1)
        mexErrMsgIdAndTxt("acf_est:checkArguments", "One or zero outputs required");
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
void mexFunction(int nlhs, mxArray **plhs, int nrhs, const mxArray **prhs)
{
    checkArguments(nlhs, plhs, nrhs, prhs);
    mxArray *y = spawnThreads(prhs[0]);
    if (nrhs >= 1)
        plhs[0] = y;
}
