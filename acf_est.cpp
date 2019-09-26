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

/** Set to 1 to spawn 0 threads. May be useful when debugging. */
#define SINGLE_THREAD_MODE 0

// Detect instruction set and set vector accordingly
#if INSTRSET >= 10 // AVX512VL
#define VEC_SINGLE_TYPE Vec16f
#define VEC_DOUBLE_TYPE Vec8d
#elif INSTRSET >= 8 // AVX2
#define VEC_SINGLE_TYPE Vec8f
#define VEC_DOUBLE_TYPE Vec4d
#elif INSTRSET == 2
#define VEC_SINGLE_TYPE Vec4f
#define VEC_DOUBLE_TYPE Vec2d
#else
#endif

// TODO list:
// * Handle exceptions gracefully when creating threads etc
// * Is it possible to add an exit signal from matlab?
// * Add README
// * Change formatter defaults
// * License
// * Instructions about vectorclass
// * Documentation

/** Parameters to spawned threads */
struct ThreadParams
{
    unsigned n_threads; /**< Total number of worker threads */
    unsigned index;     /**< Thread index (0 <= index < n_threads) */
    mwSize N;           /**< Number of rows in matrix (Size of each ACF estimation) */
    mwSize C;           /**< Number of columns in matrix (Number of ACFs to estimate) */
    void *x;            /**< Input matrix [N, C] */
    void *y;            /**< Output matrix [(2*N-1), C] */
};

template <typename T>
inline void core(T *x, mwSize k, T *s) {}

template <>
inline void core(float *x, mwSize k, float *s)
{
    VEC_SINGLE_TYPE v1 = VEC_SINGLE_TYPE().load(x);
    VEC_SINGLE_TYPE v2 = VEC_SINGLE_TYPE().load(x + k);
    VEC_SINGLE_TYPE prod = v1 * v2;
    *s = *s + horizontal_add(prod);
}

template <>
inline void core(double *x, mwSize k, double *s)
{
    VEC_DOUBLE_TYPE v1 = VEC_DOUBLE_TYPE().load(x);
    VEC_DOUBLE_TYPE v2 = VEC_DOUBLE_TYPE().load(x + k);
    VEC_DOUBLE_TYPE prod = v1 * v2;
    *s = *s + horizontal_add(prod);
}

template <typename T>
inline int vecSize()
{
    return 1;
}

template <>
inline int vecSize<float>()
{
    return VEC_SINGLE_TYPE::size();
}

template <>
inline int vecSize<double>()
{
    return VEC_DOUBLE_TYPE::size();
}

/**
 * Caluclate Batlett's estimate
 * 
 * \param p Thread parameters
 * \return Nothing
 * 
 */
template <typename T>
void *
calculate(const ThreadParams &p)
{
    T *x = (T *)p.x;
    T *y = (T *)p.y;

    // Iterate through each column
    for (mwSize c = 0; c < p.C; ++c)
    {
        // Iterate through each input index
        for (mwSize k = (c + p.index) % p.n_threads; k < p.N; k += p.n_threads)
        {
            T s = 0.0; /**< Current sum */
#ifndef VEC_SINGLE_TYPE
            // Simplest realization
            for (size_t n = 0; n < N - k; ++n)
                s += x[n] * x[n + k];
#else
            int lim = (int)p.N - (int)k - vecSize<T>() + 1; /**< Iteration limit */
            int n;
            for (n = 0; n < lim; n += vecSize<T>())
                core<T>(x + c * p.N + n, k, &s);

            // Don't forget the last elements that didn't fit into a vector
            for (; n < p.N - k; ++n)
                s += x[c * p.N + n] * x[c * p.N + n + k];
#endif

            // Divide by N and write twice, due to ACF symmetry
            y[c * p.N + k] = s / p.N;
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
 * 
 */
mxArray *spawnThreads(const mxArray *vIn)
{
    // Get matrix dimensions
    const mwSize *dims = mxGetDimensions(vIn);
    mwSize N = dims[0];
    mwSize C = dims[1];

    // Create output matrix
    mxClassID classId = mxUNKNOWN_CLASS;
    if (mxIsSingle(vIn))
        classId = mxSINGLE_CLASS;
    else
        classId = mxDOUBLE_CLASS;
    mxArray *vOut = mxCreateNumericMatrix(N, C, classId, mxREAL);

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
        params[i].x = mxGetData(vIn);
        params[i].y = mxGetData(vOut);
    }

    // Start all threads
    for (unsigned i = 0; i < n_threads; ++i)
    {
#if SINGLE_THREAD_MODE == 0
        if (mxIsSingle(vIn))
            threads[i] = std::move(std::thread(calculate<float>, params[i]));
        else
            threads[i] = std::move(std::thread(calculate<double>, params[i]));
#else
        if (mxIsSingle(vIn))
            calculate<float>(params[i]);
        else
            calculate<double>(params[i]);
#endif
    }

// Wait for all threads to finish
#if SINGLE_THREAD_MODE == 0
    for (unsigned i = 0; i < n_threads; ++i)
        threads[i].join();
#endif

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

    if (!mxIsSingle(prhs[0]) && !mxIsDouble(prhs[0]))
        mexErrMsgIdAndTxt("acf_est:checkArguments", "Input matrix must be of type single or double");

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
