/**
 * Compile with the following in matlab:
 * mex CXXFLAGS='$CXXFLAGS -std=c++1z -O3 -march=native -Wall -Wextra
 * -Wpedantic' acf_est.cpp
 */

#include <sstream>
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
#if INSTRSET >= 10  // AVX512VL
#define VEC_SINGLE_TYPE Vec16f
#define VEC_DOUBLE_TYPE Vec8d
#elif INSTRSET >= 8  // AVX2
#define VEC_SINGLE_TYPE Vec8f
#define VEC_DOUBLE_TYPE Vec4d
#elif INSTRSET == 2
#define VEC_SINGLE_TYPE Vec4f
#define VEC_DOUBLE_TYPE Vec2d
#else
#endif

/** Parameters to spawned threads */
struct ThreadParams {
  unsigned n_threads; /**< Total number of worker threads */
  unsigned index;     /**< Thread index (0 <= index < n_threads) */
  mwSize N; /**< Number of rows in matrix (Size of each ACF estimation) */
  mwSize C; /**< Number of columns in matrix (Number of ACFs to estimate) */
  void* x;  /**< Input matrix [N, C] */
  void* y;  /**< Output matrix [(2*N-1), C] */
};

void mexFunction(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void checkArguments(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
mxArray* spawnThreads(const mxArray* vIn);
template <typename T>
void* calculate(const ThreadParams& p);
unsigned detectNumberOfCores();
template <typename T>
inline void core(const T* x, mwSize k, T* s);
template <>
inline void core(const float* x, mwSize k, float* s);
template <>
inline void core(const double* x, mwSize k, double* s);
template <typename T>
inline int vecSize();
template <>
inline int vecSize<float>();
template <>
inline int vecSize<double>();

// TODO list:
// * Is it possible to add an exit signal from matlab?
// * Change name to bartlett_est
// * Better matlab command documentation

/**
 * Matlab mex entry function. Calculate Bartlett estimate of auto correlation
 * function efficiently.
 *
 * \param nlhs Number of left hand parameters
 * \param plhs Left hand parameters [nlhs]
 * \param nrhs Number of right hand parameters
 * \param prhs Right hand parameters [nrhs]
 *
 */
void mexFunction(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  checkArguments(nlhs, plhs, nrhs, prhs);
  mxArray* y = spawnThreads(prhs[0]);
  if (nrhs >= 1)
    plhs[0] = y;
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
void checkArguments(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs) {
  (void)(plhs);  // Unused

  if (nrhs != 1)
    mexErrMsgIdAndTxt("acf_est:checkArguments", "One input required");
  if (!mxIsSingle(prhs[0]) && !mxIsDouble(prhs[0]))
    mexErrMsgIdAndTxt("acf_est:checkArguments",
                      "Input matrix must be of type single or double");
  if (mxIsComplex(prhs[0]))
    mexErrMsgIdAndTxt("acf_est:checkArguments",
                      "Input matrix cannot be complex");
  if (mxGetNumberOfDimensions(prhs[0]) >= 4)
    mexErrMsgIdAndTxt("acf_est:checkArguments",
                      "Cannot handle 4-dimensional matrices or greater");
  if (nlhs > 1)
    mexErrMsgIdAndTxt("acf_est:checkArguments", "One or zero outputs required");
}

/**
 * Spawn worker threads
 *
 * \param vIn Input array
 * \return Output array
 *
 */
mxArray* spawnThreads(const mxArray* vIn) {
  // Get matrix dimensions
  const mwSize* dims = mxGetDimensions(vIn);
  mwSize N = dims[0];
  mwSize C = dims[1];

  // Create output matrix
  mxClassID classId = mxUNKNOWN_CLASS;
  if (mxIsSingle(vIn))
    classId = mxSINGLE_CLASS;
  else
    classId = mxDOUBLE_CLASS;
  mxArray* vOut = mxCreateNumericMatrix(N, C, classId, mxREAL);

  // Ensure that the first non-singular dimension is handled
  if (N == 1 && C != 1)
    std::swap(C, N);

  // Detect parallelism
  unsigned n_threads = detectNumberOfCores();

  // Allocate threads and their parameters
  std::vector<std::thread> threads(n_threads);
  std::vector<ThreadParams> params(n_threads);

  // Set parameters for each thread
  for (unsigned i = 0; i < n_threads; ++i) {
    params[i].n_threads = n_threads;
    params[i].index = i;
    params[i].N = N;
    params[i].C = C;
    params[i].x = mxGetData(vIn);
    params[i].y = mxGetData(vOut);
  }

  // Start all threads
  try {
    for (unsigned i = 0; i < n_threads; ++i) {
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
  } catch (const std::exception& ex) {
    std::stringstream ss;
    ss << "Failed to create thread: " << ex.what();
    mexErrMsgIdAndTxt("acf_est:spawnThreads", ss.str().c_str());
  }

// Wait for all threads to finish
#if SINGLE_THREAD_MODE == 0
  try {
    for (unsigned i = 0; i < n_threads; ++i)
      threads[i].join();
  } catch (const std::exception& ex) {
    std::stringstream ss;
    ss << "Failed to join thread: " << ex.what();
    mexErrMsgIdAndTxt("acf_est:spawnThreads", ss.str().c_str());
  }
#endif

  return vOut;
}

/**
 * Caluclate Bartlett's estimate
 *
 * \param p Thread parameters
 * \return Nothing
 *
 */
template <typename T>
void* calculate(const ThreadParams& p) {
  // Convert data pointers
  T* x = (T*)p.x;
  T* y = (T*)p.y;

  // Iterate through each column
  for (mwSize c = 0; c < p.C; ++c) {
    // Iterate through each input index.
    // First, make the first thread calculate r[0], the second r[1] ...
    // Then increase by the number of threads so the first thread now
    // calculates r[0 + n_threads]. For the next column, make the first
    // thread start at r[1] instead of at r[0] as earlier, to make it more
    // fair, since higher indices of r in general are easier.
    for (mwSize k = (c + p.index) % p.n_threads; k < p.N; k += p.n_threads) {
      T s = 0.0; /**< Current sum */
#ifndef VEC_SINGLE_TYPE
      // Simplest realization
      for (size_t n = 0; n < N - k; ++n)
        s += x[n] * x[n + k];
#else
      int lim = (int)p.N - (int)k - vecSize<T>() + 1; /**< Iteration limit */
      int n;                                          /**< Iteration index */
      // Vectorized loop
      for (n = 0; n < lim; n += vecSize<T>())
        core<T>(x + c * p.N + n, k, &s);

      // Non-vectorized loop for the remaining vector size - 1 elements
      for (; n < (int)(p.N - k); ++n)
        s += x[c * p.N + n] * x[c * p.N + n + k];
#endif

      // Sum is ready. Write only half of spectra due to symmetry and for
      // efficency
      y[c * p.N + k] = s / p.N;
    }
  }

  return nullptr;
}

/**
 * Detect number of CPU cores
 *
 * \return Number of CPU cores
 *
 */
unsigned detectNumberOfCores() {
  // TODO: This is a poor man's core detector :(

  /** Number of threads to spawn. Assume SMT for now. */
  unsigned n = std::thread::hardware_concurrency();

  // Sanity check
  if (n == 0)
    n = 1;

  // Assume SMT (hyperthreading) if the number of threads are even
  if (n % 2 == 0)
    n /= 2;

  return n;
}

/**
 *
 * Generic (empty) core routine function
 *
 * \tparam T Input/Output type
 * \param x Input array
 * \param k Offset
 * \param s Sum [out]
 *
 */
template <typename T>
inline void core(const T* x, mwSize k, T* s) {}

/**
 *
 * Core calculation routine for float
 *
 * \param x Input array
 * \param k Offset
 * \param s Sum [out]
 *
 */
template <>
inline void core(const float* x, mwSize k, float* s) {
  // Read two float vectors offset by k from memory
  VEC_SINGLE_TYPE v1 = VEC_SINGLE_TYPE().load(x);
  VEC_SINGLE_TYPE v2 = VEC_SINGLE_TYPE().load(x + k);

  // Multiply
  VEC_SINGLE_TYPE prod = v1 * v2;

  // Sum
  *s += horizontal_add(prod);
}

/**
 *
 * Core calculation routine for double
 *
 * \param x Input array
 * \param k Offset
 * \param s Sum [out]
 *
 */
template <>
inline void core(const double* x, mwSize k, double* s) {
  // Read two double vectors offset by k from memory
  VEC_DOUBLE_TYPE v1 = VEC_DOUBLE_TYPE().load(x);
  VEC_DOUBLE_TYPE v2 = VEC_DOUBLE_TYPE().load(x + k);

  // Multiply
  VEC_DOUBLE_TYPE prod = v1 * v2;

  // Sum
  *s += horizontal_add(prod);
}

/**
 *
 * Generic template for vector size
 *
 * \return Vector size
 *
 */
template <typename T>
inline int vecSize() {
  return 1;
}

/**
 *
 * Get vector size
 *
 * \return Vector size for float
 *
 */
template <>
inline int vecSize<float>() {
  return VEC_SINGLE_TYPE::size();
}

/**
 *
 * Get vector size
 *
 * \return Vector size for double
 *
 */
template <>
inline int vecSize<double>() {
  return VEC_DOUBLE_TYPE::size();
}