/**
 * Compile with the following in matlab:
 * mex CXXFLAGS='$CXXFLAGS -std=c++1z -O3 -march=native -Wall -Wextra
 * -Wpedantic' acf_est.cpp
 */

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

// Matlab mex headers
#include "matrix.h"
#include "mex.h"

// Agner Fog's vectorclass library
#include "vectorclass/vectorclass.h"

// TODO list:
// * Add debug mode that printouts and sets SINGLE_THREAD_MODE to 1?
// * Update vectorclass version?
// * Use vectorclass instructionset detection
// * Makefile?

/** Set to 1 to spawn 0 threads and calculate everything in main thread. May be
 * useful when debugging. */
#define SINGLE_THREAD_MODE 0

/** Set to enable vectorized instructions */
#define VECTORIZATION 1

// Handle restrict keyword
#if defined __GNUC__ || defined __clang__
#define RESTRICT __restrict__
#elif defined(MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT
#endif

// Detect instruction set and set vector types for single and double accordingly
#if INSTRSET >= 10  // AVX512VL
typedef Vec16f vec_single_t;
typedef Vec8d vec_double_t;
#elif INSTRSET >= 8  // AVX2
typedef Vec8f vec_single_t;
typedef Vec4d vec_double_t;
#elif INSTRSET == 2
typedef Vec4f vec_single_t;
typedef Vec2d vec_double_t;
#else
#define VECTORIZATION 0
typedef float vec_single_t;
typedef double vec_double_t;
#endif

/** Divide work into work items */
struct WorkItem {
  /** Row start index (inclusive, 0 <= nStart < N) */
  mwSize nStart;
  /** Row end index (exclusive, 0 <= nEnd < N) */
  mwSize nEnd;
  /** Column start index (inclusive, 0 <= cStart < C) */
  mwSize cStart;
  /** Column end index (exclusive, 0 <= cEnd < C) */
  mwSize cEnd;
};

/** Parameters to worker threads */
struct ThreadParams {
  /** Number of matrix rows (Size of each ACF estimation) */
  mwSize N;
  /** Number of matrix columns (Number of ACFs to estimate) */
  mwSize C;
  /** Input matrix [N, C] */
  const void* x;
  /** Output matrix [2N-1, C] */
  void* y;
  /** Current (global) index in work queue */
  std::atomic<size_t>* workQueueIdx;
  /** Work queue */
  std::vector<WorkItem>* workQueue;
};

void mexFunction(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
void checkArguments(int nlhs, mxArray** plhs, int nrhs, const mxArray** prhs);
mxArray* spawnThreads(const mxArray* vIn);
template <typename Tvec, typename Tscal>
void* calculate(const ThreadParams& p);
std::vector<WorkItem> divideWork(mwSize N, mwSize C, mwSize workItemsPerCol);
const WorkItem* nextWorkItem(const ThreadParams& p);

/**
 * Matlab mex entry function. Efficient calculation of Bartlett's estimate of
 * the auto correlation function.
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
    std::swap(N, C);

  // Detect parallelism. Use all threads.
  unsigned nThreads = std::thread::hardware_concurrency();
  if (nThreads == 0)
    nThreads = 1;

  // Fill work queue
  auto workItems = divideWork(N, C, nThreads);

  // Set parameters passed to threads
  ThreadParams params;
  std::atomic<size_t> workQueueIdx = 0; /**< Current work queue index */
  params.N = N;
  params.C = C;
  params.x = mxGetData(vIn);
  params.y = mxGetData(vOut);
  params.workQueue = &workItems;
  params.workQueueIdx = &workQueueIdx;

  // Allocate worker threads
  std::vector<std::thread> threads(nThreads);

  // Start all worker threads
  try {
    for (unsigned i = 0; i < nThreads; ++i) {
#if SINGLE_THREAD_MODE == 0
      if (mxIsSingle(vIn)) {
        threads[i] =
            std::move(std::thread(calculate<vec_single_t, float>, params));
      } else {
        threads[i] =
            std::move(std::thread(calculate<vec_double_t, double>, params));
      }
#else
      if (mxIsSingle(vIn))
        calculate<vec_single_t, float>(params);
      else
        calculate<vec_double_t, double>(params);
#endif
    }
  } catch (const std::exception& ex) {
    std::stringstream ss;
    ss << "Failed to create thread: " << ex.what();
    mexErrMsgIdAndTxt("acf_est:spawnThreads", ss.str().c_str());
  }

// Wait for all worker threads to finish
#if SINGLE_THREAD_MODE == 0
  try {
    for (unsigned i = 0; i < nThreads; ++i)
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
 * \tparam Tvec  Vector type to use when calculating
 * \tparam Tscal Scalar type to use when calculating
 * \param p      Worker thread parameters
 * \return NULL
 *
 */
template <typename Tvec, typename Tscal>
void* calculate(const ThreadParams& p) {
  // Cast data pointers
  const Tscal* RESTRICT x = static_cast<const Tscal*>(p.x); /**< Input */
  Tscal* RESTRICT y = static_cast<Tscal*>(p.y);             /**< Output */

  // Get next work item
  const WorkItem* w = nullptr;
  while ((w = nextWorkItem(p)) != nullptr) {
    // Iterate through columns
    for (mwSize c = w->cStart; c < w->cEnd; ++c) {
      // Iterate through rows
      for (mwSize k = w->nStart; k < w->nEnd; ++k) {
        Tscal s; /**< Current sum */
#if VECTORIZATION == 0
        // Zero s since summed to
        s = 0.0;

        // Simplest realization
        for (size_t n = 0; n < p.N - k; ++n)
          s += x[n] * x[n + k];
#else
        int lim = (int)p.N - (int)k - Tvec::size() + 1; /**< Iteration limit */
        int n;                                          /**< Iteration index */

        // Use a sum vector and do a horizontal add when finished
        Tvec sv(0);
        // Vectorized loop
        for (n = 0; n < lim; n += Tvec::size()) {
          // Read two double vectors offset by k from memory
          Tvec v1 = Tvec().load(x + c * p.N + n);
          Tvec v2 = Tvec().load(x + c * p.N + n + k);

          // Multiply and sum
          sv = mul_add(v1, v2, sv);
        }

        // Finished with vector operations. Sum vector elements.
        s = horizontal_add(sv);

        // Non-vectorized loop for the remaining vector size - 1 elements
        for (; n < (int)(p.N - k); ++n)
          s += x[c * p.N + n] * x[c * p.N + n + k];
#endif

        // Sum is ready. Write only half of spectra due to symmetry and for
        // efficency
        y[c * p.N + k] = s / p.N;
      }
    }
  }

  // Always return NULL
  return nullptr;
}

/**
 * Divide work into work items into suitable size
 *
 * \param N        Number of rows
 * \param C        Number of columns
 * \param nThreads Number of worker threads
 *
 * \return List with work items
 */
std::vector<WorkItem> divideWork(mwSize N, mwSize C, mwSize nThreads) {
  // Calculate number of work items per column as the number of
  // multiplications that takes 1 ms on a CPU with nThreads cores. Assume a
  // clock speed of 2GHz and a cost of floating point multiplication as 1 clock
  // cycle. There are N(N+1)/2 multiplications for one column (arithmetic sum).
  mwSize itemsPerCol = N * (N + 1) / (4 * nThreads * 1000000);
  if (itemsPerCol == 0)
    itemsPerCol = 1;

  // Preallocate "heavy" floating point calculations
  std::vector<std::pair<mwSize, mwSize>> lim(itemsPerCol);  // Limits
  for (unsigned n = 0; n < itemsPerCol; ++n) {
    // Divide into itemsPerCol work items for each column
    // such that all work items result in approximately the same number of
    // multiplications.
    mwSize a = (n == 0 ? 0 : lim[n - 1].second);  // a is the previous b
    double sqrtArg = std::max(
        0.0, a * a - 2 * N * a + N * N -
                 static_cast<double>(N * N) /
                     itemsPerCol);  // Ensure nonnegative, for valid sqare root
    mwSize b = (n == itemsPerCol - 1
                    ? N
                    : N - static_cast<mwSize>(std::ceil(std::sqrt(sqrtArg))));
    lim[n] = std::make_pair(a, b);
  }

  // Make work items
  std::vector<WorkItem> workItems(itemsPerCol * C);
  for (unsigned c = 0; c < C; ++c) {
    for (unsigned n = 0; n < itemsPerCol; ++n) {
      WorkItem& w = workItems[c * itemsPerCol + n];
      w.cStart = c;
      w.cEnd = w.cStart + 1;
      w.nStart = lim[n].first;
      w.nEnd = lim[n].second;
    }
  }

  return workItems;
}

/**
 * Get a new work item, or nothing
 *
 * \param p Thread parameters
 *
 * \return Pointer to work item, or NULL
 *
 */
const WorkItem* nextWorkItem(const ThreadParams& p) {
  // Expected work queue index.
  auto queueIdxExp = p.workQueueIdx->load(std::memory_order_relaxed);

  // Use CAS for speed
  while (!p.workQueueIdx->compare_exchange_weak(queueIdxExp, queueIdxExp + 1,
                                                std::memory_order_release,
                                                std::memory_order_relaxed)) {
    // Do nothing
  }

  // Check if finished
  if (queueIdxExp >= p.workQueue->size()) {
    return nullptr;
  }

  // Get unique and valid work item
  return &(*p.workQueue)[queueIdxExp];
}
