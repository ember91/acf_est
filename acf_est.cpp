/**
 * Compile with the following in matlab:
 * mex CXXFLAGS='$CXXFLAGS -std=c++1z -O3 -march=native -Wall -Wextra
 * -Wpedantic' acf_est.cpp
 */

#include <algorithm>
#include <sstream>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <sysinfoapi.h>
#endif
#ifdef __linux__
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>
#endif
#ifdef __APPLE__
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

// Matlab mex headers
#include "matrix.h"
#include "mex.h"

// Agner Fog's vectorclass library headers
#include "vectorclass/vectorclass.h"

/** Set to 1 to spawn 0 threads. May be useful when debugging. */
#define SINGLE_THREAD_MODE 0

#define VECTORIZATION 1

// Detect instruction set and set vector accordingly
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
template <typename Tvec, typename Tscal>
void* calculate(const ThreadParams& p);
unsigned detectNumberOfCores();

// TODO list:
// * Better detection of instruction sets such as FMA...
// * Calculate vectorized average?

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

  // printf("Detect %u CPU hardware cores\n", n_threads);

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
      if (mxIsSingle(vIn)) {
        threads[i] =
            std::move(std::thread(calculate<vec_single_t, float>, params[i]));
      } else {
        threads[i] =
            std::move(std::thread(calculate<vec_double_t, double>, params[i]));
      }
#else
      if (mxIsSingle(vIn))
        calculate<vec_single_t, float>(params[i]);
      else
        calculate<vec_double_t, double>(params[i]);
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
 * \tparam Tvec  Vector type to use when calculating
 * \tparam Tscal Scalar type to use when calculating
 * \param p      Thread parameters
 * \return Nothing
 *
 */
template <typename Tvec, typename Tscal>
void* calculate(const ThreadParams& p) {
  // Convert data pointers
  Tscal* x = (Tscal*)p.x;
  Tscal* y = (Tscal*)p.y;

  // Iterate through each column
  for (mwSize c = 0; c < p.C; ++c) {
    // Iterate through each input index.
    // First, make the first thread calculate r[0], the second r[1] ...
    // Then increase by the number of threads so the first thread now
    // calculates r[0 + n_threads]. For the next column, make the first
    // thread start at r[1] instead of at r[0] as earlier, to make it more
    // fair, since higher indices of r in general are easier.
    for (mwSize k = (c + p.index) % p.n_threads; k < p.N; k += p.n_threads) {
      Tscal s; /**< Current sum */
#if VECTORIZATION == 0
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

      // We're finished. Time to sum.
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

  return nullptr;
}

/**
 * Detect number of CPU cores
 *
 * Fallback in case the smarter method fails
 *
 * \return Number of CPU cores
 */
unsigned detectNumberOfCoresFallback() {
  unsigned n = std::thread::hardware_concurrency();

  if (n == 0)
    return 1;

  return n;
}

/**
 * Trim from start (in place)
 *
 * \param s String to trim
 *
 */
inline void ltrim(std::string& s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(),
                                  [](int ch) { return !std::isspace(ch); }));
}

/**
 * Trim from start (in place)
 *
 * \param s String to trim
 *
 */
inline void rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](int ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

/**
 * Trim from start and end
 *
 *
 * \param s String to trim
 *
 */
inline void trim(std::string& s) {
  ltrim(s);
  rtrim(s);
}

/**
 * Detect number of physical CPU cores
 *
 * \note From Boost library, to lessen the dependency on a heavy library
 *
 * \return Number of CPU cores
 *
 */
unsigned detectNumberOfCores() {
#ifdef _WIN32
  unsigned cores = 0;
  DWORD size = 0;

  GetLogicalProcessorInformation(NULL, &size);
  if (ERROR_INSUFFICIENT_BUFFER != GetLastError())
    return 0;
  const size_t Elements = size / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);

  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(Elements);
  if (GetLogicalProcessorInformation(&buffer.front(), &size) == FALSE)
    return 0;

  for (size_t i = 0; i < Elements; ++i) {
    if (buffer[i].Relationship == RelationProcessorCore)
      ++cores;
  }
  return cores;
#elif defined(__linux__)
  try {
    std::ifstream proc_cpuinfo("/proc/cpuinfo");

    const std::string physical_id("physical id"), core_id("core id");

    typedef std::pair<unsigned, unsigned> core_entry;  // [physical ID, core id]

    std::set<core_entry> cores;

    core_entry current_core_entry;

    const std::string trim_chars = "\t\n\v\f\r ";

    std::string line;
    while (std::getline(proc_cpuinfo, line)) {
      if (line.empty())
        continue;

      size_t idx = line.find(':');

      if (idx == std::string::npos)
        return detectNumberOfCoresFallback();

      if (line.find(':', idx + 1) != std::string::npos)
        return detectNumberOfCoresFallback();

      std::string key = line.substr(0, idx);
      std::string value = line.substr(idx + 1);

      trim(key);
      trim(value);

      if (key == physical_id) {
        current_core_entry.first = std::stoul(value);
        continue;
      }

      if (key == core_id) {
        current_core_entry.second = std::stoul(value);
        cores.insert(current_core_entry);
        continue;
      }
    }

    if (cores.size() != 0)
      return cores.size();

    return detectNumberOfCoresFallback();
  } catch (...) {
    return detectNumberOfCoresFallback();
  }
#elif defined(__APPLE__)
  int n;
  size_t size = sizeof(n);

  if (sysctlbyname("hw.physicalcpu", &n, &size, NULL, 0) == 0)
    return n;

  return detectNumberOfCoresFallback;
#else
  return detectNumberOfCoresFallback();
#endif
}
