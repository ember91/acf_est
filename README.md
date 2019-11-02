# Bartlett Autocorrelation Function Estimator

Quite fast Matlab mex Bartlett autocorrelation function estimator written in C++.

## Why

I wanted to write the fastest on-CPU Bartlett Autocorrelation Function *(ACF)* estimator in existence. Just for fun. 

## Benchmarks

In the table below are benchmarks against the fastest ACF implementation I could write in Matlab. The test was performed with a 2^17 x 8 input matrix. Remember that these speedups are for sufficiently large input vectors. 

| Computer               | Base frequency (GHz) | Instruction set | CPU cores | Speedup floats | Speedup doubles |
|------------------------|---------------------:|-----------------|----------:|---------------:|----------------:|
| Old stupid laptop      | 2.1                  | AVX2            | 4         | 130            | 60              |
| New fancy megacomputer | 2.2                  | AVX512          | 20        | 1100           | 360             |

## Definition

![Bartlett estimation formula](definition.png)

where *x* is the discrete input signal and *N* its number of elements.

## How

It uses vectorization and parallelization (threading) to achieve these results.

## Installing

Install Agner Fog's [vectorclass](https://github.com/vectorclass) library. It's a header-only library, so just extract/clone it into the project base directory.
```
cd acf_est
git clone https://github.com/vectorclass/version2.git
```
In matlab, compile this into mex with GCC or Clang as
```
mex CXXFLAGS='$CXXFLAGS -std=c++1z -O3 -march=native -Wall -Wextra -Wpedantic' acf_est.cpp
```
or with MSVC as
```
mex CXXFLAGS='$CXXFLAGS /std:c++17 /O2 /arch:AVX512 /Wall' acf_est.cpp
```
This ensures it compiles with high optimizations and vector instructions if available.

## Running tests and benchmarks

In Matlab, go to the project directory. Run *test.m*. It will both check for errors and run benchmarks.

## Caveats

For short vectors, the simple Matlab method is faster.

## Example

After compiling successfully, in Matlab, run
```
acf_est([1; 2; 3; 4])
```
to estimate. Write
```
help acf_est
```
for more details about command parameters and specifics.

## Common errors

| When      | Error | Solution |
|-----------|---------------|----------|
| Executing | "Attempt to execute SCRIPT acf_est as a function"                                  | Compile it                      |
| Executing | It gives the wrong answer                | Check that you run the binary on the machine it was compiled. Otherwise it may be a bug. |
| Compiling | "error: unrecognized command line option ‘-std=c++1z’; did you mean ‘-std=c++11’?" | Upgrade your compiler |
| Compiling | "fatal error: vectorclass/vectorclass.h: No such file or directory"                | Download the vectorclass library |

## Built With

* [vectorclass](https://github.com/vectorclass/version2) - Vectorization library
* [Visual Studio Code](https://code.visualstudio.com/) - Code editor

## Authors

* **Emil Berg** - [ember91](https://github.com/ember91)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
