# Bartlett Autocorrelation Function Estimator

Quite fast Matlab mex Bartlett autocorrelation function estimator written in C++.

This was a fun project to do. The goal was to make the fastest autocorrelation estimator possible. So I made something really fast. It uses vectorization and multithreading for maximum performance. Unfortunately, I realized that doing autocorrelation function estimation by Fourier transforming the signal, multiplying the transform with its conjugate and then reverse transforming back is much faster. I like the code I wrote, though.

## Definition

![Bartlett estimation formula](definition.png)

where *x* is the discrete input signal and *N* its number of elements.

## Authors

* **Emil Berg** - [ember91](https://github.com/ember91)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details.
