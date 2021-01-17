# Running Stan algorithms with models defined in C++ (and possibly other languages)

This little project is the demonstration of [my branch `stan-cpp-interface`](https://github.com/IvanYashchuk/stan/tree/stan-cpp-interface).
It allows defining models in C++ only in terms of `std::vector<double>` or `Eigen::VectorXd`, that is without using Stan-Math library for automatic differentiation or Stan modelling language. This work also allows using Stan's algorithms for models defined in Python and Julia. I hope I will be able to finish the examples and release them in near future.

This example runs Stan's NUTS sampler.

The model is the [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).
In `rosenbrock.hpp` the `rosenbrock_model` class is implemented and it's compatible with Stan's algorithms using my  `stan-cpp-interface` branch.

For an inspiration on how to call various Stan's algorithms see `command.hpp` from CmdStan, or `command` function from [PyStan's `stan_fit.hpp`](https://github.com/stan-dev/pystan/blob/50f491cba49c36b4d4c6ade25e9378b21921603b/pystan/stan_fit.hpp#L791).

# How to run?
You need to have CMake installed.

Clone my fork (recursively) and checkout the branch

    git clone --recursive --single-branch --branch stan-cpp-interface https://github.com/IvanYashchuk/stan.git

Set the environment variable `STAN_DIR` pointing to the location of the fork

    export STAN_DIR=/absolute/path/to/my/fork

Build Stan-Math dependencies (Stan depends on Stan-Math so it's still linked here)

    cd $STAN_DIR/lib/stan_math && make -f make/standalone math-libs && cd -

Go to this repository's folder and run

    mkdir build && cd build && cmake .. && make

It should produce an executable `rosenbrock`. Now run it

    ./rosenbrock

It will output a csv file `output_rosenbrock.csv`:

    ./stansummary /path/to/rosenbrock_interface/build/output_rosenbrock.csv

```
Input file: output_rosenbrock.csv
Warning: non-fatal error reading adapation data
Inference for Stan model: rosenbrock_model
1 chains: each with iter=(11000); warmup=(0); thin=(0); 11000 iterations saved.

Warmup took 0.067 seconds
Sampling took 0.68 seconds

                   Mean     MCSE  StdDev        5%    50%    95%  N_Eff  N_Eff/s  R_hat

lp__              -0.94    0.021    0.93  -2.8e+00  -0.65 -0.046   1980     2929    1.0
accept_stat__   9.0e-01  6.7e-03    0.17      0.52   0.97    1.0    661      978    1.0
stepsize__      4.3e-02  3.4e-03   0.022     0.039  0.039  0.067     41       61    1.0
treedepth__     3.8e+00  2.3e-02     1.8       1.0    4.0    6.0   5952     8805    1.0
n_leapfrog__    3.7e+01  5.0e-01      38       1.0     23    111   5912     8745   1.00
divergent__     2.8e-03  5.5e-04   0.053      0.00   0.00   0.00   9437    13960    1.0
energy__        1.9e+00  2.8e-02     1.4      0.35    1.6    4.6   2382     3524    1.0

theta[1]           0.97    0.021    0.68  -1.7e-01   0.97    2.1   1092     1615   1.00
theta[2]            1.4    0.043     1.4   1.1e-03   0.96    4.4   1105     1634    1.0

Samples were drawn using  with .
For each parameter, N_Eff is a crude measure of effective sample size,
and R_hat is the potential scale reduction factor on split chains (at 
convergence, R_hat=1).
```
