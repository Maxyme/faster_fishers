# faster_fishers
Fast implementation of Fisher's exact test in Rust for Python.  
Benchmarks show that this version is about 5x faster than scipy's version:

```asm
---------------------------------------------------------------------------------------------- benchmark: 2 tests ----------------------------------------------------------------------------------------------
Name (time in ms)                        Min                   Max                  Mean              StdDev                Median                 IQR            Outliers     OPS            Rounds  Iterations
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_faster_fischer       585.3761 (1.0)        589.6008 (1.0)        586.7834 (1.0)        1.6307 (1.0)        586.2591 (1.0)        1.2965 (1.0)           1;1  1.7042 (1.0)           5           1
test_benchmark_scipy              2,665.4561 (4.55)     3,448.9862 (5.85)     2,871.1293 (4.89)     328.2283 (201.28)   2,734.1146 (4.66)     292.4552 (225.58)        1;1  0.3483 (0.20)          5           1
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## Usage:
```python
>>> import numpy as np
>>> import faster_fishers
>>> lefts, right, two_tails = faster_fishers.exact(np.array([1, 3]), np.array([2, 5]), np.array([1, 4]), np.array([5, 50]))
>>> lefts
array([0.9166666666666647, 0.9963034765672586])
>>> rights
array([0.5833333333333326, 0.03970749246529451])
>>> two_tails
array([1.0, 0.03970749246529451])
```

## Developing

### Building with cargo
* Run `cargo build` in the main directory to build the project.


### Using locally
* Install environment: `poetry install`  
* Add environment to current shell `poetry shell`  
* Install faster_fishers in current environment: `maturin develop`  
* Check that it works: `python -c "import faster_fishers; print(dir(faster_fishers))"`  

To try the library in a different environment:  
    1. `maturin build --release`  
    2. `cd folder`  
    3. `pip install {wheel_path}.whl`  


### Benchmarks
* Make sure to compile in release mode with maturin first: `maturin develop --release`
*python: `pytest --benchmark-warmup -m benchmark`