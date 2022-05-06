# faster_fishers
Fast implementation of Fisher's exact test in Rust for Python.  
Benchmarks show that this version is about 20x faster than scipy's version:

```asm
--------------------------------------------------------------------------------------------- benchmark: 2 tests ---------------------------------------------------------------------------------------------
Name (time in ms)                        Min                   Max                  Mean             StdDev                Median                IQR            Outliers     OPS            Rounds  Iterations
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_faster_fischer       114.5763 (1.0)        119.2121 (1.0)        116.1288 (1.0)       1.6386 (1.0)        115.5922 (1.0)       1.6018 (1.0)           2;1  8.6111 (1.0)           9           1
test_benchmark_scipy              2,403.8024 (20.98)    2,458.8598 (20.63)    2,423.2871 (20.87)    21.4687 (13.10)    2,415.6504 (20.90)    24.6082 (15.36)         1;0  0.4127 (0.05)          5           1
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
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