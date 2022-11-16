# faster_fishers
Fast implementation of Fisher's exact test in Rust for Python.  
Benchmarks show that this version is about 30x faster than scipy's version when running on a large range of inputs and about 10x faster when running on 1 input:

```asm
---------------------------------------------------------------------------------------------- benchmark: 2 tests ---------------------------------------------------------------------------------------------
Name (time in ms)                        Min                   Max                  Mean             StdDev                Median                IQR            Outliers      OPS            Rounds  Iterations
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
test_benchmark_faster_fischer        59.8543 (1.0)         61.0522 (1.0)         60.5012 (1.0)       0.2816 (1.0)         60.5717 (1.0)       0.3114 (1.0)           3;1  16.5286 (1.0)          17           1
test_benchmark_scipy              1,859.7465 (31.07)    1,935.8237 (31.71)    1,885.3549 (31.16)    30.7295 (109.12)   1,871.9271 (30.90)    38.2479 (122.85)        1;0   0.5304 (0.03)          5           1
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
```

## Usage:

Rust:
```rust
use faster_fishers::{fishers_exact_with_odds_ratio, Alternative};
let table = [3, 5, 4, 50];
let alternative = Alternative::Less;
let (p_value, odds_ratio) = fishers_exact_with_odds_ratio(&table, alternative).unwrap();
```

Python:

```python
>>> import numpy as np
>>> import fishers
>>> lefts, right, two_tails = fishers.exact(np.array([1, 3]), np.array([2, 5]), np.array([1, 4]), np.array([5, 50]))
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

### Publishing on pypi
`docker run --rm -v $(pwd):/io ghcr.io/pyo3/maturin publish -u {USER} -p {PASSWORD}`

### Publishing on cargo
`cargo publish`

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
* Make sure to compile in release mode with maturin first: `RUSTFLAGS='-C target-cpu=native' maturin develop --release`
*python: `pytest --benchmark-warmup -m benchmark`