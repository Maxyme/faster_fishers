# faster_fishers
Fast implementation of Fisher's exact test in Rust for Python.

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
*python: `pytest --benchmark-warmup -m benchmark`