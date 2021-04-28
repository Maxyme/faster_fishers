/*
Rust version of the fisher_exact test for python originally written in cython
 */
#![allow(clippy::missing_safety_doc)] // FIX (#698)

use numpy::{
    c64, IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray, PyReadonlyArray1, PyReadonlyArrayDyn,
    ToPyArray,
};
use pyo3::prelude::{pyfunction, pymethods, pymodule, PyModule, PyResult, Python};

//use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use fishers_exact::fishers_exact;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use rayon::iter::IntoParallelRefIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;
use std::convert::TryFrom;

#[pymodule]
fn faster_fishers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact_py, m)?)?;
    Ok(())
}

#[pyfunction]
fn exact_py<'py>(
    py: Python<'py>,
    a_values: PyReadonlyArray1<'py, u32>,
    b_values: PyReadonlyArray1<'py, u32>,
    c_values: PyReadonlyArray1<'py, u32>,
    d_values: PyReadonlyArray1<'py, u32>,
) -> PyResult<(
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
)> {
    /// Entrypoint for the Python method to convert from numpy arrays
    let (lesses, greaters, two_tails) = exact(
        a_values.as_array(),
        b_values.as_array(),
        c_values.as_array(),
        d_values.as_array(),
    );
    Ok((
        PyReadonlyArray1::from(lesses.to_pyarray(py)),
        PyReadonlyArray1::from(greaters.to_pyarray(py)),
        PyReadonlyArray1::from(two_tails.to_pyarray(py)),
    ))
}

fn exact(
    a: ArrayView1<u32>,
    b: ArrayView1<u32>,
    c: ArrayView1<u32>,
    d: ArrayView1<u32>,
) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    /// Perform fisher exact calculation on a given input array
    /// Note, perhaps this should be a 2d array instead.
    let mut lesses = Array1::<f64>::zeros(a.dim());
    let mut greaters = Array1::<f64>::zeros(a.dim());
    let mut two_tails = Array1::<f64>::zeros(a.dim());

    for index in 0..a.dim() {
        let p = fishers_exact(&[a[index], b[index], c[index], d[index]])
            .expect("Issue getting fisher's exact value!");
        lesses[index] = p.less_pvalue;
        greaters[index] = p.greater_pvalue;
        two_tails[index] = p.two_tail_pvalue
    }

    use rayon::prelude::*;
    fn sum_of_squares(input: &[i32]) -> i32 {
        input
            .par_iter() // <-- just change that!
            .map(|&i| i * i)
            .sum()
    }

    (lesses, greaters, two_tails)
}

fn exact_rayon(a: ArrayView2<u32>) -> (Array1<f64>, Array1<f64>, Array1<f64>) {
    //a.par_iter().map(|&i| fishers_exact(i))
    //a.par_iter().map(|&i| fishers_exact(i))
    let x = a
        .axis_iter(Axis(0))
        .par_bridge()
        .map(|i| fishers_exact(<&[u32; 4]>::try_from(i.as_slice().expect("")).unwrap())); //.collect();
                                                                                          //let mut output: Vec<&'static str> = rx.into_iter().par_bridge().collect();
    let v = x.collect();
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2, Array2};
    use rayon::iter::IntoParallelRefMutIterator;

    #[test]
    fn test_exact() {
        // Test the exact method
        let a_values = arr1(&[1, 3]);
        let b_values = arr1(&[2, 5]);
        let c_values = arr1(&[1, 4]);
        let d_values = arr1(&[5, 50]);
        let (lesses, greaters, two_tails) = exact(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
        );

        assert_eq!(lesses, arr1(&[0.9166666666666647, 0.9963034765672586]));
        assert_eq!(greaters, arr1(&[0.5833333333333326, 0.03970749246529451]));
        assert_eq!(two_tails, arr1(&[1.0, 0.03970749246529451]));
    }

    #[test]
    fn test_exact_rayon() {
        // Test the exact method with rayon

        let mut a = Array2::<f64>::zeros((128, 128));

        // Parallel versions of regular array methods
        a.par_map_inplace(|x| *x = x.exp());
        a.par_mapv_inplace(f64::exp);

        // You can also use the parallel iterator directly
        a.par_iter_mut().for_each(|x| *x = x.exp());

        let a_values = arr2(&[[1, 2, 1, 5]]);
        let (lesses, greaters, two_tails) = exact_rayon(a_values.view());
        assert_eq!(lesses, arr1(&[0.9166666666666647, 0.9963034765672586]));
        assert_eq!(greaters, arr1(&[0.5833333333333326, 0.03970749246529451]));
        assert_eq!(two_tails, arr1(&[1.0, 0.03970749246529451]));
    }
}
