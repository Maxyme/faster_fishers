/* Rust version of the fisher_exact test for python originally written in cython */
#![allow(unused_doc_comments)]

mod fisher;

use numpy::{PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::prelude::{pyfunction, pymodule, PyModule, PyResult, Python};

use pyo3::wrap_pyfunction;

use ndarray::{Array1, ArrayView1, Array2, Array};

use rayon::iter::IntoParallelIterator;

use crate::fisher::{fishers_exact, Alternative};
use cached::proc_macro::cached;
use rayon::prelude::ParallelIterator;

#[pymodule]
fn faster_fishers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact_py, m)?)?;
    Ok(())
}

#[pyfunction]
fn exact_py<'py>(
    py: Python<'py>,
    a_values: PyReadonlyArray1<'py, usize>,
    b_values: PyReadonlyArray1<'py, usize>,
    c_values: PyReadonlyArray1<'py, usize>,
    d_values: PyReadonlyArray1<'py, usize>,
    alternative: String,
) -> PyResult<PyReadonlyArray1<'py, f64>> {
    /// Entrypoint for the Python method to convert from numpy arrays
    let alternative_enum = match alternative.as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        "two-sided" => Alternative::TwoSided,
        _ => panic!(
            "{}",
            "Error: `alternative` should be one of {'two-sided', 'less', 'greater'}"
        ),
    };

    let return_value = exact(
        a_values.as_array(),
        b_values.as_array(),
        c_values.as_array(),
        d_values.as_array(),
        alternative_enum,
    );

    Ok(PyReadonlyArray1::from(return_value.to_pyarray(py)))
}

#[cached]
fn cached_fisher_exact(table: [usize; 4], alternative: Alternative) -> f64 {
    /// Cached version of the code
    fishers_exact(&table, alternative)
}

fn exact(
    a: ArrayView1<usize>,
    b: ArrayView1<usize>,
    c: ArrayView1<usize>,
    d: ArrayView1<usize>,
    alternative: Alternative,
) -> Array1<f64> {
    /// Perform fisher exact calculation on a given input array
    /// Note, perhaps this should be a 2d array instead.
    let range = 0..a.dim();

    let return_values: Vec<f64> = range
        .into_par_iter()
        .map(|index| cached_fisher_exact([a[index], b[index], c[index], d[index]], alternative))
        .collect();

    Array1::from(return_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_exact() {
        // Test the exact method
        let a_values = arr1(&[1, 3]);
        let b_values = arr1(&[2, 5]);
        let c_values = arr1(&[1, 4]);
        let d_values = arr1(&[5, 50]);
        let lesses = exact(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
            Alternative::Less,
        );

        assert_eq!(lesses, arr1(&[0.9166666666666647, 0.9963034765672586]));

        let greaters = exact(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
            Alternative::Greater,
        );
        assert_eq!(greaters, arr1(&[0.5833333333333326, 0.03970749246529451]));

        let two_tails = exact(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
            Alternative::TwoSided,
        );
        assert_eq!(two_tails, arr1(&[1.0, 0.03970749246529451]));
    }
}
