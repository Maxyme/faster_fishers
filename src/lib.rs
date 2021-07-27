/* Rust version of the fisher_exact test for python originally written in cython */
#![allow(unused_doc_comments)]

mod fisher;

use numpy::{PyReadonlyArray1, ToPyArray, PyReadonlyArray2};
use pyo3::prelude::{pyfunction, pymodule, PyModule, PyResult, Python};

use pyo3::wrap_pyfunction;

use ndarray::{Array1, ArrayView1, Array2};

use rayon::iter::IntoParallelIterator;

use crate::fisher::{fishers_exact, fishers_exact_with_odds_ratio, Alternative};
use cached::proc_macro::cached;
use rayon::prelude::ParallelIterator;

#[pymodule]
fn faster_fishers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact, m)?)?;
    m.add_function(wrap_pyfunction!(exact_with_odds_ratios, m)?)?;
    Ok(())
}

#[pyfunction]
fn exact<'py>(
    py: Python<'py>,
    a_values: PyReadonlyArray1<'py, u64>,
    b_values: PyReadonlyArray1<'py, u64>,
    c_values: PyReadonlyArray1<'py, u64>,
    d_values: PyReadonlyArray1<'py, u64>,
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

    let return_value = exact_test(
        a_values.as_array(),
        b_values.as_array(),
        c_values.as_array(),
        d_values.as_array(),
        alternative_enum,
    );

    Ok(PyReadonlyArray1::from(return_value.to_pyarray(py)))
}

#[pyfunction]
fn exact_with_odds_ratios<'py>(
    py: Python<'py>,
    a_values: PyReadonlyArray1<'py, u64>,
    b_values: PyReadonlyArray1<'py, u64>,
    c_values: PyReadonlyArray1<'py, u64>,
    d_values: PyReadonlyArray1<'py, u64>,
    alternative: String,
) -> PyResult<PyReadonlyArray2<'py, f64>> {
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

    let return_value = exact_test_with_odds_ratio(
        a_values.as_array(),
        b_values.as_array(),
        c_values.as_array(),
        d_values.as_array(),
        alternative_enum,
    );

    Ok(PyReadonlyArray2::from(return_value.to_pyarray(py)))
}

#[cached]
fn cached_fisher_exact(table: [u64; 4], alternative: Alternative) -> f64 {
    /// Cached version of the code
    fishers_exact(&table, alternative)
}

fn exact_test(
    a: ArrayView1<u64>,
    b: ArrayView1<u64>,
    c: ArrayView1<u64>,
    d: ArrayView1<u64>,
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

#[cached]
fn cached_fisher_exact_with_odds_ratios(table: [u64; 4], alternative: Alternative) -> (f64, f64) {
    /// Cached version of the code
    fishers_exact_with_odds_ratio(&table, alternative)
}

fn exact_test_with_odds_ratio(
    a: ArrayView1<u64>,
    b: ArrayView1<u64>,
    c: ArrayView1<u64>,
    d: ArrayView1<u64>,
    alternative: Alternative,
) -> Array2<f64> {
    /// Perform fisher exact calculation on a given input array
    /// Note, perhaps this should be a 2d array instead.
    let range = 0..a.dim();

    let return_values: Vec<(f64, f64)> = range
        .into_par_iter()
        .map(|index| {
            cached_fisher_exact_with_odds_ratios(
                [a[index], b[index], c[index], d[index]],
                alternative,
            )
        })
        .collect();

    let mut arr = Array2::<f64>::default((2, return_values.len()));
    for (index, (odds_ratio, p_value)) in return_values.iter().enumerate() {
        arr[[0, index]] = *odds_ratio;
        arr[[1, index]] = *p_value;
    }
    arr
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::assert_approx_eq;
    use ndarray::arr1;
    #[test]
    fn test_exact() {
        // Test the exact method
        let a_values = arr1(&[1, 3]);
        let b_values = arr1(&[2, 5]);
        let c_values = arr1(&[1, 4]);
        let d_values = arr1(&[5, 50]);
        let lesses = exact_test(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
            Alternative::Less,
        );

        assert_approx_eq!(f64, lesses[0], 0.9166666666666659, epsilon = 1e-12);
        assert_approx_eq!(f64, lesses[1], 0.9963034765672599, epsilon = 1e-12);

        let greaters = exact_test(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
            Alternative::Greater,
        );
        assert_approx_eq!(f64, greaters[0], 0.5833333333333328, epsilon = 1e-12);
        assert_approx_eq!(f64, greaters[1], 0.03970749246529277, epsilon = 1e-12);

        let two_tails = exact_test(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
            Alternative::TwoSided,
        );
        assert_approx_eq!(f64, two_tails[0], 1.0, epsilon = 1e-12);
        assert_approx_eq!(f64, two_tails[1], 0.03970749246529276, epsilon = 1e-12);
    }
}
