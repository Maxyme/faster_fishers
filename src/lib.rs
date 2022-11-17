#![allow(unused_doc_comments)]

mod fishers;

use numpy::ndarray::{stack, Array1, Axis};
use numpy::{array, IntoPyArray, PyArray1, PyArray2, PyReadonlyArrayDyn};
use pyo3::{pymodule, types::PyModule, PyResult, Python};

use rayon::prelude::*;

pub use fishers::{fishers_exact, fishers_exact_with_odds_ratio, Alternative};

#[pymodule]
fn faster_fishers(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    #[pyo3(name = "exact")]
    fn exact_py<'py>(
        py: Python<'py>,
        a: PyReadonlyArrayDyn<'_, u64>,
        b: PyReadonlyArrayDyn<'_, u64>,
        c: PyReadonlyArrayDyn<'_, u64>,
        d: PyReadonlyArrayDyn<'_, u64>,
        alternative: &str,
    ) -> &'py PyArray1<f64> {
        /// Entrypoint for the Python method to convert from numpy arrays
        let alternative_enum = match alternative {
            "less" => Alternative::Less,
            "greater" => Alternative::Greater,
            "two-sided" => Alternative::TwoSided,
            _ => panic!("Error: `alternative` should be one of ['two-sided', 'less', 'greater']"),
        };

        // Todo: check if we can index PyReadonlyArray directly
        let a = a.as_array();
        let b = b.as_array();
        let c = c.as_array();
        let d = d.as_array();

        let range = 0..a.len();

        let p_values: Vec<f64> = range
            .into_par_iter()
            .map(|index| {
                fishers_exact(&[a[index], b[index], c[index], d[index]], alternative_enum)
                    .expect("Statrs error with the given input.")
            })
            .collect();

        let arr = Array1::from(p_values);
        arr.into_pyarray(py)
    }

    #[pyfn(m)]
    #[pyo3(name = "exact_with_odds_ratios")]
    fn exact_with_odds_ratios_py<'py>(
        py: Python<'py>,
        a: PyReadonlyArrayDyn<'_, u64>,
        b: PyReadonlyArrayDyn<'_, u64>,
        c: PyReadonlyArrayDyn<'_, u64>,
        d: PyReadonlyArrayDyn<'_, u64>,
        alternative: &str,
    ) -> &'py PyArray2<f64> {
        /// Entrypoint for the Python method to convert from numpy arrays
        let alternative_enum = match alternative {
            "less" => Alternative::Less,
            "greater" => Alternative::Greater,
            "two-sided" => Alternative::TwoSided,
            _ => panic!("Error: `alternative` should be one of ['two-sided', 'less', 'greater']"),
        };

        // Todo: check if we can use the PyReadonlyArray directly
        let a = a.as_array();
        let b = b.as_array();
        let c = c.as_array();
        let d = d.as_array();

        let range = 0..a.len();

        let odds_p_values: Vec<Array1<f64>> = range
            .into_par_iter()
            .map(|index| {
                let (odds_ratio, p_value) = fishers_exact_with_odds_ratio(
                    &[a[index], b[index], c[index], d[index]],
                    alternative_enum,
                )
                .expect("Statrs error with the given input.");
                array![odds_ratio, p_value]
            })
            .collect();

        // Convert into an array of odds_ratios and p_values
        let array_2 = stack![Axis(1), odds_p_values[0], odds_p_values[1]];
        array_2.into_pyarray(py)
    }
    Ok(())
}
