/*
Rust version of the fisher_exact test for python originally written in cython
 */
#![allow(unused_doc_comments)]

mod fisher;

use numpy::{PyReadonlyArray1, ToPyArray, };
use pyo3::prelude::{pyfunction, pymodule, PyModule, PyResult, Python};

use pyo3::wrap_pyfunction;

use ndarray::{Array1, ArrayView1};

use rayon::iter::{IntoParallelRefIterator, IntoParallelIterator};
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelBridge;
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
) -> PyResult<(PyReadonlyArray1<'py, f64>)> {
    /// Entrypoint for the Python method to convert from numpy arrays

    let alternative_enum = match alternative.as_str() {
        "less" => Alternative::Less,
        "greater" => Alternative::Greater,
        "two-sided" => Alternative::TwoSided,
        _ => panic!("{}", "ValueError: `alternative` should be one of {'two-sided', 'less', 'greater'}")
    };

    //let new_a: ArrayView1<u32> = ArrayView1::from(a_values.as_array());
    let return_value = exact(
        a_values.as_array(),
        b_values.as_array(),
        c_values.as_array(),
        d_values.as_array(),
        alternative_enum
    );
    Ok(PyReadonlyArray1::from(return_value.to_pyarray(py)))
}

use cached::proc_macro::cached;
use crate::fisher::{FishersExactPvalues, fishers_exact, Alternative};

#[cached]
fn cached_fisher_exact(table: [usize; 4], alternative:Alternative) -> f64 {
    /// Cached version of the code
    fishers_exact(&table, alternative)
}

fn exact(
    a: ArrayView1<usize>,
    b: ArrayView1<usize>,
    c: ArrayView1<usize>,
    d: ArrayView1<usize>,
    alternative: Alternative
) -> Array1<f64> {
    /// Perform fisher exact calculation on a given input array
    /// Note, perhaps this should be a 2d array instead.
    // let mut return_values = Array1::<f64>::zeros(a.dim());
    //
    // for index in 0..a.dim() {
    //     return_values[index] = cached_fisher_exact([a[index], b[index], c[index], d[index]], alternative);
    // }
    //
    // return_values

    let values = 0..a.dim();

    let return_values = values.into_iter()//.iter()
        //.zip(&buffer[i - 12..i])
        .map(|index| cached_fisher_exact([a[index], b[index], c[index], d[index]], alternative));
        //.sum::<i64>() >> qlp_shift;
    //let delta = buffer[i];
    return_values.collect()
}

// fn exact_rayon(a: ArrayView2<u32>) { //(Array1<f64>, Array1<f64>, Array1<f64>) {
//     //vec![obj1, obj2, obj3].par_iter().map(|o| { operation_on(o) })
//
//     let a = Array::linspace(0., 63., 64).into_shape((4, 16)).unwrap();
//     let mut sums = Vec::new();
//     a.axis_iter(Axis(0))
//         .into_par_iter()
//         .map(|row| row.sum())
//         .collect_into_vec(&mut sums);
//
//     //a.axis_iter(Axis(0))
//
//     // for a in a.into_par_iter() {//.into_iter().par_bridge().par_iter() {
//     //     println!("{:?}", a);
//     // }
//
//     // for a in a.lanes(Axis(1)).into_par_iter() {//.into_iter().par_bridge().par_iter() {
//     //     println!("{:?}", a);
//     // }
//     //a.par_iter().map(|&i| fishers_exact(i))
//     //a.par_iter().map(|&i| fishers_exact(i))
//     // let x = a.axis_iter(Axis(0)).par_iter()
//     //     //.par_bridge()
//     //     .map(|i| fishers_exact(<&[u32; 4]>::try_from(i.as_slice().expect("")).unwrap())); //.collect();
//     // //let mut output: Vec<&'static str> = rx.into_iter().par_bridge().collect();
//     // let v = x.collect();
//     // v
// }

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;
    use rayon::iter::IntoParallelRefMutIterator;

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
            Alternative::Less
        );

        assert_eq!(lesses, arr1(&[0.9166666666666647, 0.9963034765672586]));

        let greaters = exact(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
            Alternative::Greater
        );
        assert_eq!(greaters, arr1(&[0.5833333333333326, 0.03970749246529451]));

        let two_tails = exact(
            a_values.view(),
            b_values.view(),
            c_values.view(),
            d_values.view(),
            Alternative::TwoSided
        );
        assert_eq!(two_tails, arr1(&[1.0, 0.03970749246529451]));
    }

    #[test]
    fn test_exact_rayon() {
        // Test the exact method with rayon

        //let mut a = Array2::<f64>::zeros((128, 128));

        // Parallel versions of regular array methods
        // a.par_map_inplace(|x| *x = x.exp());
        // a.par_mapv_inplace(f64::exp);
        //
        // // You can also use the parallel iterator directly
        // a.par_iter_mut().for_each(|x| *x = x.exp());

        //let a_values = arr2(&[[1, 2, 1, 5], [3, 5, 4, 5]]);
        //exact_rayon(a_values.view());
        //let (lesses, greaters, two_tails) = exact_rayon(a_values.view());
        // assert_eq!(lesses, arr1(&[0.9166666666666647, 0.9963034765672586]));
        // assert_eq!(greaters, arr1(&[0.5833333333333326, 0.03970749246529451]));
        // assert_eq!(two_tails, arr1(&[1.0, 0.03970749246529451]));
    }
}
