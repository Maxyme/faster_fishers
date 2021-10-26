use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ndarray::arr1;
//use faster_fisher::{exact_test};
use crate::fisher::{fishers_exact, fishers_exact_with_odds_ratio, Alternative};

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n-1) + fibonacci(n-2),
    }
}



fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("fib 20", |b| b.iter(|| fibonacci(black_box(20))));
    // let a_values = arr1(&[1, 3]);
    // let b_values = arr1(&[2, 5]);
    // let c_values = arr1(&[1, 4]);
    // let d_values = arr1(&[5, 50]);
    // let lesses = exact_test(
    //     a_values.view(),
    //     b_values.view(),
    //     c_values.view(),
    //     d_values.view(),
    //     Alternative::Less,
    // );
    // c.bench_function("fib 20", |b| b.iter(|| exact_test( a_values.view(),
    //                                                                       b_values.view(),
    //                                                                       c_values.view(),
    //                                                                       d_values.view(),
    //                                                                       Alternative::Less,)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);