// Perform a Fisher exact test on a 2x2 contingency table.
// Based on scipy's fisher test: https://github.com/scipy/scipy/blob/v1.7.0/scipy/stats/stats.py#L40757

use statrs::distribution::DiscreteCDF;
use statrs::distribution::{Discrete, Hypergeometric};
use std::cmp::{max, min};

fn binary_search(
    n: usize,
    n1: usize,
    n2: usize,
    mode: usize,
    p_exact: f64,
    epsilon: f64,
    upper: bool,
) -> usize {
    // Binary search in two-sided test with starting bound as argument
    let (mut min_val, mut max_val) = {
        if upper {
            (mode, n)
        } else {
            (0, mode)
        }
    };

    let population = n1 + n2;
    let successes = n1;
    let draws = n;
    let dist = Hypergeometric::new(population as u64, successes as u64, draws as u64).unwrap();

    let mut guess: usize = 0; // = -1;
    loop {
        if (max_val - min_val) <= 1 {
            break;
        }
        guess = {
            if max_val == min_val + 1 && guess == min_val {
                max_val
            } else {
                //(maxval + minval) // 2
                (max_val + min_val) / 2
            }
        };

        let p_guess = dist.pmf(guess as u64);

        let ng = {
            if upper {
                guess - 1
            } else {
                guess + 1
            }
        };
        let pmf_comp = dist.pmf(ng as u64);
        if p_guess <= p_exact && p_exact < pmf_comp {
            break;
        } else if p_guess < p_exact {
            max_val = guess
        } else {
            min_val = guess
        }
    }

    if guess == 0 {
        //-1 {
        guess = min_val
    }
    if upper {
        loop {
            if guess > 0 && dist.pmf(guess as u64) < p_exact * epsilon {
                guess -= 1;
            } else {
                break;
            }
        }
        loop {
            if dist.pmf(guess as u64) > p_exact / epsilon {
                guess += 1;
            } else {
                break;
            }
        }
    } else {
        loop {
            if dist.pmf(guess as u64) < p_exact * epsilon {
                guess += 1;
            } else {
                break;
            }
        }
        loop {
            if guess > 0 && dist.pmf(guess as u64) > p_exact / epsilon {
                guess -= 1;
            } else {
                break;
            }
        }
    }
    guess
}

// alternative : {'two-sided', 'less', 'greater'}, optional
// Defines the alternative hypothesis.
// The following options are available (default is 'two-sided'):
// * 'two-sided'
// * 'less': one-sided
// * 'greater': one-sided
// See the Notes for more details.
// Describe the destination type of a key attribute
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

const EPSILON: f64 = 1.0 - 1e-4;

pub fn fishers_exact(table: &[usize; 4], alternative: Alternative) -> f64 {
    // Fisher p_value like scipy

    // If both values in a row or column are zero, the p-value is 1 and
    // the odds ratio is NaN.
    if (table[0] == 0 && table[2] == 0) || (table[1] == 0 && table[3] == 0) {
        //return (f64::NAN, 1.0);
        return 1.0;
    }

    let odds_ratio = {
        if table[1] > 0 && table[2] > 0 {
            (table[0] * table[3]) as f64 / (table[1] * table[2]) as f64
        } else {
            f64::INFINITY
        }
    };

    let n1 = table[0] + table[1];
    let n2 = table[2] + table[3];
    let n = table[0] + table[2];

    let p_value = {
        let population = n1 + n2;
        let successes = n1;

        match alternative {
            Alternative::TwoSided => {
                let draw = n;
                let dist =
                    Hypergeometric::new(population as u64, successes as u64, draw as u64).unwrap();
                let p_exact = dist.pmf(table[0] as u64);

                let mode = (((n + 1) * (n1 + 1)) as f64 / (n1 + n2 + 2) as f64) as usize; // todo: check floor?
                let p_mode = dist.pmf(mode as u64);

                if (p_exact - p_mode).abs() / p_exact.max(p_mode) <= 1.0 - EPSILON {
                    return 1.0;
                }

                if table[0] < mode {
                    let p_lower = dist.cdf(table[0] as u64);
                    if dist.pmf(n as u64) > p_exact / EPSILON {
                        return p_lower;
                    }
                    let guess = binary_search(n, n1, n2, mode, p_exact, EPSILON, true);
                    let p_value = p_lower + (1.0 - dist.cdf((guess - 1) as u64));
                    return p_value;
                } else {
                    let p_upper = 1.0 - dist.cdf((table[0] - 1) as u64);
                    let p_value = {
                        if dist.pmf(0) > p_exact / EPSILON {
                            p_upper
                        } else {
                            let guess = binary_search(n, n1, n2, mode, p_exact, EPSILON, false);
                            p_upper + dist.cdf(guess as u64)
                        }
                    };
                    return p_value;
                }
            }
            Alternative::Less => {
                let draw = n;
                let dist =
                    Hypergeometric::new(population as u64, successes as u64, draw as u64).unwrap();
                dist.cdf(table[0] as u64)
            }
            Alternative::Greater => {
                let draw = table[1] + table[3];
                let dist =
                    Hypergeometric::new(population as u64, successes as u64, draw as u64).unwrap();
                dist.cdf(table[1] as u64)
            }
        }
    };

    p_value.min(1.0)
}

#[cfg(test)]
mod tests {
    use super::fishers_exact;
    use crate::fisher::Alternative;
    use float_cmp::assert_approx_eq;

    #[test]
    fn test_fishers_exact() {
        let cases = [
            (
                [3, 5, 4, 50],
                0.9963034765672599,
                0.03970749246529277,
                0.03970749246529276,
            ),
            (
                [61, 118, 2, 1],
                0.27535061623455315,
                0.9598172545684959,
                0.27535061623455315,
            ),
            (
                [172, 46, 90, 127],
                1.0,
                6.662405187351769e-16,
                9.041009036528785e-16,
            ),
            (
                [127, 38, 112, 43],
                0.8637599357870167,
                0.20040942958644145,
                0.3687862842650179,
            ),
            (
                [186, 177, 111, 154],
                0.9918518696328176,
                0.012550663906725129,
                0.023439141644624434,
            ),
            (
                [137, 49, 135, 183],
                0.999999999998533,
                5.6517533666400615e-12,
                8.870999836202932e-12,
            ),
            (
                [37, 115, 37, 152],
                0.8834621182590621,
                0.17638403366123565,
                0.29400927608021704,
            ),
            (
                [124, 117, 119, 175],
                0.9956704915461392,
                0.007134712391455461,
                0.011588218284387445,
            ),
            (
                [70, 114, 41, 118],
                0.9945558498544903,
                0.010384865876586255,
                0.020438291037108678,
            ),
            (
                [173, 21, 89, 7],
                0.2303739114068352,
                0.8808002774812677,
                0.4027047267306024,
            ),
            (
                [18, 147, 123, 58],
                4.077820702304103e-29,
                0.9999999999999817,
                0.0,
            ),
            (
                [116, 20, 92, 186],
                0.9999999999998267,
                6.598118571034892e-25,
                8.164831402188242e-25,
            ),
            (
                [9, 22, 44, 38],
                0.01584272038710196,
                0.9951463496539362,
                0.021581786662999272,
            ),
            (
                [9, 101, 135, 7],
                3.3336213533847776e-50,
                1.0,
                3.3336213533847776e-50,
            ),
            (
                [153, 27, 191, 144],
                0.9999999999950817,
                2.473736787266208e-11,
                3.185816623300107e-11,
            ),
            (
                [111, 195, 189, 69],
                6.665245982898848e-19,
                0.9999999999994574,
                1.0735744915712542e-18,
            ),
            (
                [125, 21, 31, 131],
                0.99999999999974,
                9.720661317939016e-34,
                1.0352129312860277e-33,
            ),
            (
                [201, 192, 69, 179],
                0.9999999988714893,
                3.1477232259550017e-09,
                4.761075937088169e-09,
            ),
            (
                [124, 138, 159, 160],
                0.30153826772785475,
                0.7538974235759873,
                0.5601766196310243,
            ),
        ];

        for &(table, less_expected, greater_expected, two_sided_expected) in cases.iter() {
            for (alternative, expected) in [
                Alternative::Less,
                Alternative::Greater,
                Alternative::TwoSided,
            ]
            .iter()
            .zip(vec![less_expected, greater_expected, two_sided_expected])
            {
                let p_value = fishers_exact(&table, *alternative);
                assert_approx_eq!(f64, p_value, expected, epsilon = 1e-12);
            }
        }
    }
}