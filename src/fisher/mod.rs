use statrs::distribution::{Discrete, DiscreteCDF, Hypergeometric};
use statrs::StatsError;

#[derive(Debug, Clone, Copy)]
pub enum Alternative {
    TwoSided,
    Less,
    Greater,
}

const EPSILON: f64 = 1.0 - 1e-4;
/// Binary search in two-sided test with starting bound as argument
fn binary_search<F>(
    min_val: u64,
    max_val: u64,
    p_exact: f64,
    epsilon: f64,
    upper: bool,
    func: F
) -> u64 where
    F: Fn(u64) -> f64 {

    let (mut min_val, mut max_val) = (min_val, max_val);

    let mut guess = 0;
    loop {
        if max_val - min_val <= 1 {
            break;
        }
        guess = {
            if max_val == min_val + 1 && guess == min_val {
                max_val
            } else {
                (max_val + min_val) / 2
            }
        };

        let ng = {
            if upper {
                guess - 1
            } else {
                guess + 1
            }
        };

        let pmf_comp = func(ng);
        let p_guess = func(guess);
        if p_guess <= p_exact && p_guess < pmf_comp {
            break;
        }
        if p_guess < p_exact {
            max_val = guess
        } else {
            min_val = guess
        }
    }

    if guess == 0 {
        guess = min_val
    }
    if upper {
        while guess > 0 && func(guess) < p_exact * epsilon {
            guess -= 1;
        }
        while func(guess) > p_exact / epsilon {
            guess += 1;
        }
    } else {
        while func(guess) < p_exact * epsilon {
            guess += 1;
        }
        while guess > 0 && func(guess) > p_exact / epsilon {
            guess -= 1;
        }
    }
    guess
}

/// Perform a Fisher exact test on a 2x2 contingency table.
/// Based on scipy's fisher test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html#scipy-stats-fisher-exact
/// Returns the odds ratio and p_value
/// # Examples
///
/// ```
/// use statrs::statis_tests::{Alternative, fishers_exact};
/// let table = [3, 5, 4, 50];
/// let (odds_ratio, p_value) = fishers_exact_with_odds_ratio(&table, Alternative::Less).unwrap();
/// ```
pub fn fishers_exact_with_odds_ratio(
    table: &[u64; 4],
    alternative: Alternative,
) -> Result<(f64, f64), StatsError> {
    if (table[0] == 0 && table[2] == 0) || (table[1] == 0 && table[3] == 0) {
        // If both values in a row or column are zero, p-value is 1 and odds ratio is NaN.
        return Ok((f64::NAN, 1.0));
    }

    let odds_ratio = {
        if table[1] * table[2] == 0 {
            // Prevent division by zero
            f64::INFINITY
        } else {
            (table[0] * table[3]) as f64 / (table[1] * table[2]) as f64
        }
    };

    let p_value = fishers_exact(table, alternative)?;
    Ok((odds_ratio, p_value))
}

/// Perform a Fisher exact test on a 2x2 contingency table.
/// Based on scipy's fisher test: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html#scipy-stats-fisher-exact
/// Returns only the p_value
/// # Examples
///
/// ```
/// use statrs::statis_tests::{Alternative, fishers_exact};
/// let table = [3, 5, 4, 50];
/// let p_value = fishers_exact(&table, Alternative::Less).unwrap();
/// ```
pub fn fishers_exact(table: &[u64; 4], alternative: Alternative) -> Result<f64, StatsError> {
    // If both values in a row or column are zero, the p-value is 1 and
    // the odds ratio is NaN.
    if (table[0] == 0 && table[2] == 0) || (table[1] == 0 && table[3] == 0) {
        return Ok(1.0);
    }

    let n1 = table[0] + table[1];
    let n2 = table[2] + table[3];
    let n = table[0] + table[2];

    let p_value = {
        let population = n1 + n2;
        let successes = n1;
        let draws = n;
        let dist = Hypergeometric::new(population, successes, draws)?;

        match alternative {
            Alternative::Less => {
                dist.cdf(table[0])
            }
            Alternative::Greater => {
                let draws = table[1] + table[3];
                let dist = Hypergeometric::new(population, successes, draws)?;
                dist.cdf(table[1])
            }
            Alternative::TwoSided => {
                let p_exact = dist.pmf(table[0]);
                let mode = ((n + 1) * (n1 + 1)) / (n1 + n2 + 2) as u64; // todo: check floor?
                let p_mode = dist.pmf(mode);

                if (p_exact - p_mode).abs() / p_exact.max(p_mode) <= 1.0 - EPSILON {
                    return Ok(1.0);
                }

                let func = |x| dist.pmf(x);
                if table[0] < mode {
                    let p_lower = dist.cdf(table[0]);
                    if dist.pmf(n) > p_exact / EPSILON {
                        return Ok(p_lower);
                    }
                    let guess = binary_search(mode, n, p_exact, EPSILON, true, func);
                    return Ok(p_lower + 1.0 - dist.cdf(guess - 1));
                }

                let p_upper = 1.0 - dist.cdf(table[0] - 1);
                if dist.pmf(0) > p_exact / EPSILON {
                    return Ok(p_upper);
                }

                let guess = binary_search(0, mode, p_exact, EPSILON, false, func);
                p_upper + dist.cdf(guess)
            }
        }
    };

    Ok(p_value.min(1.0))
}

#[cfg(test)]
mod tests {
    use super::{fishers_exact, fishers_exact_with_odds_ratio, Alternative};
    use float_cmp::assert_approx_eq;

    /// Test fishers_exact by comparing against values from scipy.
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

        for (table, less_expected, greater_expected, two_sided_expected) in cases.iter() {
            for (alternative, expected) in [
                Alternative::Less,
                Alternative::Greater,
                Alternative::TwoSided,
            ]
            .iter()
            .zip(vec![less_expected, greater_expected, two_sided_expected])
            {
                let p_value = fishers_exact(&table, *alternative).unwrap();
                assert_approx_eq!(f64, p_value, *expected, epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn test_fishers_exact_with_odds() {
        let table = [3, 5, 4, 50];
        let (odds_ratio, p_value) =
            fishers_exact_with_odds_ratio(&table, Alternative::Less).unwrap();
        assert_approx_eq!(f64, p_value, 0.9963034765672599, epsilon = 1e-12);
        assert_approx_eq!(f64, odds_ratio, 7.5, epsilon = 1e-1);
    }
}
