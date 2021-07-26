// Perform a Fisher exact test on a 2x2 contingency table.
// Based on scipy's fisher test: https://github.com/scipy/scipy/blob/v1.7.0/scipy/stats/stats.py#L40757

// Based on fishers_exact: https://docs.rs/fishers_exact/1.0.1/src/fishers_exact/lib.rs.html#8-426

fn lngamm(z: f64) -> f64
// Reference: "Lanczos, C. 'A precision approximation
// of the gamma function', J. SIAM Numer. Anal., B, 1, 86-96, 1964."
// Translation of  Alan Miller's FORTRAN-implementation
// See http://lib.stat.cmu.edu/apstat/245
{
    let mut x = 0.0;
    x += 0.1659470187408462e-06 / (z + 7.0);
    x += 0.9934937113930748e-05 / (z + 6.0);
    x -= 0.1385710331296526 / (z + 5.0);
    x += 12.50734324009056 / (z + 4.0);
    x -= 176.6150291498386 / (z + 3.0);
    x += 771.3234287757674 / (z + 2.0);
    x -= 1259.139216722289 / (z + 1.0);
    x += 676.5203681218835 / (z);
    x += 0.9999999999995183;
    x.ln() - 5.58106146679532777 - z + (z - 0.5) * (z + 6.5).ln()
}

fn lnfact(n: usize) -> f64 {
    if n <= 1 {
        return 0.0;
    }
    lngamm((n + 1) as f64)
}

fn lnbico(n: usize, k: usize) -> f64 {
    lnfact(n) - lnfact(k) - lnfact(n - k)
}

fn hyper_323(n11: usize, n1_: usize, n_1: usize, n: usize) -> f64 {
    (lnbico(n1_, n11) + lnbico(n - n1_, n_1 - n11) - lnbico(n, n_1)).exp()
}

fn hyper(s: &mut HyperState, n11: usize) -> f64 {
    hyper0(s, n11, 0, 0, 0)
}

struct HyperState {
    n11: usize,
    n1_: usize,
    n_1: usize,
    n: usize,
    prob: f64,
    valid: bool,
}

impl HyperState {
    fn new() -> HyperState {
        HyperState {
            n11: 0,
            n1_: 0,
            n_1: 0,
            n: 0,
            prob: 0.0,
            valid: false,
        }
    }
}

fn hyper0(s: &mut HyperState, n11i: usize, n1_i: usize, n_1i: usize, ni: usize) -> f64 {
    if s.valid && (n1_i | n_1i | ni) == 0 {
        if n11i % 10 != 0 {
            if n11i == s.n11 + 1 {
                s.prob *= ((s.n1_ - s.n11) as f64 / n11i as f64)
                    * ((s.n_1 - s.n11) as f64 / (n11i + s.n - s.n1_ - s.n_1) as f64);
                s.n11 = n11i;
                return s.prob;
            }
            if n11i + 1 == s.n11 {
                s.prob *= ((s.n11 as f64) / (s.n1_ - n11i) as f64)
                    * ((s.n11 + s.n - s.n1_ - s.n_1) as f64 / (s.n_1 - n11i) as f64);
                s.n11 = n11i;
                return s.prob;
            }
        }
        s.n11 = n11i;
    } else {
        s.n11 = n11i;
        s.n1_ = n1_i;
        s.n_1 = n_1i;
        s.n = ni;
        s.valid = true
    }
    return hyper_323(s.n11, s.n1_, s.n_1, s.n);
}

// Returns prob,sleft,sright,sless,slarg
fn exact(n11: usize, n1_: usize, n_1: usize, n: usize) -> (f64, f64, f64, f64, f64) {
    let mut s_left: f64;
    let mut s_right: f64;
    let sless: f64;
    let slarg: f64;
    let mut p: f64;
    let mut i;
    let mut j;
    let prob: f64;
    let mut max = n1_;
    if n_1 < max {
        max = n_1;
    }
    let min = {
        if (n1_ + n_1) > n {
            n1_ + n_1 - n
        } else {
            0
        }
    };

    if min == max {
        return (1.0, 1.0, 1.0, 1.0, 1.0);
    }
    let mut s = HyperState::new();
    prob = hyper0(&mut s, n11, n1_, n_1, n);
    s_left = 0.0;
    p = hyper(&mut s, min);
    i = min + 1;
    while p <= 0.99999999 * prob {
        s_left += p;
        p = hyper(&mut s, i);
        i += 1;
    }
    i -= 1;
    if p <= 1.00000001 * prob {
        s_left += p;
    } else {
        i += 1;
    }
    s_right = 0.0;
    p = hyper(&mut s, max);
    j = max - 1;
    while p <= 0.99999999 * prob {
        s_right += p;
        p = hyper(&mut s, j);
        j -= 1;
    }
    j += 1;
    if p <= 1.00000001 * prob {
        s_right += p;
    } else {
        j += 1;
    }
    if (i as isize - n11 as isize).abs() < (j as isize - n11 as isize).abs() {
        sless = s_left;
        slarg = 1.0 - s_left + prob;
    } else {
        sless = 1.0 - s_right + prob;
        slarg = s_right;
    }
    (prob, s_left, s_right, sless, slarg)
}

/// `FishersExactPvalues` holds the pvalues calculated by the `fishers_exact` function.
#[derive(Clone, Copy, Debug)]
pub struct FishersExactPvalues {
    /// pvalue for the two-tailed test. Use this when there is no prior alternative.
    pub two_tail_pvalue: f64,
    /// pvalue for the "left" or "lesser" tail. Use this when the alternative to
    /// independence is that there is negative association between the variables.
    /// That is, the observations tend to lie in lower left and upper right.
    pub less_pvalue: f64,
    /// Use this when the alternative to independence is that there is positive
    /// association between the variables. That is, the observations tend to lie
    /// in upper left and lower right.
    pub greater_pvalue: f64,
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

/// Computes the Fisher's exact pvales to determine if there are nonrandom associations between two
/// categorical variables, in a two by two contingency table.
///
/// The test is computed using code ported from Øyvind Langsrud's JavaScript
/// implementation at [http://www.langsrud.com/fisher.htm](http://www.langsrud.com/fisher.htm).
///
/// # Examples
/// ```
/// use fishers_exact::fishers_exact;
///
/// let p = fishers_exact(&[1,9,11,3]).unwrap();
///
/// assert!((p.less_pvalue - 0.001346).abs() < 0.0001);
/// assert!((p.greater_pvalue - 0.9999663).abs() < 0.0001);
/// assert!((p.two_tail_pvalue - 0.0027594).abs() < 0.0001);
/// ```
///
pub fn fishers_exact(table: &[usize; 4], alternative: Alternative) -> f64 {

    // If both values in a row or column are zero, the p-value is 1 and
    // the odds ratio is NaN.
    if (table[0] == 0 && table[2] == 0) || (table[1] == 0 && table[3] == 0) {
        //return (f64::NAN, 1.0);
        return 1.0
    }

    let odds_ratio = {
        if table[1] > 0 && table[2] > 0 {
            (table[0] * table[3]) as f64 / (table[1] * table[2]) as f64
        } else {
            f64::INFINITY
        }
    };

    //let (n11, n12, n21, n22) = (table[0], table[1], table[2], table[3]);
    let n1_ = table[0] + table[1];
    let n_1 = table[0] + table[2];
    let sum = table.iter().sum(); //[0] + table[1] + table[2] +  table[3];
    let (_, sleft, sright, left, right) = exact(table[0], n1_, n_1, sum);

    let two_tail = {
        if (sleft + sright) < 1.0 {
            sleft + sright
        } else {
            1.0
        }
    };


    let values = FishersExactPvalues {
        two_tail_pvalue: two_tail,
        less_pvalue: left,
        greater_pvalue: right,
    };

    let value = {
        match alternative {
            Alternative::TwoSided => values.two_tail_pvalue,
            Alternative::Less => values.less_pvalue,
            Alternative::Greater => values.greater_pvalue
        }
    };
    value
}

#[cfg(test)]
mod tests {
    use super::fishers_exact;
    use crate::fisher::Alternative;

    fn fuzzy_eq(a: f64, b: f64) -> bool {
        (a - b).abs() < 0.000001
    }

    #[test]
    fn test_fishers_exact() {
        // 20 cases randomly generated via scipy.
        // ([a,b,c,d], less, greater, two-tail)
        let cases = [
            (
                [61, 118, 2, 1],
                Alternative::Less,
                0.27535061623455315,
                //0.9598172545684959,
                //0.27535061623455315,
            ),
            (
                [172, 46, 90, 127],
                Alternative::Less,
                1.0,
                //6.662405187351769e-16,
                //9.041009036528785e-16,
            ),
            (
                [127, 38, 112, 43],
                Alternative::Less,
                0.8637599357870167,
                //0.20040942958644145,
                //0.3687862842650179,
            ),
            (
                [186, 177, 111, 154],
                Alternative::Greater,
                //0.9918518696328176,
                0.012550663906725129,
                //0.023439141644624434,
            ),
            (
                [137, 49, 135, 183],
                Alternative::Greater,
                //0.999999999998533,
                5.6517533666400615e-12,
                //8.870999836202932e-12,
            ),
            (
                [37, 115, 37, 152],
                Alternative::Greater,
                //0.8834621182590621,
                0.17638403366123565,
                //0.29400927608021704,
            ),
            (
                [124, 117, 119, 175],
                Alternative::Greater,
                //0.9956704915461392,
                0.007134712391455461,
                //0.011588218284387445,
            ),
            (
                [70, 114, 41, 118],
                Alternative::Greater,
                //0.9945558498544903,
                0.010384865876586255,
                //0.020438291037108678,
            ),
            (
                [173, 21, 89, 7],
                Alternative::TwoSided,
                //0.2303739114068352,
                //0.8808002774812677,
                0.4027047267306024,
            ),
            (
                [18, 147, 123, 58],
                Alternative::TwoSided,
                //4.077820702304103e-29,
                //0.9999999999999817,
                7.686224774594537e-29,
            ),
            (
                [116, 20, 92, 186],
                Alternative::TwoSided,
                //0.9999999999998267,
                //6.598118571034892e-25,
                8.164831402188242e-25,
            ),
            (
                [9, 22, 44, 38],
                Alternative::TwoSided,
                //0.01584272038710196,
                //0.9951463496539362,
                0.021581786662999272,
            ),
            (
                [9, 101, 135, 7],
                Alternative::Greater,
                //3.3336213533847776e-50,
                1.0,
                //3.3336213533847776e-50,
            ),
            (
                [153, 27, 191, 144],
                Alternative::Less,
                0.9999999999950817,
                //2.473736787266208e-11,
                //3.185816623300107e-11,
            ),
            (
                [111, 195, 189, 69],
                Alternative::Greater,
                // 6.665245982898848e-19,
                0.9999999999994574,
                //1.0735744915712542e-18,
            ),
            (
                [125, 21, 31, 131],
                Alternative::Greater,
                //0.99999999999974,
                9.720661317939016e-34,
                //1.0352129312860277e-33,
            ),
            (
                [201, 192, 69, 179],
                Alternative::Less,
                0.9999999988714893,
                // 3.1477232259550017e-09,
                // 4.761075937088169e-09,
            ),
            // (
            //     [167, 184, 141, 28],
            //     7.045789653297585e-16,
            //     1.0,
            //     9.362858503272341e-16,
            // ),
            // (
            //     [194, 74, 141, 182],
            //     0.9999999999999848,
            //     1.2268868025030845e-12,
            //     1.8076995960009742e-12,
            // ),
            (
                [124, 138, 159, 160],
                Alternative::Greater,
                //0.30153826772785475,
                0.7538974235759873,
                //0.5601766196310243,
            ),
        ];

        for &(table, alternative, expected) in cases.iter() {
            let p = fishers_exact(&table, alternative);
            assert!(fuzzy_eq(p, expected));
        }
    }
}
