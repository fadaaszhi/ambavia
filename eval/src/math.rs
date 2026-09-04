pub fn sort_perm(list: &[f64]) -> Vec<usize> {
    let mut indices = (0..list.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| list[*a].total_cmp(&list[*b]));
    indices
}

pub fn hypot3(x: f64, y: f64, z: f64) -> f64 {
    let max = x.abs().max(y.abs()).max(z.abs());

    if max == 0.0 {
        return if x.is_nan() || y.is_nan() || z.is_nan() {
            f64::NAN
        } else {
            0.0
        };
    }

    max * ((x / max).powi(2) + (y / max).powi(2) + (z / max).powi(2)).sqrt()
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Rational {
    num: f64,
    den: f64,
}

impl Rational {
    const MAX_VALUE: f64 = (1u64 << 53) as f64 - 1.0;

    fn exact(x: f64) -> Option<Rational> {
        const LIMIT: f64 = 10u64.pow(12) as f64;

        if x.abs() > LIMIT {
            return None;
        }

        let max_denominator = (LIMIT / x.abs()).sqrt().round().min(LIMIT);
        Rational::best_approximation(x, max_denominator).filter(|y| y.num / y.den == x)
    }

    // https://axotron.se/blog/fast-algorithm-for-rational-approximation-of-floating-point-numbers/
    fn best_approximation(x: f64, max_denominator: f64) -> Option<Rational> {
        debug_assert_eq!(
            max_denominator,
            max_denominator.floor().clamp(1.0, Self::MAX_VALUE)
        );

        if x.is_nan() || x.abs() > Self::MAX_VALUE {
            return None;
        }

        let mut a = Rational { num: 0.0, den: 1.0 };
        let mut b = Rational { num: 1.0, den: 1.0 };

        let sign = x.signum();
        let x = x.abs();
        let integer_part = x.floor();
        let x = x.fract();

        loop {
            let e = x.mul_add(a.den, -a.num).max(0.0);
            let f = x.mul_add(-b.den, b.num).max(0.0);

            let go = |a: Rational, b: &mut Rational, e: f64, f: f64| {
                // It's ok if e = 0 and causes k_skip = inf because we min() afterwards
                let k_skip = (f / e).floor();
                // Prevent b.d + k * a.d from going above max_denominator
                let k_max = ((max_denominator - b.den) / a.den).floor();
                let k = k_skip.min(k_max);
                b.num += k * a.num;
                b.den += k * a.den;
                k_skip >= k_max
            };

            if if f > e {
                go(a, &mut b, e, f)
            } else {
                go(b, &mut a, f, e)
            } {
                break;
            }
        }

        assert!(a.den <= max_denominator && b.den <= max_denominator);

        // As a tie breaker prefer the one with the smaller denominator
        let mut y = if (x - a.num / a.den, a.den) < (b.num / b.den - x, b.den) {
            a
        } else {
            b
        };
        y.num = sign * (integer_part * y.den + y.num);
        (y.num < Self::MAX_VALUE).then_some(y)
    }
}

/// Returns `round((value - offset) / step) * step + offset` computed with rational arithmetic.
pub fn apply_slider_step(value: f64, offset: f64, step: f64, round: fn(f64) -> f64) -> f64 {
    let step = step.abs();
    let step_rat = Rational::exact(step);
    let value_rat = Rational::exact(value);
    let offset_rat = Rational::exact(offset);

    let a = round(match (value_rat, offset_rat, step_rat) {
        (Some(value), Some(offset), Some(step)) => {
            ((value.num * offset.den - offset.num * value.den) * step.den)
                / (value.den * offset.den * step.num)
        }
        (Some(value), Some(offset), None) => {
            (value.num * offset.den - offset.num * value.den) / (value.den * offset.den * step)
        }
        (_, _, Some(step)) => (value - offset) * step.den / step.num,
        _ => (value - offset) / step,
    });

    let result = match (step_rat, offset_rat) {
        (Some(step), Some(offset)) => {
            (a * step.num * offset.den + offset.num * step.den) / (step.den * offset.den)
        }
        (Some(step), None) => a * step.num / step.den + offset,
        _ => a * step + offset,
    };

    if result.is_finite() {
        result
    } else {
        let result = a * step + offset;
        if result.is_finite() { result } else { value }
    }
}

pub fn apply_slider(mut value: f64, min: f64, max: f64, step: f64) -> f64 {
    if max.is_finite() && value >= max {
        if min.is_finite() {
            return max.max(min);
        }
        return max;
    }

    if step.is_finite() && step != 0.0 {
        let offset = if min.is_finite() { min } else { 0.0 };
        value = apply_slider_step(value, offset, step, f64::round);
    }

    if max.is_finite() {
        value = f64::min(value, max);
    }

    if min.is_finite() {
        value = f64::max(value, min);
    }

    value
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    trait IntoOptionRational {
        fn into_option_rational(self) -> Option<Rational>;
    }

    impl IntoOptionRational for (i64, u64) {
        fn into_option_rational(self) -> Option<Rational> {
            assert!((-Rational::MAX_VALUE as i64..=Rational::MAX_VALUE as i64).contains(&self.0));
            assert!((0..=Rational::MAX_VALUE as u64).contains(&self.1));
            Some(Rational {
                num: self.0 as f64,
                den: self.1 as f64,
            })
        }
    }

    impl IntoOptionRational for (i64, f64) {
        fn into_option_rational(self) -> Option<Rational> {
            assert!((-Rational::MAX_VALUE as i64..=Rational::MAX_VALUE as i64).contains(&self.0));
            assert!((0.0..=Rational::MAX_VALUE).contains(&self.1));
            assert_eq!(self.1.fract(), 0.0);
            Some(Rational {
                num: self.0 as f64,
                den: self.1,
            })
        }
    }

    impl IntoOptionRational for Option<Rational> {
        fn into_option_rational(self) -> Option<Rational> {
            self
        }
    }

    #[rstest]
    #[case(0.0, (0, 1))]
    #[case(-0.0, (0, 1))]
    #[case(1.0, (1, 1))]
    #[case(-1.0, (-1, 1))]
    #[case(f64::INFINITY, None)]
    #[case(-f64::INFINITY, None)]
    #[case(f64::NAN, None)]
    #[case(0.1, (1, 10))]
    #[case(0.2, (1, 5))]
    #[case(0.3, (3, 10))]
    #[case(1.9, (19, 10))]
    #[case(1e-12, (1, 1e12))]
    #[case(1e-13, None)]
    #[case(7550.927693761815, (15977763, 2116))]
    #[case(0.5, (1, 2))]
    #[case(1.0 / 3001.0, (1, 3001))]
    #[case(17.0 / 65536.0, (17, 65536))]
    #[case(0.288, (36, 125))]
    fn rational_eq(#[case] value: f64, #[case] expected: impl IntoOptionRational) {
        let expected = expected.into_option_rational();
        print!("testing {value} ?= ");
        match expected {
            Some(Rational { num, den }) => println!("{num}/{den}"),
            None => println!("None"),
        };
        assert_eq!(Rational::exact(value), expected);
    }

    #[rstest]
    #[case(1.9, None, None, 0.1, 1.9)]
    fn slider(
        #[case] value: f64,
        #[case] min: impl Into<Option<f64>>,
        #[case] max: impl Into<Option<f64>>,
        #[case] step: impl Into<Option<f64>>,
        #[case] expected: f64,
    ) {
        let (min, max, step) = (min.into(), max.into(), step.into());
        println!("value = {value}");
        println!("min = {min:?}");
        println!("max = {max:?}");
        println!("step = {step:?}");

        assert_eq!(
            apply_slider(
                value,
                min.unwrap_or(f64::NAN),
                max.unwrap_or(f64::NAN),
                step.unwrap_or(f64::NAN)
            ),
            expected
        );
    }
}
