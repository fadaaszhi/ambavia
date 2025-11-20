use std::time::Instant;

use glam::DVec2;

use crate::utility::mix;

fn sd_segment_squared(p: DVec2, a: DVec2, b: DVec2) -> (f64, f64) {
    let ap = p - a;
    let ab = b - a;
    let t = (ap.dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
    (ap.distance_squared(ab * t), t)
}

fn any_part_of_segment_in_aabb(p0: DVec2, p1: DVec2, a: DVec2, b: DVec2) -> bool {
    let ta = (a - p0) / (p1 - p0);
    let tb = (b - p0) / (p1 - p0);
    let t1 = ta.min(tb);
    let t2 = ta.max(tb);
    let (t1, t2) = (t1.x.max(t1.y), t2.x.min(t2.y));
    t1 < t2 && t1 < 1.0 && 0.0 < t2
}

struct Sampler<F> {
    f: F,
    vp_min: DVec2,
    vp_max: DVec2,
    tolerance_squared: f64,
    points: Vec<DVec2>,
}

impl<F: FnMut(f64) -> DVec2> Sampler<F> {
    /// Expects `p0` to have already been added to the point list, but not `p1`.
    fn subdivide(
        &mut self,
        t0: f64,
        t1: f64,
        p0: DVec2,
        p1: DVec2,
        mut depth: usize,
        // Number of times to recurse when trying to find where a function starts being defined again
        mut non_finite_depth: usize,
    ) {
        // If both are infinite or NaN then don't draw anything
        if !p0.is_finite() && !p1.is_finite() {
            return;
        }

        if p0 == p1 {
            self.points.push(p1);
            return;
        }

        if p0.is_finite() && p1.is_finite() {
            let outside_aabb = !any_part_of_segment_in_aabb(p0, p1, self.vp_min, self.vp_max);
            if outside_aabb {
                self.points.push(p1);
                return;
            }

            depth -= 1;
        } else {
            non_finite_depth -= 1;
        }

        if depth == 0 || non_finite_depth == 0 {
            self.points.push(DVec2::NAN);
            if p1.is_finite() {
                self.points.push(p1);
            }
            return;
        }

        let t = t0.midpoint(t1);
        let p = (self.f)(t);

        if p0.is_finite() && p.is_finite() && p1.is_finite() {
            let (d2, u) = sd_segment_squared(p, p0, p1);
            let u_bound = 0.1;

            if d2 < self.tolerance_squared && u_bound < u && u < 1.0 - u_bound {
                self.points.push(p);
                self.points.push(p1);
                return;
            }
        }

        self.subdivide(t0, t, p0, p, depth, non_finite_depth);
        self.subdivide(t, t1, p, p1, depth, non_finite_depth);
    }
}

pub fn sample_function(
    mut f: impl FnMut(f64) -> DVec2,
    t_min: f64,
    t_max: f64,
    vp_min: DVec2,
    vp_max: DVec2,
    tolerance: f64,
    uniform_sample_count: usize,
) -> Vec<DVec2> {
    let start = Instant::now();
    let half_uniform_sample_count = uniform_sample_count / 2;
    let mut f_eval_count = 0;
    let mut f = |x: f64| {
        f_eval_count += 1;
        f(x)
    };
    let p = f(t_min);
    let mut s = Sampler {
        f,
        vp_min,
        vp_max,
        tolerance_squared: tolerance.powi(2),
        points: vec![if p.is_finite() { p } else { DVec2::NAN }],
    };

    for i in 0..half_uniform_sample_count {
        let t0 = mix(t_min, t_max, i as f64 / half_uniform_sample_count as f64);
        let t1 = mix(
            t_min,
            t_max,
            (i + 1) as f64 / half_uniform_sample_count as f64,
        );
        let p0 = *s.points.last().unwrap();
        let p1 = (s.f)(t1);
        s.subdivide(t0, t1, p0, p1, 10, 20);
    }

    let points = s.points;
    let elapsed = start.elapsed();
    println!();
    println!("points.len() = {}", points.len());
    println!("f eval count = {}", f_eval_count);
    println!("  time taken = {:?}", elapsed);

    if points.len() < 10 {
        dbg!(&points);
    }
    points
}
