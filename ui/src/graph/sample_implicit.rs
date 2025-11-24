use glam::{DVec2, UVec2, dmat2, uvec2};

pub fn sample_implicit(
    mut f: impl FnMut(DVec2) -> f64,
    vp_min: DVec2,
    vp_max: DVec2,
) -> Vec<DVec2> {
    let n = UVec2::splat(100);
    let mut points = vec![];
    let draw_line = |a: DVec2, b: DVec2| {
        if points.last() != Some(&a) {
            if let Some(p) = points.last()
                && p.is_finite()
            {
                points.push(DVec2::NAN);
            }
            points.push(a);
        }
        points.push(b);
    };
    let f = |p| f(p) < 0.0;
    triangle_odc(f, vp_min, vp_max, n, draw_line);
    // marching_triangles(f, vp_min, vp_max, n, draw_line);
    points
}

fn binary_search(
    f: &mut impl FnMut(DVec2) -> bool,
    f0: bool,
    mut p0: DVec2,
    mut p1: DVec2,
    iters: u32,
) -> DVec2 {
    assert_ne!(f0, f(p1));
    for _ in 0..iters {
        let m = (p0 + p1) / 2.0;
        if f0 == f(m) {
            p0 = m;
        } else {
            p1 = m;
        }
    }
    (p0 + p1) / 2.0
}

#[allow(unused)]
fn marching_triangles<F: FnMut(DVec2) -> bool>(
    mut f: F,
    b0: DVec2,
    b1: DVec2,
    n: UVec2,
    mut draw_line: impl FnMut(DVec2, DVec2),
) {
    let p = |x: u32, y: u32| uvec2(x, y).as_dvec2() / n.as_dvec2() * (b1 - b0) + b0;
    for x in 0..n.x {
        for y in 0..n.y {
            let p00 = p(x, y);
            let p01 = p(x, y + 1);
            let p10 = p(x + 1, y);
            let p11 = p(x + 1, y + 1);
            for p in [[p00, p01, p11], [p11, p10, p00]] {
                let b = |f: &mut F, i: usize, j: usize| {
                    let a = f(p[i]);
                    binary_search(f, a, p[i], p[j], 15)
                };
                let a = p.map(&mut f);

                match (a[0] == a[1], a[0] == a[2]) {
                    (true, true) => {}
                    (true, false) => draw_line(b(&mut f, 1, 2), b(&mut f, 2, 0)),
                    (false, true) => draw_line(b(&mut f, 0, 1), b(&mut f, 1, 2)),
                    (false, false) => draw_line(b(&mut f, 2, 0), b(&mut f, 0, 1)),
                }
            }
        }
    }
}

fn line_binary_search(
    f: &mut impl FnMut(DVec2) -> bool,
    f0: bool,
    p_is_intersection: bool,
    mut p: DVec2,
    d: DVec2,
    line_iters: u32,
    binary_iters: u32,
) -> Option<DVec2> {
    for i in 0..line_iters {
        if f(p + d) != f0 {
            return Some(if i == 0 && p_is_intersection {
                p
            } else {
                binary_search(f, f0, p, p + d, binary_iters)
            });
        }
        p += d;
    }
    None
}

fn intersect_lines(p0: DVec2, d0: DVec2, p1: DVec2, d1: DVec2) -> DVec2 {
    p0 + d0 * (dmat2(d0, -d1).inverse() * (p1 - p0)).x
}

fn is_parallel(a: DVec2, b: DVec2, cos_theta: f64) -> bool {
    a.normalize().dot(b.normalize()).abs() > cos_theta
}

fn point_in_triangle(p: DVec2, v0: DVec2, v1: DVec2, v2: DVec2) -> bool {
    // trial-n-error'd signs
    (p - v0).perp_dot(v1 - v0) <= 0.0
        && (p - v1).perp_dot(v2 - v1) <= 0.0
        && (p - v2).perp_dot(v0 - v2) <= 0.0
}

#[allow(unused)]
fn triangle_odc(
    mut f: impl FnMut(DVec2) -> bool,
    b0: DVec2,
    b1: DVec2,
    n: UVec2,
    mut draw_line: impl FnMut(DVec2, DVec2),
) {
    let p = |x: u32, y: u32| uvec2(x, y).as_dvec2() / n.as_dvec2() * (b1 - b0) + b0;
    for x in 0..n.x {
        for y in 0..n.y {
            let p00 = p(x, y);
            let p01 = p(x, y + 1);
            let p10 = p(x + 1, y);
            let p11 = p(x + 1, y + 1);
            for (p0, p1, p2) in [(p00, p01, p11), (p11, p10, p00)] {
                // draw_line((p0, p1));
                // draw_line((p1, p2));
                // draw_line((p2, p0));
                // continue;
                let (f0, f1, f2) = (f(p0), f(p1), f(p2));
                let (f0, p0, p1, p2) = match (f0 == f1, f0 == f2) {
                    (true, true) => continue,
                    (true, false) => (f2, p2, p0, p1),
                    (false, true) => (f1, p1, p2, p0),
                    (false, false) => (f0, p0, p1, p2),
                };
                let line_iters = 10;
                let binary_iters = 7;
                let pe1 = binary_search(&mut f, f0, p0, p1, binary_iters);
                let pe2 = binary_search(&mut f, f0, p0, p2, binary_iters);
                let m = (pe1 + pe2) / 2.0;
                let l = (pe2 - pe1).normalize() * p00.distance(p11) / line_iters as f64;
                let d = l.perp();
                let fm = f(m);
                let d = if f0 == fm { 1.0 } else { -1.0 } * d.dot(m - p0).signum() * d;
                let mut qf = 'qf: {
                    let Some(q) =
                        line_binary_search(&mut f, fm, false, m, d, line_iters, binary_iters)
                    else {
                        break 'qf m;
                    };
                    if is_parallel(q - pe1, q - pe2, 0.9999) {
                        break 'qf q;
                    }
                    // points.push(q);
                    let (Some(q1), Some(q2)) = (
                        line_binary_search(&mut f, fm, true, q, -l, line_iters, binary_iters),
                        line_binary_search(&mut f, fm, true, q, l, line_iters, binary_iters),
                    ) else {
                        break 'qf q;
                    };

                    let d1 = q1 - pe1;
                    let d2 = q2 - pe2;
                    if is_parallel(d1, d2, 0.9999) {
                        break 'qf m;
                    }
                    let qf = intersect_lines(pe1, d1, pe2, d2);
                    assert!(qf.is_finite());
                    qf
                };
                if point_in_triangle(qf, p0, p1, p2) {
                    qf = m;
                }
                draw_line(pe1, qf);
                draw_line(qf, pe2);
            }
        }
    }
}
