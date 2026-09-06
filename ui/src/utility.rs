use glam::{DVec2, UVec2, dvec2, uvec2};
use winit::dpi::{PhysicalPosition, PhysicalSize};

use crate::ui::Bounds;

pub trait AsGlam {
    type G;
    fn as_glam(&self) -> Self::G;
}

impl AsGlam for PhysicalPosition<f64> {
    type G = DVec2;
    fn as_glam(&self) -> DVec2 {
        dvec2(self.x, self.y)
    }
}

impl AsGlam for PhysicalSize<u32> {
    type G = UVec2;
    fn as_glam(&self) -> UVec2 {
        uvec2(self.width, self.height)
    }
}

pub fn flip_y(v: DVec2) -> DVec2 {
    dvec2(v.x, -v.y)
}

pub trait Mix<I> {
    fn mix(self, other: Self, t: I) -> Self;
}

pub fn mix<T: Mix<I>, I>(x: T, y: T, t: I) -> T {
    x.mix(y, t)
}

impl Mix<f64> for f64 {
    fn mix(self, other: Self, t: f64) -> Self {
        self * (1.0 - t) + other * t
    }
}

impl Mix<f64> for DVec2 {
    fn mix(self, other: Self, t: f64) -> Self {
        self.lerp(other, t)
    }
}

impl Mix<DVec2> for DVec2 {
    fn mix(self, other: Self, t: DVec2) -> Self {
        dvec2(mix(self.x, other.x, t.x), mix(self.y, other.y, t.y))
    }
}

pub fn unmix(t: f64, x: f64, y: f64) -> f64 {
    let z = (t - x) / (y - x);
    if z.is_finite() { z } else { 0.0 }
}

pub fn snap(x: f64, w: u32) -> f64 {
    let a = 0.5 * (w % 2) as f64;
    (x - a).round() + a
}

pub fn max(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(-f64::INFINITY, f64::max)
}

pub fn union(bounds: impl IntoIterator<Item = Bounds>) -> Bounds {
    let mut bounds = bounds.into_iter();
    let first = bounds.next().expect("need at least one bound to union");
    bounds.fold(first, Bounds::union)
}

/// If the provided values differ then this sets the first argument to the second
/// argument and returns `true`, otherwise it returns `false`.
pub fn set<T, U>(dst: &mut T, src: U) -> bool
where
    T: PartialEq<U> + From<U>,
{
    if *dst != src {
        *dst = src.into();
        true
    } else {
        false
    }
}
