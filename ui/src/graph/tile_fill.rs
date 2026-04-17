use glam::{DVec2, U8Vec2, Vec2, Vec2Swizzles, dvec2, ivec2, u8vec2, uvec2};
use rayon::prelude::*;

use crate::{timer::Timer, ui::Bounds, utility::mix};

pub const TILE_SIZE: u32 = 16;

pub enum Item {
    Rectangle { width: f32 },
    Tile(Tile),
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct Tile {
    segments_start: u32,
    segments_end: u32,
    winding_number: i32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub struct Segment {
    a: U8Vec2,
    b: U8Vec2,
}

fn edges(path: &[DVec2]) -> impl Iterator<Item = (DVec2, DVec2)> {
    (0..if path.len() < 3 { 0 } else { path.len() }).map(|i| (path[i], path[(i + 1) % path.len()]))
}

// References:
// 1. "Random-Access Rendering of General Vector Graphics" by Diego Nehab & Hugues Hoppe
//     https://hhoppe.com/ravg.pdf
// 2. "A sort-middle architecture for 2D graphics" by Raph Levien
//     https://raphlinus.github.io/rust/graphics/gpu/2020/06/12/sort-middle.html
pub fn tile_fill(
    bounds: Bounds,
    vertices: &[DVec2],
    segments: &mut Vec<Segment>,
    mut emit: impl FnMut(Vec2, Item),
) {
    let mut timer = Timer::default();
    let mut tile_fill_timer = timer.start("tile_fill");

    let bmin = (bounds.pos / TILE_SIZE as f64).floor();
    let bmax = ((bounds.pos + bounds.size) / TILE_SIZE as f64).ceil();
    let bsize = bmax - bmin;
    let n_tiles = bsize.as_u16vec2();

    // 0. Find subpolygons separated by non-finite points
    // TODO do this via non-allocating iterators

    let polygons_timer = tile_fill_timer.start("subpolygons");
    let mut polygons = vec![];
    let mut current = vec![];

    for v in vertices {
        if v.is_finite() {
            current.push(*v);
        } else if !current.is_empty() {
            polygons.push(std::mem::take(&mut current));
        }
    }

    if !current.is_empty() {
        if vertices.first().is_some_and(|v| v.is_finite()) && !polygons.is_empty() {
            polygons[0].extend(current);
        } else {
            polygons.push(current);
        }
    }
    drop(polygons_timer);

    // 1. Find the tiles pierced by each polygon edge (coarse rasterization)

    struct TileSegment {
        y: u16,
        /// x=0 is used as a sentinel for backdrop tiles. Actual tiles start at 1
        x: u16,
        /// Normalized segment end points, where (0,0) is at the top-left of the
        /// tile and (255,255) is at the bottom-right.
        p0: U8Vec2,
        p1: U8Vec2,
    }

    let mut tile_segments = vec![];

    let dda_timer = tile_fill_timer.start("dda");
    for (a, b) in polygons.iter().flat_map(|vs| edges(vs)) {
        assert!(a.is_finite() && b.is_finite());

        // Transform coordinates so that tiles are unit length and the top-left
        // of the screen is at the origin
        let a = a / TILE_SIZE as f64 - bmin;
        let b = b / TILE_SIZE as f64 - bmin;

        let split = split_segment_at_x(a, b, 0.0);

        // Compute backdrop from part of segment left of screen
        if let Some((a, b)) = split.left {
            let min_y =
                (((a.y.min(b.y) * 255.0).round() / 255.0).clamp(0.0, bsize.y)).ceil() as u16;
            let max_y =
                (((a.y.max(b.y) * 255.0).round() / 255.0).clamp(0.0, bsize.y)).ceil() as u16;

            let (p0, p1) = if a.y < b.y {
                (u8vec2(1, 0), u8vec2(1, 1))
            } else {
                (u8vec2(1, 1), u8vec2(1, 0))
            };

            for y in min_y..max_y {
                tile_segments.push(TileSegment { y, x: 0, p0, p1 });
            }
        }

        // Clip segment to screen
        let Some((a, b)) = split
            .right
            .and_then(|(a, b)| clip_segment(a, b, DVec2::ZERO, bsize))
        else {
            continue;
        };

        if a == b {
            continue;
        }

        // Initialize DDA traversal
        let init_dda_axis = |a: f64, b: f64| {
            let (first, last, step, t_next, t_delta);

            if a < b {
                first = a.floor();
                last = b.ceil() - 1.0;
                step = 1;
                t_next = (first + 1.0 - a) / (b - a);
                t_delta = 1.0 / (b - a);
            } else {
                first = a.ceil() - 1.0;
                last = b.floor();
                step = -1;
                t_next = if a == b {
                    f64::INFINITY
                } else {
                    (first - a) / (b - a)
                };
                t_delta = 1.0 / (a - b);
            }

            (first as i32, last as i32, step, t_next, t_delta)
        };

        let (mut x, last_x, step_x, mut t_next_x, t_delta_x) = init_dda_axis(a.x, b.x);
        let (mut y, last_y, step_y, mut t_next_y, t_delta_y) = init_dda_axis(a.y, b.y);

        debug_assert!(0 <= x && x < n_tiles.x as i32);
        debug_assert!(0 <= y && y < n_tiles.y as i32);
        debug_assert!(0 <= last_x && last_x < n_tiles.x as i32);
        debug_assert!(0 <= last_y && last_y < n_tiles.y as i32);

        // The start point of the segment within the current tile
        let mut p0 = (a * 255.0).round();

        loop {
            debug_assert!(0 <= x && x < n_tiles.x as i32);
            debug_assert!(0 <= y && y < n_tiles.y as i32);

            // Current tile
            let x0 = x;
            let y0 = y;

            // The end t-value for the section of segment within the current tile
            let t1;

            // Step to next tile
            if t_next_x < t_next_y {
                if x == last_x {
                    t1 = 1.0;
                } else {
                    t1 = t_next_x;
                    t_next_x += t_delta_x;
                    x += step_x;
                }
            } else {
                if y == last_y {
                    t1 = 1.0;
                } else {
                    t1 = t_next_y;
                    t_next_y += t_delta_y;
                    y += step_y;
                }
            }

            // The end point of the segment within the current tile
            let p1 = (mix(a, b, t1) * 255.0).round();

            // Make end points relative to tile origin
            let tile_origin = (ivec2(x0, y0) * 255).as_dvec2();
            let q0 = p0 - tile_origin;
            let q1 = p1 - tile_origin;

            debug_assert!(0.0 <= q0.x && q0.x <= 255.0);
            debug_assert!(0.0 <= q0.y && q0.y <= 255.0);
            debug_assert!(0.0 <= q1.x && q1.x <= 255.0);
            debug_assert!(0.0 <= q1.y && q1.y <= 255.0);

            // Ignore segments that got rounded to nothing
            if q0 != q1 {
                // Output the section of the segment within the current tile
                tile_segments.push(TileSegment {
                    y: y0 as u16,
                    // +1 because x=0 is used for sentinel for offscreen segments
                    x: x0 as u16 + 1,
                    p0: q0.as_u8vec2(),
                    p1: q1.as_u8vec2(),
                });
            }

            if t1 >= 1.0 {
                // Segment is finished
                break;
            }

            // The segment's end point in the current tile will be the start
            // point in the next tile
            p0 = p1;
        }
    }
    drop(dda_timer);

    // 2. Sort the tiles to be in row-major order

    // Threshold determined empirically on my MacBook
    // Under this threshold, par_sort got about 2x slower
    if tile_segments.len() < 12000 {
        tile_fill_timer.time("sort", || {
            tile_segments.sort_unstable_by_key(|f| (f.y, f.x))
        });
    } else {
        tile_fill_timer.time("par_sort", || {
            tile_segments.par_sort_unstable_by_key(|f| (f.y, f.x))
        });
    }

    // 3. Iterate over each row of tiles to:
    //   - accumulate backdrop winding number from tiles to the left
    //   - collect segments that fall within each tile
    //   - create solid fill rectangle in the gaps between tiles

    let to_physical = |x: u16, y: u16| {
        debug_assert_ne!(x, 0);
        (uvec2((x - 1) as _, y as _) * TILE_SIZE).as_vec2() + (bmin * TILE_SIZE as f64).as_vec2()
    };

    // Keep track of average segments per tile for fun
    let initial_segments_len = segments.len();
    let mut tile_count = 0;

    // Index into tile_segments
    let mut i = 0;
    // Current row being processed
    let mut y = u16::MAX;
    // Winding number from backdrops to the left
    let mut row_winding_number = 0;

    let scan_timer = tile_fill_timer.start("scan");
    while i < tile_segments.len() {
        // Check if we've entered a new row
        if tile_segments[i].y != y {
            y = tile_segments[i].y;
            row_winding_number = 0;
        }

        let x = tile_segments[i].x;
        let winding_number = row_winding_number;

        // Collect all the segements in the current tile
        let segments_start = segments.len() as u32;

        while i < tile_segments.len() && tile_segments[i].y == y && tile_segments[i].x == x {
            let f = &tile_segments[i];
            row_winding_number += (f.p0.y == 0) as i32 - (f.p1.y == 0) as i32;
            i += 1;

            if x != 0 {
                segments.push(Segment { a: f.p0, b: f.p1 });
            }
        }

        // Emit tile
        let segments_end = segments.len() as u32;
        if segments_start != segments_end {
            let tile = Item::Tile(Tile {
                segments_start,
                segments_end,
                winding_number,
            });
            emit(to_physical(x, y), tile);
            tile_count += 1;
        }

        // Create filled rectangle after tile if non-zero backdrop
        if row_winding_number != 0 {
            // Only needed if next tile is not right after this one
            let x = x + 1;
            let next_x = if i < tile_segments.len() && tile_segments[i].y == y {
                tile_segments[i].x
            } else {
                n_tiles.x + 1
            };
            if x != next_x {
                let width = ((next_x - x) as u32 * TILE_SIZE) as f32;
                emit(to_physical(x, y), Item::Rectangle { width });
            }
        }
    }
    drop(scan_timer);

    if false {
        println!(
            "average segments per tile: {:.2}",
            (segments.len() - initial_segments_len) as f64 / tile_count as f64
        );
    }

    drop(tile_fill_timer);
    // println!("{}", timer.string());
}
pub struct SplitSegment {
    left: Option<(DVec2, DVec2)>,
    right: Option<(DVec2, DVec2)>,
}
impl SplitSegment {
    fn new(e1: Option<(DVec2, DVec2)>, e2: Option<(DVec2, DVec2)>, e1_left: bool) -> Self {
        match e1_left {
            true => Self {
                left: e1,
                right: e2,
            },
            false => Self {
                left: e2,
                right: e1,
            },
        }
    }
}
/// Split a line segment at the given x-value and return the parts to the left
/// and right of it, or `None` if the segment doesn't pass through that side.
/// Preserves the orientation of the segment.
fn split_segment_at_x(a: DVec2, b: DVec2, x: f64) -> SplitSegment {
    let t = (x - a.x) / (b.x - a.x);
    let (e1, e2);
    if 0.0 < t && t < 1.0 {
        let m = dvec2(x, mix(a.y, b.y, t));
        e1 = Some((a, m));
        e2 = Some((m, b));
    } else {
        e1 = Some((a, b));
        e2 = None;
    }
    SplitSegment::new(e1, e2, a.x < x)
}

/// Clip the line segment from `a` to `b` against the AABB from `min` to `max`
fn clip_segment(a: DVec2, b: DVec2, min: DVec2, max: DVec2) -> Option<(DVec2, DVec2)> {
    let f = |a, b, min, max| {
        split_segment_at_x(a, b, min)
            .right
            .and_then(|(a, b)| split_segment_at_x(a, b, max).left)
            .map(|(a, b)| (a.yx(), b.yx()))
    };
    f(a, b, min.x, max.x).and_then(|(a, b)| f(a, b, min.y, max.y))
}
