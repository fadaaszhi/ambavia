use glam::{DVec2, dvec2};

use crate::{
    katex_font::{Font, get_glyph},
    ui::{Bounds, Color, Quad, QuadKind},
};

pub struct Label<'a> {
    text: &'a str,
    pub scale: f64,
    font: Font,
    raw_bounds: Bounds,
}

impl<'a> Label<'a> {
    pub fn new(text: &'a str, scale: f64, font: Font) -> Label<'a> {
        let mut x = 0.0;
        let mut min = DVec2::INFINITY;
        let mut max = -DVec2::INFINITY;

        for c in text.chars() {
            let glyph = get_glyph(font, c);
            min = min.min(dvec2(x + glyph.plane.left, glyph.plane.top));
            max = max.max(dvec2(x + glyph.plane.right, glyph.plane.bottom));
            x += glyph.advance;
        }

        let size = (max - min).max(DVec2::ZERO);
        let pos = if size.x == 0.0 || size.y == 0.0 {
            DVec2::ZERO
        } else {
            min
        };

        Label {
            text,
            scale,
            font,
            raw_bounds: Bounds { pos, size },
        }
    }

    pub fn size(&self) -> DVec2 {
        self.raw_bounds.size * self.scale
    }

    pub fn render_from_cursor(
        &self,
        cursor: DVec2,
        color: impl Color,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        render_label(self.text, cursor, self.scale, color, self.font, draw_quad);
    }

    pub fn render_from_top_left(
        &self,
        top_left: DVec2,
        color: impl Color,
        draw_quad: &mut impl FnMut(Quad),
    ) {
        self.render_from_cursor(
            top_left - self.raw_bounds.pos * self.scale,
            color,
            draw_quad,
        );
    }
}

pub fn render_label(
    text: &str,
    mut cursor: DVec2,
    scale: f64,
    color: impl Color,
    font: Font,
    draw_quad: &mut impl FnMut(Quad),
) {
    let color = color.to_rgbaf64();
    for c in text.chars() {
        let glyph = get_glyph(font, c);
        draw_quad(Quad {
            kind: QuadKind::MsdfGlyph,
            p0: cursor + dvec2(glyph.plane.left, glyph.plane.top) * scale,
            p1: cursor + dvec2(glyph.plane.right, glyph.plane.bottom) * scale,
            uv0: dvec2(glyph.atlas.left, glyph.atlas.top),
            uv1: dvec2(glyph.atlas.right, glyph.atlas.bottom),
            color,
        });
        cursor.x += glyph.advance * scale;
    }
}
