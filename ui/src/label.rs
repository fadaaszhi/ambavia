use glam::{DVec2, dvec2};

use crate::{
    katex_font::{Font, get_glyph},
    ui::{Color, Quad, QuadKind},
};

pub fn label(
    text: &str,
    position: DVec2,
    scale: f64,
    color: impl Color,
    font: Font,
    draw_quad: &mut impl FnMut(Quad),
) {
    let color = color.to_rgbaf64();
    let mut cursor = position;
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
