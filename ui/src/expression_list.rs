use std::collections::HashMap;
use std::fmt::Write;

use bytemuck::{Zeroable, offset_of};
use derive_more::{Add, From, Into, Sub};
use glam::{DVec2, U16Vec2, Vec2, dvec2, u16vec2, uvec2, vec2};
use typed_index_collections::{TiVec, ti_vec};
use winit::{
    event::{ElementState, MouseButton},
    window::CursorIcon,
};

use crate::{
    AppGraphics,
    graph::{Geometry, GeometryKind},
    math_field::{Cursor, Interactiveness, MathField, Message, UserSelection},
    ui::{Bounds, Context, CursorMode, Event, QuadKind, Response},
    utility::{mix, unmix},
};
use eval::{
    compiler::compile_assignments,
    vm::{self, Vm, apply_slider, apply_slider_step},
};
use parse::{
    analyze_expression_list::{ExpressionResult, PlotKind, analyze_expression_list},
    ast,
    ast_parser::{parse_standalone_expression, parse_statement},
    latex_parser::parse_latex,
    latex_tree::{self, Bracket},
    name_resolver::{Domain, ExpressionIndex, ExpressionListEntry, Slider as NrSlider},
    type_checker::Type,
};

#[derive(Default)]
struct Output {
    ui: OutputUi,
    data: OutputData,
}

// TODO rename this since its used for sliders as well
#[derive(Debug, Default, Clone, Copy, PartialEq)]
enum DomainFocusState {
    #[default]
    None,
    Hovered,
    Focussed,
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
struct DomainState {
    state: DomainFocusState,
    error: bool,
}

#[derive(Default)]
enum OutputUi {
    #[default]
    None,
    Slider {
        value: f64,
        min: f64,
        max: f64,
        step: f64,
        dragging: Option<f64>,
        hovered: bool,
        name: String,
        name_field: MathField,
        step_label_field: MathField,
        min_state: DomainState,
        max_state: DomainState,
        step_state: DomainState,
    },
    Field(MathField),
    Domain {
        name: String,
        name_field: MathField,
        min_state: DomainState,
        max_state: DomainState,
    },
}

/// Creates a MathField initialized with ≤name≤
fn create_le_name_le(name: &str) -> MathField {
    use latex_tree::Node;

    let mut latex = vec![];
    latex.push(Node::CtrlSeq("le"));
    let mut parts = name.split("_");
    let a = parts.next().unwrap();
    if a.chars().count() > 1 {
        latex.push(Node::CtrlSeq(a));
    } else {
        latex.push(Node::Char(a.chars().next().unwrap()));
    }
    if let Some(b) = parts.next() {
        let b = b.strip_prefix("{").unwrap().strip_suffix("}").unwrap();
        latex.push(Node::SubSup {
            sub: Some(b.chars().map(Node::Char).collect()),
            sup: None,
        });
    }
    latex.push(Node::CtrlSeq("le"));

    let mut field = MathField::from(&latex);
    field.interactiveness = Interactiveness::None;
    field.scale = 18.0;

    field
}

fn number_to_latex(nodes: &mut Vec<latex_tree::Node>, mut x: f64) {
    use latex_tree::Node::{self, Char as C};

    if x.is_nan() {
        nodes.push(Node::Frac {
            num: vec![C('0')],
            den: vec![C('0')],
        });
        return;
    }

    if x.is_sign_negative() {
        nodes.push(C('-'));
        x = -x;
    }

    if x.is_infinite() {
        nodes.push(Node::CtrlSeq("infty"));
        return;
    }

    let mut buffer = ryu::Buffer::new();
    let mut s = buffer.format_finite(x).split('e');
    let m = s.next().unwrap();
    nodes.extend(m.strip_suffix(".0").unwrap_or(m).chars().map(C));

    if let Some(e) = s.next() {
        nodes.extend([
            Node::CtrlSeq("times"),
            C('1'),
            C('0'),
            Node::SubSup {
                sub: None,
                sup: Some(e.chars().map(C).collect()),
            },
        ]);
    }
}

impl OutputUi {
    fn field_from_latex(latex: &[latex_tree::Node]) -> OutputUi {
        let mut field = MathField::from(latex);
        field.interactiveness = Interactiveness::Select;
        field.scale = 18.0;
        field.left_padding = 0.22;
        field.right_padding = 0.4;
        field.bottom_padding = 0.19;
        field.top_padding = 0.25;
        OutputUi::Field(field)
    }

    fn set_domain(&mut self, name: &str) {
        if let OutputUi::Domain { name: existing, .. } = self
            && existing == name
        {
            return;
        }

        *self = OutputUi::Domain {
            name: name.into(),
            name_field: create_le_name_le(name),
            min_state: Default::default(),
            max_state: Default::default(),
        }
    }
}

#[derive(Debug, Default)]
enum OutputData {
    #[default]
    None,
    Error(String),
    DraggablePoint(Geometry),
    Geometry(Vec<Geometry>),
}

const SLIDER_SOFT_MIN_DEFAULT: f64 = -10.0;
const SLIDER_SOFT_MAX_DEFAULT: f64 = 10.0;
const SLIDER_STEP_DEFAULT: f64 = 0.0;
const PARAMETRIC_DOMAIN_MIN_DEFAULT: f64 = 0.0;
const PARAMETRIC_DOMAIN_MAX_DEFAULT: f64 = 1.0;

impl Output {
    const NONE: Output = Output {
        ui: OutputUi::None,
        data: OutputData::None,
    };

    fn new_error(error: String) -> Output {
        Output {
            ui: OutputUi::None,
            data: OutputData::Error(error),
        }
    }

    fn set_slider_name(&mut self, new_name: &str) {
        self.data = OutputData::None;

        // If there was an already existing slider then we just need to update its name
        if let OutputUi::Slider {
            name, name_field, ..
        } = &mut self.ui
        {
            if *name != new_name {
                *name = new_name.into();
                *name_field = create_le_name_le(new_name);
            }
            return;
        }

        // Otherwise we need to create a whole new slider UI
        let mut step_label_field = MathField::from(
            &"Step:"
                .chars()
                .map(latex_tree::Node::Char)
                .collect::<Vec<_>>(),
        );
        step_label_field.no_italic(true);
        step_label_field.scale = 15.7;
        step_label_field.left_padding = 0.69;
        step_label_field.right_padding = -0.15;
        step_label_field.interactiveness = Interactiveness::None;
        self.ui = OutputUi::Slider {
            value: 0.0,
            min: SLIDER_SOFT_MIN_DEFAULT,
            max: SLIDER_SOFT_MAX_DEFAULT,
            step: SLIDER_STEP_DEFAULT,
            dragging: None,
            hovered: false,
            name: new_name.into(),
            name_field: create_le_name_le(new_name),
            step_label_field,
            min_state: Default::default(),
            max_state: Default::default(),
            step_state: Default::default(),
        };
    }

    fn set_slider_fields(
        &mut self,
        slider: &mut Slider,
        new_value: Option<f64>,
        new_min: f64,
        new_max: f64,
        new_step: Option<f64>,
    ) {
        let OutputUi::Slider {
            value,
            min,
            max,
            step,
            ..
        } = &mut self.ui
        else {
            unreachable!("slider UI should've already been created before");
        };
        if let Some(new_value) = new_value {
            *value = new_value;
        }
        for (old, field, new) in [
            (min, &mut slider.hard_min, new_min),
            (max, &mut slider.hard_max, new_max),
        ] {
            if *old != new {
                *old = new;
                let mut latex = vec![];
                number_to_latex(&mut latex, new);
                field.0.set_placeholder(&latex);
            }
        }

        *step = new_step.unwrap_or(SLIDER_STEP_DEFAULT);
    }

    const SLIDER_BAR_RADIUS: f64 = 3.0;
    const SLIDER_TICK_RADIUS: f64 = Self::SLIDER_BAR_RADIUS / 3.0;
    const SLIDER_STEP_TICKS_THRESHOLD: f64 = 0.03;
    const SLIDER_POINT_RADIUS: f64 = 11.0;
    // TODO rename this constant since its used for sliders too
    const DOMAIN_VALUE_MIN_WIDTH: f64 = 35.0;
    const DOMAIN_VALUE_MAX_WIDTH: f64 = 70.0;

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        padding: f64,
        top_left: DVec2,
        width: f64,
        field_has_focus: bool,
        slider: &mut Slider,
        parametric_domain: &mut Domain<(MathField, Result<ast::Expression, String>)>,
    ) -> (Response, Option<f64>, Option<Message>, Bounds) {
        match &mut self.ui {
            OutputUi::None => (Response::default(), None, None, Bounds::default()),
            OutputUi::Slider {
                value,
                min,
                max,
                step,
                dragging,
                hovered: point_hovered,
                name: _,
                name_field,
                step_label_field,
                min_state,
                max_state,
                step_state,
            } => {
                let is_slider_edit_shown = field_has_focus
                    || slider.hard_min.0.has_focus()
                    || slider.hard_max.0.has_focus()
                    || slider.step.0.has_focus()
                    || min_state.error
                    || max_state.error
                    || step_state.error;

                for field in [&mut slider.hard_max.0, &mut slider.hard_min.0] {
                    field.use_placeholder_if_empty = !is_slider_edit_shown;
                    field.grayed = !is_slider_edit_shown;
                    field.scale = if is_slider_edit_shown { 15.7 } else { 12.9 };
                }

                let mut slider_min_size = slider.hard_min.0.expression_size().map(|s| ctx.ceil(s));
                let mut slider_max_size = slider.hard_max.0.expression_size().map(|s| ctx.ceil(s));

                let mut response = Response::default();

                if is_slider_edit_shown {
                    let name_size = name_field.expression_size().map(|s| ctx.ceil(s));
                    let step_label_size = step_label_field.expression_size().map(|s| ctx.ceil(s));
                    let mut slider_step_size = slider.step.0.expression_size().map(|s| ctx.ceil(s));

                    for x in [
                        &mut slider_min_size.x,
                        &mut slider_max_size.x,
                        &mut slider_step_size.x,
                    ] {
                        *x = x.clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);
                    }

                    let height = 0f64 // muh formatting
                        .max(slider_min_size.y)
                        .max(name_size.y)
                        .max(slider_max_size.y)
                        .max(step_label_size.y)
                        .max(slider_step_size.y);

                    let slider_min_bounds = Bounds {
                        pos: dvec2(
                            top_left.x + padding,
                            top_left.y + (height - slider_min_size.y) / 2.0,
                        ),
                        size: slider_min_size,
                    };
                    let name_bounds = Bounds {
                        pos: dvec2(
                            slider_min_bounds.right(),
                            top_left.y + (height - name_size.y) / 2.0,
                        ),
                        size: name_size,
                    };
                    let slider_max_bounds = Bounds {
                        pos: dvec2(
                            name_bounds.right(),
                            top_left.y + (height - slider_max_size.y) / 2.0,
                        ),
                        size: slider_max_size,
                    };
                    let step_label_bounds = Bounds {
                        pos: dvec2(
                            slider_max_bounds.right(),
                            top_left.y + (height - step_label_size.y) / 2.0,
                        ),
                        size: step_label_size,
                    };
                    let slider_step_bounds = Bounds {
                        pos: dvec2(
                            step_label_bounds.right(),
                            top_left.y + (height - slider_step_size.y) / 2.0,
                        ),
                        size: slider_step_size,
                    };

                    let (mut response_slider_min, mut message_slider_min) =
                        slider.hard_min.0.update(ctx, event, slider_min_bounds);
                    let (response_name, _) = name_field.update(ctx, event, name_bounds);
                    let (mut response_slider_max, mut message_slider_max) =
                        slider.hard_max.0.update(ctx, event, slider_max_bounds);
                    let (response_step_label, _) =
                        step_label_field.update(ctx, event, step_label_bounds);
                    let (mut response_slider_step, mut message_slider_step) =
                        slider.step.0.update(ctx, event, slider_step_bounds);

                    match message_slider_min {
                        Some(Message::ContentsChanged) => {
                            slider.soft_min = SLIDER_SOFT_MIN_DEFAULT;
                            if !slider.hard_min.0.is_empty() {
                                slider.hard_min.1 =
                                    parse_standalone_expression(&slider.hard_min.0.to_latex());
                            }
                        }
                        Some(Message::Left) => message_slider_min = None,
                        Some(Message::Right) => {
                            message_slider_min = None;
                            slider.hard_min.0.unfocus();
                            slider.hard_max.0.select_all();
                        }
                        Some(Message::Up | Message::Down | Message::Add) => {
                            slider.hard_min.0.unfocus();
                        }
                        Some(Message::Remove) => {
                            message_slider_min = None;
                            if slider.soft_min != SLIDER_SOFT_MIN_DEFAULT {
                                slider.soft_min = SLIDER_SOFT_MIN_DEFAULT;
                                message_slider_min = Some(Message::ContentsChanged);
                            }
                        }
                        None => {}
                    }

                    match message_slider_max {
                        Some(Message::ContentsChanged) => {
                            slider.soft_max = SLIDER_SOFT_MAX_DEFAULT;
                            if !slider.hard_max.0.is_empty() {
                                slider.hard_max.1 =
                                    parse_standalone_expression(&slider.hard_max.0.to_latex());
                            }
                        }
                        Some(Message::Left) => {
                            message_slider_max = None;
                            slider.hard_max.0.unfocus();
                            slider.hard_min.0.select_all();
                        }
                        Some(Message::Right) => {
                            message_slider_max = None;
                            slider.hard_max.0.unfocus();
                            slider.step.0.select_all();
                        }
                        Some(Message::Up | Message::Down | Message::Add) => {
                            slider.hard_max.0.unfocus();
                        }
                        Some(Message::Remove) => {
                            message_slider_max = None;
                            if slider.soft_max != SLIDER_SOFT_MAX_DEFAULT {
                                slider.soft_max = SLIDER_SOFT_MAX_DEFAULT;
                                message_slider_max = Some(Message::ContentsChanged);
                            }
                        }
                        None => {}
                    }

                    match message_slider_step {
                        Some(Message::ContentsChanged) => {
                            if slider.step.0.is_empty() {
                                slider.step.1 = Ok(ast::Expression::Number(0.0));
                            } else {
                                slider.step.1 =
                                    parse_standalone_expression(&slider.step.0.to_latex());
                            }
                        }
                        Some(Message::Left) => {
                            message_slider_step = None;
                            slider.step.0.unfocus();
                            slider.hard_max.0.select_all();
                        }
                        Some(Message::Right | Message::Remove) => message_slider_step = None,
                        Some(Message::Up | Message::Down | Message::Add) => {
                            slider.step.0.unfocus();
                        }
                        None => {}
                    }

                    let f =
                        |f: &MathField, b: Bounds, s: &mut DomainFocusState, r: &mut Response| {
                            let new = if f.has_focus() {
                                DomainFocusState::Focussed
                            } else if b.contains(ctx.cursor) {
                                DomainFocusState::Hovered
                            } else {
                                DomainFocusState::None
                            };
                            if new != *s {
                                *s = new;
                                r.request_redraw();
                            }
                        };
                    f(
                        &slider.hard_min.0,
                        slider_min_bounds,
                        &mut min_state.state,
                        &mut response_slider_min,
                    );
                    f(
                        &slider.hard_max.0,
                        slider_max_bounds,
                        &mut max_state.state,
                        &mut response_slider_max,
                    );
                    f(
                        &slider.step.0,
                        slider_step_bounds,
                        &mut step_state.state,
                        &mut response_slider_step,
                    );

                    let response = response_slider_min
                        .or(response_name)
                        .or(response_slider_max)
                        .or(response_step_label)
                        .or(response_slider_step);

                    let bounds = slider_min_bounds
                        .union(name_bounds)
                        .union(slider_max_bounds)
                        .union(step_label_bounds)
                        .union(slider_step_bounds);

                    (
                        response,
                        None,
                        message_slider_min
                            .or(message_slider_max)
                            .or(message_slider_step),
                        bounds,
                    )
                } else {
                    let point_radius = ctx.round_nonzero(Self::SLIDER_POINT_RADIUS);
                    let height = (point_radius * 2.0)
                        .max(slider_min_size.y)
                        .max(slider_max_size.y);

                    let slider_min_bounds = Bounds {
                        pos: top_left
                            + dvec2(0.5 * padding, height / 2.0 - slider_min_size.y / 2.0),
                        size: slider_min_size,
                    };
                    let slider_max_bounds = Bounds {
                        pos: top_left
                            + dvec2(
                                width - 0.5 * padding - slider_max_size.x,
                                height / 2.0 - slider_max_size.y / 2.0,
                            ),
                        size: slider_max_size,
                    };

                    let slider_bar_left = slider_min_bounds.right() + 0.8 * padding;
                    let slider_bar_right = slider_max_bounds.left() - 0.8 * padding;
                    let mut point = dvec2(
                        mix(
                            slider_bar_left,
                            slider_bar_right,
                            unmix(*value, *min, *max).clamp(0.0, 1.0),
                        ),
                        top_left.y + height / 2.0,
                    );
                    let point_bounds = Bounds {
                        pos: point - point_radius,
                        size: DVec2::splat(2.0 * point_radius),
                    };
                    let new_point_hovered = point_bounds.contains(ctx.cursor);
                    let new_slider_min_hovered = slider_min_bounds.contains(ctx.cursor);
                    let new_slider_max_hovered = slider_max_bounds.contains(ctx.cursor);

                    let mut new_value = None;
                    let mut slider_touched = false;

                    match event {
                        Event::CursorMoved { .. } if dragging.is_some() => {
                            let offset = dragging.unwrap();
                            // Not using `.clamp()` because it panics if sidebar is resized too small
                            point.x = (ctx.cursor.x + offset)
                                .max(slider_bar_left)
                                .min(slider_bar_right);
                            *value = mix(
                                *min,
                                *max,
                                unmix(point.x, slider_bar_left, slider_bar_right),
                            );
                            *value = apply_slider(*value, *min, *max, *step);
                            new_value = Some(*value);
                            slider_touched = true;
                            response.consume_event();
                            response.request_redraw();
                        }
                        Event::MouseInput(ElementState::Pressed, MouseButton::Left) => {
                            if new_point_hovered {
                                *dragging = Some(point.x - ctx.cursor.x);
                                slider_touched = true;
                                response.consume_event();
                            } else if new_slider_min_hovered {
                                slider.hard_min.0.select_all();
                                min_state.state = DomainFocusState::Focussed;
                                response.consume_event();
                                response.request_redraw()
                            } else if new_slider_max_hovered {
                                slider.hard_max.0.select_all();
                                max_state.state = DomainFocusState::Focussed;
                                response.consume_event();
                                response.request_redraw()
                            }
                        }
                        Event::MouseInput(ElementState::Released, MouseButton::Left)
                            if dragging.is_some() =>
                        {
                            *dragging = None;
                            response.consume_event();
                        }
                        _ => {}
                    }

                    if slider_touched {
                        slider.soft_min = slider.soft_min.min(*value);
                        slider.soft_max = slider.soft_max.max(*value);
                    }

                    let new_point_hovered = new_point_hovered || dragging.is_some();

                    if *point_hovered != new_point_hovered {
                        *point_hovered = new_point_hovered;
                        response.request_redraw();
                    }

                    #[cfg(not(windows))]
                    let (grab, grabbing) = (CursorIcon::Grab, CursorIcon::Grabbing);

                    // https://github.com/rust-windowing/winit/issues/1043
                    #[cfg(windows)]
                    let (grab, grabbing) = (CursorIcon::EwResize, CursorIcon::EwResize);

                    if dragging.is_some() {
                        response.cursor_mode = CursorMode::Icon(grabbing);
                    } else if *point_hovered {
                        response.cursor_mode = CursorMode::Icon(grab);
                    } else if new_slider_min_hovered || new_slider_max_hovered {
                        response.cursor_mode = CursorMode::Icon(CursorIcon::Pointer);
                    }

                    let bounds = Bounds {
                        pos: top_left,
                        size: dvec2(width, height),
                    };

                    (response, new_value, None, bounds)
                }
            }
            OutputUi::Field(field) => {
                let size = field.expression_size().map(|s| ctx.ceil(s));
                let right = top_left.x + width - 0.5 * padding;
                let left = (right - size.x).max(top_left.x + padding);
                let bounds = Bounds {
                    pos: dvec2(left, top_left.y),
                    size: dvec2(right - left, size.y),
                };
                let (mut response, _) = field.update(ctx, event, bounds);

                if let Some(UserSelection { anchor, focus }) = field.get_selection() {
                    let mut clamp = |mut cursor: Cursor| {
                        let index = cursor
                            .path
                            .first_mut()
                            .map_or(&mut cursor.index, |(index, _)| index);
                        if *index == 0 {
                            *index = 1;
                            response.request_redraw();
                        }
                        cursor
                    };
                    field.set_selection((clamp(anchor.clone()), clamp(focus.clone())));
                }

                (response, None, None, bounds)
            }
            OutputUi::Domain {
                name: _,
                name_field,
                min_state,
                max_state,
            } => {
                let mut size1 = parametric_domain
                    .min
                    .0
                    .expression_size()
                    .map(|s| ctx.ceil(s));
                let size2 = name_field.expression_size().map(|s| ctx.ceil(s));
                let mut size3 = parametric_domain
                    .max
                    .0
                    .expression_size()
                    .map(|s| ctx.ceil(s));

                size1.x = size1
                    .x
                    .clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);
                size3.x = size3
                    .x
                    .clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);

                let height = size1.y.max(size2.y).max(size3.y);

                let left1 = top_left.x + padding;
                let left2 = left1 + size1.x;
                let left3 = left2 + size2.x;

                let bounds1 = Bounds {
                    pos: dvec2(left1, top_left.y + (height - size1.y) / 2.0),
                    size: size1,
                };
                let bounds2 = Bounds {
                    pos: dvec2(left2, top_left.y + (height - size2.y) / 2.0),
                    size: size2,
                };
                let bounds3 = Bounds {
                    pos: dvec2(left3, top_left.y + (height - size3.y) / 2.0),
                    size: size3,
                };

                let (mut response1, mut message1) =
                    parametric_domain.min.0.update(ctx, event, bounds1);
                let (response2, _) = name_field.update(ctx, event, bounds2);
                let (mut response3, mut message3) =
                    parametric_domain.max.0.update(ctx, event, bounds3);

                match message1 {
                    Some(Message::ContentsChanged) => {
                        let mut latex = parametric_domain.min.0.to_latex();
                        if latex.is_empty() {
                            latex = parametric_domain.min.0.get_placeholder();
                        }
                        parametric_domain.min.1 = parse_standalone_expression(&latex)
                    }
                    Some(Message::Left | Message::Remove) => message1 = None,
                    Some(Message::Right) => {
                        message1 = None;
                        parametric_domain.min.0.unfocus();
                        parametric_domain.max.0.select_all();
                    }
                    Some(Message::Up | Message::Down | Message::Add) => {
                        parametric_domain.min.0.unfocus()
                    }
                    None => {}
                }

                match message3 {
                    Some(Message::ContentsChanged) => {
                        let mut latex = parametric_domain.max.0.to_latex();
                        if latex.is_empty() {
                            latex = parametric_domain.max.0.get_placeholder();
                        }
                        parametric_domain.max.1 = parse_standalone_expression(&latex)
                    }
                    Some(Message::Left) => {
                        message3 = None;
                        parametric_domain.max.0.unfocus();
                        parametric_domain.min.0.select_all();
                    }
                    Some(Message::Right | Message::Remove) => message3 = None,
                    Some(Message::Up | Message::Down | Message::Add) => {
                        parametric_domain.max.0.unfocus()
                    }
                    None => {}
                }

                let f = |f: &MathField, b: Bounds, s: &mut DomainFocusState, r: &mut Response| {
                    let new = if f.has_focus() {
                        DomainFocusState::Focussed
                    } else if b.contains(ctx.cursor) {
                        DomainFocusState::Hovered
                    } else {
                        DomainFocusState::None
                    };
                    if new != *s {
                        *s = new;
                        r.request_redraw();
                    }
                };
                f(
                    &parametric_domain.min.0,
                    bounds1,
                    &mut min_state.state,
                    &mut response1,
                );
                f(
                    &parametric_domain.max.0,
                    bounds3,
                    &mut max_state.state,
                    &mut response3,
                );

                let response = response1.or(response2).or(response3);

                let bounds = bounds1.union(bounds2).union(bounds3);

                (response, None, message1.or(message3), bounds)
            }
        }
    }

    fn render(
        &mut self,
        ctx: &Context,
        padding: f64,
        top_left: DVec2,
        width: f64,
        field_has_focus: bool,
        slider: &mut Slider,
        parametric_domain: &mut Domain<(MathField, Result<ast::Expression, String>)>,
        draw_quad: &mut impl FnMut(DVec2, DVec2, QuadKind),
    ) -> f64 {
        match &mut self.ui {
            OutputUi::None => 0.0,
            OutputUi::Slider {
                value,
                min,
                max,
                step,
                hovered,
                name_field,
                step_label_field,
                min_state,
                max_state,
                step_state,
                ..
            } => {
                let is_slider_edit_shown = field_has_focus
                    || slider.hard_min.0.has_focus()
                    || slider.hard_max.0.has_focus()
                    || slider.step.0.has_focus()
                    || min_state.error
                    || max_state.error
                    || step_state.error;

                for field in [&mut slider.hard_max.0, &mut slider.hard_min.0] {
                    field.use_placeholder_if_empty = !is_slider_edit_shown;
                    field.grayed = !is_slider_edit_shown;
                    field.scale = if is_slider_edit_shown { 15.7 } else { 12.9 };
                }

                let mut slider_min_size = slider.hard_min.0.expression_size().map(|s| ctx.ceil(s));
                let mut slider_max_size = slider.hard_max.0.expression_size().map(|s| ctx.ceil(s));

                if is_slider_edit_shown {
                    let name_size = name_field.expression_size().map(|s| ctx.ceil(s));
                    let step_label_size = step_label_field.expression_size().map(|s| ctx.ceil(s));
                    let mut slider_step_size = slider.step.0.expression_size().map(|s| ctx.ceil(s));

                    for x in [
                        &mut slider_min_size.x,
                        &mut slider_max_size.x,
                        &mut slider_step_size.x,
                    ] {
                        *x = x.clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);
                    }

                    let height = 0f64 // muh formatting
                        .max(slider_min_size.y)
                        .max(name_size.y)
                        .max(slider_max_size.y)
                        .max(step_label_size.y)
                        .max(slider_step_size.y);

                    let slider_min_bounds = Bounds {
                        pos: dvec2(
                            top_left.x + padding,
                            top_left.y + (height - slider_min_size.y) / 2.0,
                        ),
                        size: slider_min_size,
                    };
                    let name_bounds = Bounds {
                        pos: dvec2(
                            slider_min_bounds.right(),
                            top_left.y + (height - name_size.y) / 2.0,
                        ),
                        size: name_size,
                    };
                    let slider_max_bounds = Bounds {
                        pos: dvec2(
                            name_bounds.right(),
                            top_left.y + (height - slider_max_size.y) / 2.0,
                        ),
                        size: slider_max_size,
                    };
                    let step_label_bounds = Bounds {
                        pos: dvec2(
                            slider_max_bounds.right(),
                            top_left.y + (height - step_label_size.y) / 2.0,
                        ),
                        size: step_label_size,
                    };
                    let slider_step_bounds = Bounds {
                        pos: dvec2(
                            step_label_bounds.right(),
                            top_left.y + (height - slider_step_size.y) / 2.0,
                        ),
                        size: slider_step_size,
                    };

                    slider.hard_min.0.render(ctx, slider_min_bounds, draw_quad);
                    name_field.render(ctx, name_bounds, draw_quad);
                    slider.hard_max.0.render(ctx, slider_max_bounds, draw_quad);
                    step_label_field.render(ctx, step_label_bounds, draw_quad);
                    slider.step.0.render(ctx, slider_step_bounds, draw_quad);

                    let mut f = |b: Bounds, s: DomainState| {
                        draw_quad(
                            ctx.scale_factor * (b.pos + dvec2(0.0, b.size.y - 1.0)),
                            ctx.scale_factor
                                * (b.pos
                                    + b.size
                                    + dvec2(
                                        0.0,
                                        if s.state == DomainFocusState::None && !s.error {
                                            0.0
                                        } else {
                                            1.0
                                        },
                                    )),
                            match s.state {
                                _ if s.error => QuadKind::DomainBoundError,
                                DomainFocusState::None | DomainFocusState::Hovered => {
                                    QuadKind::DomainBoundUnfocussed
                                }
                                DomainFocusState::Focussed => QuadKind::DomainBoundFocussed,
                            },
                        )
                    };

                    f(slider_min_bounds, *min_state);
                    f(slider_max_bounds, *max_state);
                    f(slider_step_bounds, *step_state);

                    let bounds = slider_min_bounds
                        .union(name_bounds)
                        .union(slider_max_bounds)
                        .union(step_label_bounds)
                        .union(slider_step_bounds);
                    bounds.size.y
                } else {
                    let bar_radius = ctx.round_nonzero(Self::SLIDER_BAR_RADIUS);
                    let tick_radius = ctx.round_nonzero(Self::SLIDER_TICK_RADIUS);
                    let point_radius = ctx.round_nonzero(Self::SLIDER_POINT_RADIUS);
                    let height = (point_radius * 2.0)
                        .max(slider_min_size.y)
                        .max(slider_max_size.y);

                    let slider_min_bounds = Bounds {
                        pos: top_left
                            + dvec2(0.5 * padding, height / 2.0 - slider_min_size.y / 2.0),
                        size: slider_min_size,
                    };
                    let slider_max_bounds = Bounds {
                        pos: top_left
                            + dvec2(
                                width - 0.5 * padding - slider_max_size.x,
                                height / 2.0 - slider_max_size.y / 2.0,
                            ),
                        size: slider_max_size,
                    };

                    let slider_bar_left = slider_min_bounds.right() + 0.8 * padding;
                    let slider_bar_right = slider_max_bounds.left() - 0.8 * padding;
                    let point = dvec2(
                        mix(
                            slider_bar_left,
                            slider_bar_right,
                            unmix(*value, *min, *max).clamp(0.0, 1.0),
                        ),
                        top_left.y + height / 2.0,
                    );
                    let bounds = Bounds {
                        pos: top_left,
                        size: dvec2(width, height),
                    };

                    draw_quad(
                        ctx.scale_factor * (dvec2(slider_bar_left, point.y - bar_radius)),
                        ctx.scale_factor * (dvec2(slider_bar_right, point.y + bar_radius)),
                        QuadKind::SliderBar,
                    );

                    if step.abs() >= (*max - *min) * Self::SLIDER_STEP_TICKS_THRESHOLD {
                        let n = (((*max - *min) / step.abs()).ceil() as u32).max(1) - 1;
                        for i in 1..=n {
                            let value = *min + *step * i as f64;
                            let tick = dvec2(
                                mix(slider_bar_left, slider_bar_right, unmix(value, *min, *max)),
                                point.y,
                            );
                            draw_quad(
                                ctx.scale_factor * (tick - tick_radius),
                                ctx.scale_factor * (tick + tick_radius),
                                QuadKind::SliderStepTick,
                            );
                        }
                    }

                    if (*min..=*max).contains(&0.0) {
                        let tick = dvec2(
                            mix(slider_bar_left, slider_bar_right, unmix(0.0, *min, *max)),
                            point.y,
                        );
                        draw_quad(
                            ctx.scale_factor * (tick - tick_radius),
                            ctx.scale_factor * (tick + tick_radius),
                            QuadKind::SliderZeroTick,
                        );
                    }

                    draw_quad(
                        ctx.scale_factor * (point - point_radius),
                        ctx.scale_factor * (point + point_radius),
                        QuadKind::SliderPointOuter,
                    );
                    let inner_radius = if *hovered { point_radius } else { bar_radius };
                    draw_quad(
                        ctx.scale_factor * (point - inner_radius),
                        ctx.scale_factor * (point + inner_radius),
                        QuadKind::SliderPointInner,
                    );

                    slider.hard_min.0.render(ctx, slider_min_bounds, draw_quad);
                    slider.hard_max.0.render(ctx, slider_max_bounds, draw_quad);

                    bounds.size.y
                }
            }
            OutputUi::Field(field) => {
                let size = field.expression_size().map(|s| ctx.ceil(s));
                let right = top_left.x + width - 0.5 * padding;
                let left = (right - size.x).max(top_left.x + padding);
                let bounds = Bounds {
                    pos: dvec2(left, top_left.y),
                    size: dvec2(right - left, size.y),
                };
                draw_quad(
                    ctx.scale_factor * bounds.pos,
                    ctx.scale_factor * (bounds.pos + bounds.size),
                    QuadKind::OutputValueBox,
                );
                field.render(ctx, bounds, draw_quad);
                bounds.size.y
            }
            OutputUi::Domain {
                name: _,
                name_field,
                min_state,
                max_state,
            } => {
                let mut size1 = parametric_domain
                    .min
                    .0
                    .expression_size()
                    .map(|s| ctx.ceil(s));
                let size2 = name_field.expression_size().map(|s| ctx.ceil(s));
                let mut size3 = parametric_domain
                    .max
                    .0
                    .expression_size()
                    .map(|s| ctx.ceil(s));

                size1.x = size1
                    .x
                    .clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);
                size3.x = size3
                    .x
                    .clamp(Self::DOMAIN_VALUE_MIN_WIDTH, Self::DOMAIN_VALUE_MAX_WIDTH);

                let height = size1.y.max(size2.y).max(size3.y);

                let left1 = top_left.x + padding;
                let left2 = left1 + size1.x;
                let left3 = left2 + size2.x;

                let bounds1 = Bounds {
                    pos: dvec2(left1, top_left.y + (height - size1.y) / 2.0),
                    size: size1,
                };
                let bounds2 = Bounds {
                    pos: dvec2(left2, top_left.y + (height - size2.y) / 2.0),
                    size: size2,
                };
                let bounds3 = Bounds {
                    pos: dvec2(left3, top_left.y + (height - size3.y) / 2.0),
                    size: size3,
                };

                parametric_domain.min.0.render(ctx, bounds1, draw_quad);
                name_field.render(ctx, bounds2, draw_quad);
                parametric_domain.max.0.render(ctx, bounds3, draw_quad);

                let mut f = |b: Bounds, s: DomainState| {
                    draw_quad(
                        ctx.scale_factor * (b.pos + dvec2(0.0, b.size.y - 1.0)),
                        ctx.scale_factor
                            * (b.pos
                                + b.size
                                + dvec2(
                                    0.0,
                                    if s.state == DomainFocusState::None && !s.error {
                                        0.0
                                    } else {
                                        1.0
                                    },
                                )),
                        match s.state {
                            _ if s.error => QuadKind::DomainBoundError,
                            DomainFocusState::None | DomainFocusState::Hovered => {
                                QuadKind::DomainBoundUnfocussed
                            }
                            DomainFocusState::Focussed => QuadKind::DomainBoundFocussed,
                        },
                    )
                };

                f(bounds1, *min_state);
                f(bounds3, *max_state);

                let bounds = bounds1.union(bounds2).union(bounds3);
                bounds.size.y
            }
        }
    }
}

/// If `expr` is a numeric literal then this returns its value, otherwise it returns `None`.
fn get_numeric_literal(expr: &parse::ast::Expression) -> Option<f64> {
    match expr {
        parse::ast::Expression::Number(x) => Some(*x),
        parse::ast::Expression::Op {
            operation: parse::op::OpName::Neg,
            args: arguments,
        } => Some(-get_numeric_literal(
            arguments.first().expect("neg should have one argument"),
        )?),
        _ => None,
    }
}

/// If the hard bound is empty, then the soft bound is used.
struct Slider {
    hard_min: (MathField, Result<parse::ast::Expression, String>),
    soft_min: f64,
    hard_max: (MathField, Result<parse::ast::Expression, String>),
    soft_max: f64,
    step: (MathField, Result<parse::ast::Expression, String>),
    /// This is what is displayed to the user when a slider is shown instead of
    /// the actual math field. It's to handle desync between the actual value vs
    /// clamped slider value, e.g., when slider bounds get animated.
    // TODO fix this ugly solution, it's annoying having to maintain both fake_field and real field
    fake_field: MathField,
    fake_field_value: f64,
}

struct Expression {
    field: MathField,
    slider: Slider,
    parametric_domain: Domain<(MathField, Result<parse::ast::Expression, String>)>,
    ast: Option<Result<parse::ast::Statement, String>>,
    output: Output,
}

impl Default for Expression {
    fn default() -> Self {
        use latex_tree::Node;
        let f = |s: &str| {
            let mut f = MathField::default();
            f.set_placeholder(&s.chars().map(Node::Char).collect::<Vec<_>>());
            f.scale = 15.7;
            f.left_padding = 0.22;
            f.bottom_padding = 0.25;
            f.top_padding = 0.25;
            f
        };
        Self {
            field: Default::default(),
            slider: Slider {
                hard_min: (
                    f(&SLIDER_SOFT_MIN_DEFAULT.to_string()),
                    Ok(ast::Expression::Number(SLIDER_SOFT_MIN_DEFAULT)),
                ),
                soft_min: SLIDER_SOFT_MIN_DEFAULT,
                hard_max: (
                    f(&SLIDER_SOFT_MAX_DEFAULT.to_string()),
                    Ok(ast::Expression::Number(SLIDER_SOFT_MAX_DEFAULT)),
                ),
                soft_max: SLIDER_SOFT_MAX_DEFAULT,
                step: (f(""), Ok(ast::Expression::Number(0.0))),
                fake_field: Default::default(),
                fake_field_value: 0.0,
            },
            parametric_domain: Domain {
                min: (
                    f(&PARAMETRIC_DOMAIN_MIN_DEFAULT.to_string()),
                    Ok(ast::Expression::Number(PARAMETRIC_DOMAIN_MIN_DEFAULT)),
                ),
                max: (
                    f(&PARAMETRIC_DOMAIN_MAX_DEFAULT.to_string()),
                    Ok(ast::Expression::Number(PARAMETRIC_DOMAIN_MAX_DEFAULT)),
                ),
            },
            ast: None,
            output: Default::default(),
        }
    }
}

impl From<&[latex_tree::Node<'_>]> for Expression {
    fn from(latex: &[latex_tree::Node]) -> Self {
        let mut e = Expression::default();
        e.set_latex(latex);
        e
    }
}

fn create_slider_latex<'a>(name_equal_field: &MathField, value: f64) -> latex_tree::Nodes<'a> {
    use latex_tree::Node::Char as C;
    let name = name_equal_field
        .to_latex()
        .iter()
        .take_while(|n| n != &&C('='))
        .cloned()
        .collect::<Vec<_>>();
    let mut latex = name;
    latex.push(C('='));
    latex.extend(value.to_string().chars().map(C));
    latex
}

impl Expression {
    const PADDING: f64 = 16.0;

    fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        top_left: DVec2,
        width: f64,
    ) -> (Response, Option<Message>, f64) {
        let mut response = Response::default();
        let mut message = None;
        let mut height = 0.0;

        let use_fake_field = matches!(self.output.ui, OutputUi::Slider { .. });
        let field = match use_fake_field {
            true => &self.slider.fake_field,
            false => &self.field,
        };
        let padding = ctx.round(Self::PADDING);
        height += padding;
        let field_bounds = Bounds {
            pos: top_left + dvec2(padding, height),
            size: dvec2(width - padding * 1.5, ctx.ceil(field.expression_size().y)),
        };
        height += field_bounds.size.y;
        height += 0.5 * padding;
        let (output_response, new_value, output_message, output_bounds) = self.output.update(
            ctx,
            event,
            padding,
            top_left + dvec2(0.0, height),
            width,
            field.has_focus(),
            &mut self.slider,
            &mut self.parametric_domain,
        );

        // Update new value from slider
        if let Some(value) = new_value {
            self.set_latex(&create_slider_latex(&self.field, value));
            message = Some(Message::ContentsChanged);
        }

        response = response.or(output_response);
        height += output_bounds.size.y;
        height += 0.5 * padding;

        let bounds = Bounds {
            pos: top_left,
            size: dvec2(width, height),
        };

        let mut r = Response::default();
        if bounds.contains(ctx.cursor)
            && !field_bounds.contains(ctx.cursor)
            && !output_bounds.contains(ctx.cursor)
        {
            r.cursor_mode = CursorMode::Icon(CursorIcon::Pointer);
            if event == &Event::MouseInput(ElementState::Pressed, MouseButton::Left) {
                r.consume_event();
                if !self.field.has_focus() {
                    self.field.focus();
                    r.request_redraw();
                }
            }
        }
        response = response.or(r.or_else(|| {
            let field = match use_fake_field {
                true => &mut self.slider.fake_field,
                false => &mut self.field,
            };
            let (r, m) = field.update(ctx, event, field_bounds);
            if m == Some(Message::ContentsChanged) {
                if use_fake_field {
                    self.field = self.slider.fake_field.clone();
                }
                self.parse_ast();
                if let Some(Ok(ast::Statement::Assignment { value, .. })) = &self.ast
                    && let Some(value) = get_numeric_literal(value)
                {
                    self.slider.fake_field_value = value;
                    if !use_fake_field {
                        // We just became a slider. Transfer control over to fake field
                        self.slider.fake_field = self.field.clone();
                    }

                    if let OutputUi::Slider { min, max, step, .. } = self.output.ui {
                        // maybe using `offset` instead of unconditionally
                        // using `min` reduces floating-point error?
                        let offset = if self.slider.hard_min.0.is_empty() {
                            0.0
                        } else {
                            min
                        };

                        if value < min {
                            self.slider.hard_min.0.clear();
                        }
                        if value > max {
                            self.slider.hard_max.0.clear();
                        }
                        if value != max
                            && value != apply_slider_step(value - offset, step, f64::round) + offset
                        {
                            self.slider.step.0.clear();
                        }
                    }
                }
            }
            if message.is_none() {
                message = m;
            }
            r
        }));

        // Maybe the parametric domain or slider settings got changed
        if let Some(m) = output_message {
            message = match m {
                Message::ContentsChanged | Message::Down | Message::Add => Some(m),
                Message::Left | Message::Right | Message::Remove => unreachable!(),
                Message::Up => {
                    self.focus();
                    None
                }
            };
        }

        (response, message, height)
    }

    fn set_latex(&mut self, latex: &[latex_tree::Node]) {
        self.field = MathField::from(latex);
        self.parse_ast();

        if let Some(Ok(ast::Statement::Assignment { value, .. })) = &self.ast
            && let Some(value) = get_numeric_literal(value)
        {
            self.slider.fake_field_value = value;
            self.slider.fake_field = self.field.clone();
        }
    }

    fn parse_ast(&mut self) {
        let latex = self.field.to_latex();
        self.ast = latex
            .iter()
            .any(|n| n != &latex_tree::Node::Char(' '))
            .then(|| parse_statement(&latex));
    }

    fn focus(&mut self) {
        self.field.focus();
        self.slider.fake_field.focus();
    }

    fn unfocus(&mut self) {
        self.field.unfocus();
        self.slider.fake_field.unfocus();
    }

    fn has_focus(&self) -> bool {
        self.field.has_focus()
    }

    fn render(
        &mut self,
        ctx: &Context,
        top_left: DVec2,
        width: f64,
        draw_quad: &mut impl FnMut(DVec2, DVec2, QuadKind),
    ) -> f64 {
        let mut height = 0.0;

        let use_fake_field = matches!(self.output.ui, OutputUi::Slider { .. });
        let field = match use_fake_field {
            true => &self.slider.fake_field,
            false => &self.field,
        };
        let padding = ctx.round(Self::PADDING);
        height += padding;
        let field_bounds = Bounds {
            pos: top_left + dvec2(padding, height),
            size: dvec2(width - padding * 1.5, ctx.ceil(field.expression_size().y)),
        };
        height += field_bounds.size.y;
        height += 0.5 * padding;
        height += self.output.render(
            ctx,
            padding,
            top_left + dvec2(0.0, height),
            width,
            field.has_focus(),
            &mut self.slider,
            &mut self.parametric_domain,
            draw_quad,
        );
        height += 0.5 * padding;

        let field = match use_fake_field {
            true => &mut self.slider.fake_field,
            false => &mut self.field,
        };
        field.render(ctx, field_bounds, draw_quad);

        height
    }
}

#[derive(Debug, Clone, Copy, From, Into, Add, Sub, PartialEq)]
pub struct ExpressionId(usize);

pub struct ExpressionList {
    expressions: TiVec<ExpressionId, Expression>,
    expressions_changed: bool,
    scroll: f64,
    height: f64,
    expression_bottoms: TiVec<ExpressionId, f64>,
    vm_vars: vm::Vars,

    pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    uniforms_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Uniforms {
    resolution: Vec2,
    scale_factor: f32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Vertex {
    position: Vec2,
    uv: U16Vec2,
    kind: u32,
}

fn create_index_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("latex_index_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::INDEX,
        mapped_at_creation: false,
    })
}

fn create_vertex_buffer(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("latex_vertex_buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::VERTEX,
        mapped_at_creation: false,
    })
}

impl ExpressionList {
    pub fn new(
        AppGraphics {
            device,
            queue,
            config,
            ..
        }: &AppGraphics,
    ) -> Self {
        let module = device.create_shader_module(wgpu::include_wgsl!("latex.wgsl"));
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("latex_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("latex"),
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("latex_pipeline_layout"),
                    bind_group_layouts: &[Some(&layout)],
                    immediate_size: 0,
                }),
            ),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_latex"),
                compilation_options: Default::default(),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: size_of::<Vertex>() as _,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x2,
                            offset: offset_of!(Vertex::zeroed(), Vertex, position) as _,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Unorm16x2,
                            offset: offset_of!(Vertex::zeroed(), Vertex, uv) as _,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Uint32,
                            offset: offset_of!(Vertex::zeroed(), Vertex, kind) as _,
                            shader_location: 2,
                        },
                    ],
                }],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: Some(wgpu::IndexFormat::Uint32),
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_latex"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            cache: None,
            multiview_mask: None,
        });

        let index_buffer = create_index_buffer(device, 256);
        let vertex_buffer = create_vertex_buffer(device, 256);

        let uniforms_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms_buffer"),
            size: size_of::<Uniforms>().next_multiple_of(16) as _,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        let font_image = image::load_from_memory(include_bytes!("KaTeX.png")).unwrap();
        let font_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("font_texture"),
            size: wgpu::Extent3d {
                width: font_image.width(),
                height: font_image.height(),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &font_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &font_image.to_rgba8(),
            wgpu::TexelCopyBufferLayout {
                bytes_per_row: Some(4 * font_image.width()),
                ..Default::default()
            },
            font_texture.size(),
        );

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("latex_bind_group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(
                        uniforms_buffer.as_entire_buffer_binding(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &font_texture.create_view(&Default::default()),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&device.create_sampler(
                        &wgpu::SamplerDescriptor {
                            label: Some("bilinear"),
                            mag_filter: wgpu::FilterMode::Linear,
                            min_filter: wgpu::FilterMode::Linear,
                            ..Default::default()
                        },
                    )),
                },
            ],
        });

        let expressions = [];
        Self {
            expressions: expressions
                .iter()
                .chain(Some(&""))
                .map(|s| Expression::from(parse_latex(s).unwrap().as_slice()))
                .collect(),
            expressions_changed: true,
            scroll: 0.0,
            height: 0.0,
            expression_bottoms: ti_vec![],
            vm_vars: Default::default(),

            pipeline,
            vertex_buffer,
            index_buffer,
            uniforms_buffer,
            bind_group,
        }
    }

    pub fn point_dragged(&mut self, i: ExpressionId, p: DVec2) {
        use parse::latex_tree::Node::{self, Char as C};
        let name = self.expressions[i]
            .field
            .to_latex()
            .iter()
            .take_while(|n| n != &&C('='))
            .cloned()
            .collect::<Vec<_>>();
        let mut latex = name;
        latex.push(C('='));
        let mut inner = vec![];
        inner.extend(p.x.to_string().chars().map(C));
        inner.push(C(','));
        inner.extend(p.y.to_string().chars().map(C));
        latex.push(Node::DelimitedGroup {
            left: Bracket::Paren,
            right: Bracket::Paren,
            inner,
        });
        self.expressions[i].set_latex(&latex);
        self.expressions_changed = true;
    }

    // Positive `delta` moves the expressions down
    fn scroll(&mut self, delta: f64) {
        const SCROLL_EXTRA: f64 = 50.0;
        self.scroll = (self.scroll - delta)
            .min(SCROLL_EXTRA + self.expression_bottoms.last().unwrap_or(&0.0) - self.height)
            .max(0.0);
    }

    fn scroll_into_view(&mut self, i: ExpressionId) {
        const SCROLL_PADDING: f64 = 25.0;
        let bottom = self.expression_bottoms[i];
        let top = if i.0 == 0 {
            0.0
        } else {
            self.expression_bottoms[i - 1.into()]
        };
        self.scroll((self.height - SCROLL_PADDING - (bottom - self.scroll)).min(0.0));
        self.scroll((SCROLL_PADDING - (top - self.scroll)).max(0.0));
    }

    const SEPARATOR_WIDTH: f64 = 1.0;

    pub fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        bounds: Bounds,
    ) -> (Response, Option<(Vec<Geometry>, vm::Vars)>) {
        self.height = bounds.size.y;
        let mut response = Response::default();
        let mut redraw_geometry = false;

        match event {
            Event::MouseWheel(delta)
                if bounds.contains(ctx.cursor) && delta.abs().y >= delta.x.abs() =>
            {
                self.scroll(delta.y);
                response.consume_event();
                response.request_redraw();
            }
            _ => {
                let mut next_y = bounds.pos.y - self.scroll;
                let mut message = None;
                let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
                let expression_width = bounds.size.x - separator_width;
                let mut original_focus = None;
                self.expression_bottoms.clear();

                for (i, expression) in self.expressions.iter_mut_enumerated() {
                    if expression.has_focus() {
                        original_focus = Some(i);
                    }

                    let (r, m, height) = expression.update(
                        ctx,
                        event,
                        dvec2(bounds.pos.x, next_y),
                        expression_width,
                    );
                    next_y += height;
                    response = response.or(r);
                    message = message.or(m.map(|m| (i, m)));
                    next_y += separator_width;
                    self.expression_bottoms
                        .push(next_y - (bounds.pos.y - self.scroll));
                }

                if let Some((i, m)) = message {
                    match m {
                        Message::ContentsChanged => {
                            self.expressions_changed = true;
                            self.scroll_into_view(i);
                        }
                        Message::Left | Message::Right => {}
                        Message::Up => {
                            if i.0 > 0 {
                                self.expressions[i].unfocus();
                                self.expressions[i - 1.into()].focus();
                                response.request_redraw();
                            }
                        }
                        Message::Down => {
                            if i.0 == self.expressions.len() - 1 {
                                self.expressions.push(Default::default());
                            }
                            self.expressions[i].unfocus();
                            self.expressions[i + 1.into()].focus();
                            response.request_redraw();
                        }
                        Message::Add => {
                            self.expressions_changed = true;
                            self.expressions.insert(i + 1.into(), Default::default());
                            self.expressions[i].unfocus();
                            self.expressions[i + 1.into()].focus();
                            response.request_redraw();
                        }
                        Message::Remove => {
                            self.expressions.remove(i);
                            self.expressions_changed = true;
                            if self.expressions.is_empty() {
                                self.expressions.push(Default::default());
                            }
                            self.expressions[ExpressionId(i.0.saturating_sub(1))].focus();
                            response.request_redraw();
                        }
                    }
                }

                if self.expressions.last().unwrap().has_focus() {
                    self.expressions.push(Default::default());
                    response.request_redraw();
                }

                let new_focus = self
                    .expressions
                    .iter_enumerated()
                    .find_map(|(i, e)| e.has_focus().then_some(i));
                redraw_geometry |= self.expressions_changed || original_focus != new_focus;

                if let Some(i) = new_focus
                    && original_focus != new_focus
                {
                    self.scroll_into_view(i);
                }

                if self.expressions_changed {
                    use latex_tree::Node::{self, Char as C};
                    let colors = [
                        [0.780, 0.267, 0.251, 1.0],
                        [0.176, 0.439, 0.702, 1.0],
                        [0.204, 0.522, 0.263, 1.0],
                        [0.376, 0.259, 0.651, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ];
                    let line_width = 2.5;
                    let fill_opacity = 0.4;
                    let point2 = |nodes: &mut Vec<Node>, x: f64, y: f64| {
                        let mut inner = vec![];
                        number_to_latex(&mut inner, x);
                        inner.push(C(','));
                        number_to_latex(&mut inner, y);
                        nodes.push(Node::DelimitedGroup {
                            left: Bracket::Paren,
                            right: Bracket::Paren,
                            inner,
                        });
                    };
                    let point3 = |nodes: &mut Vec<Node>, x: f64, y: f64, z: f64| {
                        let mut inner = vec![];
                        number_to_latex(&mut inner, x);
                        inner.push(C(','));
                        number_to_latex(&mut inner, y);
                        inner.push(C(','));
                        number_to_latex(&mut inner, z);
                        nodes.push(Node::DelimitedGroup {
                            left: Bracket::Paren,
                            right: Bracket::Paren,
                            inner,
                        });
                    };

                    let mut ei_to_oi: TiVec<ExpressionIndex, ExpressionId> = ti_vec![];
                    let mut list: TiVec<ExpressionIndex, _> = ti_vec![];

                    for (i, e) in self.expressions.iter_mut_enumerated() {
                        let ast = match &e.ast {
                            Some(Ok(ast)) => ast,
                            Some(Err(err)) => {
                                e.output = Output::new_error(format!("parse error: {err}"));
                                continue;
                            }
                            None => {
                                e.output = Output::NONE;
                                continue;
                            }
                        };
                        let mut slider = None;
                        if let parse::ast::Statement::Assignment { name, value } = ast {
                            if get_numeric_literal(value).is_some() {
                                e.output.set_slider_name(name);

                                let OutputUi::Slider {
                                    min_state,
                                    max_state,
                                    step_state,
                                    ..
                                } = &mut e.output.ui
                                else {
                                    unreachable!("we just set slider name so it should exist")
                                };
                                min_state.error = false;
                                max_state.error = false;
                                step_state.error = false;

                                let [min, max, step] =
                                    [&e.slider.hard_min, &e.slider.hard_max, &e.slider.step].map(
                                        |f| (!f.0.is_empty()).then(|| f.1.as_ref().ok()).flatten(),
                                    );
                                slider = Some(NrSlider { min, max, step });
                            } else if let parse::ast::Expression::Op {
                                operation: parse::op::OpName::Point,
                                args: arguments,
                            } = value
                                && arguments.len() == 2
                                && let Some(x) = get_numeric_literal(&arguments[0])
                                && let Some(y) = get_numeric_literal(&arguments[1])
                            {
                                let mut latex = vec![C('=')];
                                point2(&mut latex, x, y);
                                e.output = Output {
                                    ui: OutputUi::None,
                                    data: OutputData::DraggablePoint(Geometry {
                                        width: 8.0,
                                        color: colors[i.0 % colors.len()],
                                        kind: GeometryKind::Point {
                                            p: dvec2(x, y),
                                            draggable: Some(i),
                                        },
                                    }),
                                };
                            } else {
                                e.output.data = OutputData::None;
                            }
                        } else {
                            e.output.data = OutputData::None;
                        }
                        list.push(ExpressionListEntry {
                            expression: ast,
                            parametric_domain: Domain {
                                min: match &e.parametric_domain.min.1 {
                                    Ok(ast) => ast,
                                    Err(_) => &ast::Expression::Number(0.0),
                                },
                                max: match &e.parametric_domain.max.1 {
                                    Ok(ast) => ast,
                                    Err(_) => &ast::Expression::Number(1.0),
                                },
                            },
                            slider,
                        });
                        ei_to_oi.push(i);
                    }

                    let builtin_constants = [
                        ("pi", std::f64::consts::PI),
                        ("tau", std::f64::consts::TAU),
                        ("e", std::f64::consts::E),
                        ("infty", f64::INFINITY),
                    ];
                    let analysis = analyze_expression_list(
                        &list,
                        &builtin_constants.map(|(name, _)| name),
                        false,
                    );

                    let mut function_id_map = HashMap::new();
                    let (program, mut functions, var_indices) = compile_assignments(
                        analysis.constants.iter().map(|&i| &analysis.assignments[i]),
                        analysis.results.iter_enumerated().filter_map(|(i, r)| {
                            let ExpressionResult::Plot {
                                parameters,
                                assignments,
                                ..
                            } = r
                            else {
                                return None;
                            };
                            function_id_map.insert(i, function_id_map.len());
                            Some((
                                parameters.iter().cloned(),
                                assignments.iter().map(|&i| &analysis.assignments[i]),
                            ))
                        }),
                        analysis
                            .builtin_constants
                            .values()
                            .map(|&id| (id, Type::Number)),
                    );
                    let mut functions = function_id_map
                        .into_iter()
                        .map(|(id, i)| (id, std::mem::take(&mut functions[i])))
                        .collect::<HashMap<_, _>>();
                    let mut vm = Vm::new(
                        &program,
                        Default::default(),
                        analysis
                            .builtin_constants
                            .values()
                            .map(|id| var_indices[id]),
                    );

                    for (name, value) in builtin_constants {
                        vm.vars[var_indices[&analysis.builtin_constants[name]]] =
                            vm::Value::Number(value);
                    }

                    vm.run(false);

                    'results_loop: for (ei, r) in analysis.results.into_iter_enumerated() {
                        let i = ei_to_oi[ei];
                        let expression = &mut self.expressions[i];
                        let output = &mut expression.output;

                        if let OutputData::Error(_) = output.data {
                            continue 'results_loop;
                        }

                        match r {
                            ExpressionResult::None => *output = Output::NONE,
                            ExpressionResult::Err(e) => {
                                *output = Output::new_error(format!("analysis error: {e}"))
                            }
                            ExpressionResult::Value(id, ty)
                            | ExpressionResult::Plot { value: id, ty, .. } => {
                                let mut nodes = vec![C('=')];

                                let color = colors[i.0 % colors.len()];
                                let mut geometry = vec![];
                                let mut draw_point = |x: f64, y: f64| {
                                    geometry.push(Geometry {
                                        width: 8.0,
                                        color,
                                        kind: GeometryKind::Point {
                                            p: dvec2(x, y),
                                            draggable: None,
                                        },
                                    });
                                };
                                let list_limit = 10;

                                if let ExpressionResult::Plot {
                                    ref kind,
                                    value,
                                    ref parameters,
                                    ..
                                } = r
                                {
                                    let kind = match kind {
                                        PlotKind::Normal => PlotKind::Normal,
                                        PlotKind::Inverse => PlotKind::Inverse,
                                        PlotKind::Parametric(d) => {
                                            output
                                                .ui
                                                .set_domain(&analysis.freevars[&parameters[0]]);

                                            let min = match &expression.parametric_domain.min.1 {
                                                Ok(_) => match &d.min {
                                                    Ok(id) => match vm.vars[var_indices[id]]
                                                        .clone()
                                                        .number()
                                                    {
                                                        x if x.is_finite() => Ok(x),
                                                        _ => Err(
                                                            "value error: domain bound should be finite".into(),
                                                        ),
                                                    },
                                                    Err(e) => Err(format!("analysis error: {e}")),
                                                },
                                                Err(e) => Err(format!("parse error: {e}")),
                                            };
                                            let max = match &expression.parametric_domain.max.1 {
                                                Ok(_) => match &d.max {
                                                    Ok(id) => match vm.vars[var_indices[id]]
                                                        .clone()
                                                        .number()
                                                    {
                                                        x if x.is_finite() => Ok(x),
                                                        _ => Err(
                                                            "value error: domain bound should be finite".into(),
                                                        ),
                                                    },
                                                    Err(e) => Err(format!("analysis error: {e}")),
                                                },
                                                Err(e) => Err(format!("parse error: {e}")),
                                            };
                                            let OutputUi::Domain {
                                                min_state,
                                                max_state,
                                                ..
                                            } = &mut output.ui
                                            else {
                                                unreachable!()
                                            };
                                            min_state.error = min.is_err();
                                            max_state.error = max.is_err();

                                            let (min, max) = match (min, max) {
                                                (Ok(min), Ok(max)) => (min, max),
                                                (min, max) => {
                                                    output.data = OutputData::Error(
                                                        [("min", min), ("max", max)]
                                                            .into_iter()
                                                            .filter_map(|(n, m)| {
                                                                m.err().map(|e| {
                                                                    format!("(parametric {n}) {e}")
                                                                })
                                                            })
                                                            .collect::<Vec<_>>()
                                                            .join("\n"),
                                                    );
                                                    continue 'results_loop;
                                                }
                                            };

                                            if min > max {
                                                output.data = OutputData::Error("invalid domain limits: min should be less than max".into());
                                                min_state.error = true;
                                                max_state.error = true;
                                                continue 'results_loop;
                                            }

                                            PlotKind::Parametric(Domain { min, max })
                                        }
                                        PlotKind::Implicit => PlotKind::Implicit,
                                    };
                                    output.data = OutputData::Geometry(vec![Geometry {
                                        width: line_width,
                                        color,
                                        kind: GeometryKind::Plot {
                                            kind,
                                            inputs: parameters
                                                .iter()
                                                .map(|p| var_indices[p])
                                                .collect(),
                                            output: var_indices[&value],
                                            instructions: functions.remove(&ei).unwrap(),
                                        },
                                    }]);
                                }

                                if match r {
                                    ExpressionResult::Plot {
                                        ref kind,
                                        ref parameters,
                                        ..
                                    } => {
                                        !matches!(kind, PlotKind::Parametric(_))
                                            && !parameters.is_empty()
                                    }
                                    _ => false,
                                } {
                                    output.ui = OutputUi::None;
                                }

                                if match r {
                                    ExpressionResult::Plot { parameters, .. } => {
                                        parameters.is_empty()
                                    }
                                    _ => true,
                                } {
                                    let v = var_indices[&id];
                                    match ty {
                                        Type::Number => {
                                            number_to_latex(&mut nodes, vm.vars[v].clone().number())
                                        }
                                        Type::NumberList => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, x) in a.borrow().as_slice().iter().enumerate() {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    number_to_latex(&mut inner, *x);
                                                } else {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                    break;
                                                }
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Point2 => {
                                            let x = vm.vars[v].clone().number();
                                            let y = vm.vars[v + 1.into()].clone().number();
                                            draw_point(x, y);
                                            point2(&mut nodes, x, y);
                                        }
                                        Type::Point2List => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, &[x, y]) in
                                                a.borrow().as_chunks().0.iter().enumerate()
                                            {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    point2(&mut inner, x, y);
                                                } else if i == list_limit {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                }
                                                draw_point(x, y);
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Point3 => {
                                            let x = vm.vars[v].clone().number();
                                            let y = vm.vars[v + 1.into()].clone().number();
                                            let z = vm.vars[v + 2.into()].clone().number();
                                            point3(&mut nodes, x, y, z);
                                        }
                                        Type::Point3List => {
                                            let a = vm.vars[v].clone().list();
                                            let mut inner = vec![];
                                            for (i, &[x, y, z]) in
                                                a.borrow().as_chunks().0.iter().enumerate()
                                            {
                                                if i < list_limit {
                                                    if i > 0 {
                                                        inner.push(C(','));
                                                    }
                                                    point3(&mut inner, x, y, z);
                                                } else if i == list_limit {
                                                    inner.extend([C(','), C('.'), C('.'), C('.')]);
                                                }
                                            }
                                            nodes.push(Node::DelimitedGroup {
                                                left: Bracket::Square,
                                                right: Bracket::Square,
                                                inner,
                                            });
                                        }
                                        Type::Polygon => {
                                            let a = vm.vars[v].clone().list();
                                            let a = a.borrow();
                                            let fill = Geometry {
                                                width: line_width,
                                                color: [color[0], color[1], color[2], fill_opacity],
                                                kind: GeometryKind::Fill(
                                                    a.chunks(2)
                                                        .map(|p| dvec2(p[0], p[1]))
                                                        .collect(),
                                                ),
                                            };
                                            let line = Geometry {
                                                width: line_width,
                                                color,
                                                kind: GeometryKind::Line(
                                                    a.chunks(2)
                                                        .chain(a.chunks(2).next())
                                                        .map(|p| dvec2(p[0], p[1]))
                                                        .collect(),
                                                ),
                                            };
                                            geometry.extend([fill, line]);
                                        }
                                        Type::PolygonList => {
                                            let a = vm.vars[v].clone().polygon_list();
                                            geometry.extend(a.borrow().iter().flat_map(|a| {
                                                let a = a.borrow();
                                                let fill = Geometry {
                                                    width: line_width,
                                                    color: [
                                                        color[0],
                                                        color[1],
                                                        color[2],
                                                        fill_opacity,
                                                    ],
                                                    kind: GeometryKind::Fill(
                                                        a.chunks(2)
                                                            .map(|p| dvec2(p[0], p[1]))
                                                            .collect(),
                                                    ),
                                                };
                                                let line = Geometry {
                                                    width: line_width,
                                                    color,
                                                    kind: GeometryKind::Line(
                                                        a.chunks(2)
                                                            .chain(a.chunks(2).take(
                                                                if a.len() > 2 { 1 } else { 0 },
                                                            ))
                                                            .map(|p| dvec2(p[0], p[1]))
                                                            .collect(),
                                                    ),
                                                };
                                                [fill, line]
                                            }));
                                        }
                                        Type::Bool | Type::BoolList => unreachable!(),
                                        Type::EmptyList => nodes.push(Node::DelimitedGroup {
                                            left: Bracket::Square,
                                            right: Bracket::Square,
                                            inner: vec![],
                                        }),
                                    }

                                    if ty.as_single() == Type::Polygon {
                                        output.ui = OutputUi::None;
                                    } else if !matches!(output.data, OutputData::DraggablePoint(_))
                                    {
                                        output.ui = OutputUi::field_from_latex(&nodes);
                                    }
                                    if let OutputData::None = output.data {
                                        output.data = OutputData::Geometry(geometry);
                                    }
                                }
                            }
                            ExpressionResult::Slider { value, slider } => {
                                let OutputUi::Slider {
                                    min_state,
                                    max_state,
                                    step_state,
                                    ..
                                } = &mut output.ui
                                else {
                                    // hm maybe we should set it for the first time here!
                                    unreachable!()
                                };

                                let mut error_msg = String::new();
                                let [min, max, step] = [
                                    (
                                        "min",
                                        &expression.slider.hard_min,
                                        &slider.min,
                                        &mut *min_state,
                                    ),
                                    (
                                        "max",
                                        &expression.slider.hard_max,
                                        &slider.max,
                                        &mut *max_state,
                                    ),
                                    (
                                        "step",
                                        &expression.slider.step,
                                        &slider.step,
                                        &mut *step_state,
                                    ),
                                ]
                                .map(
                                    |(name, field, result, state)| {
                                        if field.0.is_empty() {
                                            return None;
                                        }

                                        let nl = if error_msg.is_empty() { "" } else { "\n" };

                                        if let Err(e) = &field.1 {
                                            write!(
                                                &mut error_msg,
                                                "{nl}(slider {name}) parse error: {e}"
                                            )
                                            .unwrap();
                                            state.error = true;
                                            return None;
                                        }

                                        let id = match result.as_ref().unwrap() {
                                            Ok(id) => id,
                                            Err(e) => {
                                                write!(
                                                    &mut error_msg,
                                                    "{nl}(slider {name}) analysis error: {e}"
                                                )
                                                .unwrap();
                                                state.error = true;
                                                return None;
                                            }
                                        };

                                        let value = vm.vars[var_indices[id]].clone().number();
                                        if !value.is_finite() {
                                            write!(
                                                &mut error_msg,
                                                "{nl}invalid slider {name}: value should be finite"
                                            )
                                            .unwrap();
                                            state.error = true;
                                            return None;
                                        }

                                        Some(value)
                                    },
                                );

                                if !error_msg.is_empty() {
                                    output.data = OutputData::Error(error_msg);
                                } else if let (Some(min), Some(max)) = (min, max)
                                    && min > max
                                {
                                    output.data = OutputData::Error(
                                        "invalid slider limits: min should be less than max".into(),
                                    );
                                    min_state.error = true;
                                    max_state.error = true;
                                }

                                let value =
                                    value.map(|id| vm.vars[var_indices[&id]].clone().number());
                                let slider_min = min.unwrap_or(apply_slider_step(
                                    value.unwrap_or(0.0).min(expression.slider.soft_min),
                                    step.unwrap_or(SLIDER_STEP_DEFAULT),
                                    f64::floor,
                                ));
                                let slider_max = max.unwrap_or({
                                    let max = value.unwrap_or(0.0).max(expression.slider.soft_max);
                                    if let Some(step) = step {
                                        let offset = min.unwrap_or(0.0);
                                        apply_slider_step(max - offset, step, f64::ceil) + offset
                                    } else {
                                        max
                                    }
                                });

                                output.set_slider_fields(
                                    &mut expression.slider,
                                    value,
                                    slider_min,
                                    slider_max,
                                    step,
                                );
                            }
                        }
                    }

                    self.vm_vars = vm.vars;

                    let mut has_error = false;
                    for (i, e) in self.expressions.iter().enumerate() {
                        if let OutputData::Error(e) = &e.output.data {
                            println!("expression {} {e}", i + 1);
                            has_error = true;
                        }
                    }
                    if has_error {
                        println!();
                    }
                }
            }
        }

        for expression in &mut self.expressions {
            if let OutputUi::Slider { value, .. } = expression.output.ui
                && expression.slider.fake_field_value != value
                && !expression.slider.fake_field.has_focus()
            {
                expression.slider.fake_field_value = value;
                expression.slider.fake_field =
                    MathField::from(&create_slider_latex(&expression.field, value));
            }
        }

        let mut geometry = None;

        if redraw_geometry {
            let mut regular_geometry = vec![];
            let mut draggable_points = vec![];
            let mut focussed_geometry = vec![];

            for e in &self.expressions {
                match &e.output.data {
                    OutputData::DraggablePoint(p) => {
                        let mut p = p.clone();
                        if e.has_focus() {
                            p.width *= 1.15;
                            draggable_points.push(p);
                        } else {
                            draggable_points.push(p);
                        }
                    }
                    OutputData::Geometry(geometry) => {
                        if e.has_focus() {
                            for mut g in geometry.iter().cloned() {
                                g.width *= match g.kind {
                                    GeometryKind::Line(_) | GeometryKind::Plot { .. } => 1.4,
                                    GeometryKind::Point { .. } => 1.2,
                                    GeometryKind::Fill(_) => 1.0,
                                };
                                g.color[3] = match g.kind {
                                    GeometryKind::Line(_) | GeometryKind::Plot { .. } => 1.0,
                                    GeometryKind::Point { .. } | GeometryKind::Fill(_) => {
                                        // Same as blending over itself, as if it was rendered twice
                                        1.0 - (1.0 - g.color[3]).powi(2)
                                    }
                                };
                                focussed_geometry.push(g);
                            }
                        } else {
                            regular_geometry.extend_from_slice(geometry);
                        }
                    }
                    _ => {}
                }
            }

            regular_geometry.append(&mut draggable_points);
            regular_geometry.append(&mut focussed_geometry);

            geometry = Some((regular_geometry, self.vm_vars.clone()));
        }

        self.expressions_changed = false;

        if response.requested_redraw {
            // If something wanted a redraw then some heights probably got
            // altered so it would be good to reclamp the scroll
            self.scroll(0.0);
        }

        (response, geometry)
    }

    pub fn render(
        &mut self,
        ctx: &Context,
        AppGraphics {
            device,
            queue,
            config,
            ..
        }: &AppGraphics,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        bounds: Bounds,
    ) {
        let mut indices = vec![];
        let mut vertices = vec![];
        let draw_quad = &mut |p0: DVec2, p1: DVec2, kind: QuadKind| {
            let p0 = p0.as_vec2();
            let p1 = p1.as_vec2();
            let (uv0, uv1) = match kind {
                QuadKind::MsdfGlyph(uv0, uv1)
                | QuadKind::TranslucentMsdfGlyph(uv0, uv1)
                | QuadKind::PlaceholderMsdfGlyph(uv0, uv1)
                | QuadKind::GrayedMsdfGlyph(uv0, uv1) => (uv0, uv1),
                _ => (DVec2::splat(0.0), DVec2::splat(1.0)),
            };
            let kind = kind.index();
            let uv0 = uv0
                .map(|x| (x.clamp(0.0, 1.0) * 65535.0).round())
                .as_u16vec2();
            let uv1 = uv1
                .map(|x| (x.clamp(0.0, 1.0) * 65535.0).round())
                .as_u16vec2();

            indices.push(vertices.len() as u32);
            indices.push(vertices.len() as u32 + 1);
            indices.push(vertices.len() as u32 + 2);
            indices.push(vertices.len() as u32 + 3);
            indices.push(0xffffffff);

            vertices.push(Vertex {
                position: p0,
                uv: uv0,
                kind,
            });
            vertices.push(Vertex {
                position: vec2(p1.x, p0.y),
                uv: u16vec2(uv1.x, uv0.y),
                kind,
            });
            vertices.push(Vertex {
                position: vec2(p0.x, p1.y),
                uv: u16vec2(uv0.x, uv1.y),
                kind,
            });
            vertices.push(Vertex {
                position: p1,
                uv: uv1,
                kind,
            });
        };
        let mut next_y = bounds.pos.y - self.scroll;
        let separator_width = ctx.round_nonzero(Self::SEPARATOR_WIDTH);
        let expression_width = bounds.size.x - separator_width;

        for expression in &mut self.expressions {
            let height = expression.render(
                ctx,
                dvec2(bounds.pos.x, next_y),
                expression_width,
                draw_quad,
            );
            next_y += height;
            let p0 = dvec2(bounds.pos.x, next_y);
            let p1 = p0 + dvec2(bounds.size.x, separator_width);
            draw_quad(
                ctx.scale_factor * p0,
                ctx.scale_factor * p1,
                QuadKind::GrayBox,
            );
            next_y += separator_width;
        }

        {
            let p0 = dvec2(bounds.right() - separator_width, bounds.top());
            let p1 = dvec2(bounds.right(), bounds.bottom());
            draw_quad(
                ctx.scale_factor * p0,
                ctx.scale_factor * p1,
                QuadKind::GrayBox,
            );
        }

        let indices_size = size_of_val(&indices[..]) as u64;
        if indices_size > self.index_buffer.size() {
            self.index_buffer = create_index_buffer(device, indices_size);
        }

        let vertices_size = size_of_val(&vertices[..]) as u64;
        if vertices_size > self.vertex_buffer.size() {
            self.vertex_buffer = create_vertex_buffer(device, vertices_size);
        }

        queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));
        queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        queue.write_buffer(
            &self.uniforms_buffer,
            0,
            bytemuck::cast_slice(&[Uniforms {
                resolution: uvec2(config.width, config.height).as_vec2(),
                scale_factor: ctx.scale_factor as f32,
            }]),
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("latex"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });
        ctx.set_scissor_rect(&mut pass, bounds);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(&self.pipeline);
        pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        pass.draw_indexed(0..indices.len() as _, 0, 0..1);
    }
}
