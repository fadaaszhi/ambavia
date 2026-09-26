use serde::{Deserialize, Serialize};
use serde_repr::{Deserialize_repr, Serialize_repr};

// References:
// - https://github.com/DesModder/DesModder/blob/main/graph-state/state.ts
// - https://www.desmos.com/api

#[derive(Debug, Serialize, Deserialize)]
pub struct GraphState {
    pub hash: String,
    pub product: Product,
    #[serde(default, skip_serializing_if = "is_default")]
    pub title: String,
    pub state: State,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
pub enum Product {
    #[default]
    #[serde(rename = "graphing")]
    Graphing,
    #[serde(rename = "graphing-3d")]
    Graphing3D,
    #[serde(rename = "geometry-calculator")]
    GeometryCalculator,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct State {
    pub random_seed: String,
    pub graph: Graph,
    pub expressions: Expressions,
}

fn default_true() -> bool {
    true
}

fn is_true(value: &bool) -> bool {
    *value
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Graph {
    #[serde(default, skip_serializing_if = "is_default")]
    pub product: Product,

    pub viewport: Viewport,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub square_axes: bool,

    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub show_grid: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub show_x_axis: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub show_y_axis: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub x_axis_numbers: bool,
    #[serde(default = "default_true", skip_serializing_if = "is_true")]
    pub y_axis_numbers: bool,

    #[serde(default, skip_serializing_if = "is_default")]
    pub degree_mode: bool,
    #[serde(default, skip_serializing_if = "is_default")]
    pub three_d_mode: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Viewport {
    pub xmin: f64,
    pub ymin: f64,
    pub xmax: f64,
    pub ymax: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Expressions {
    pub list: Vec<ExpressionItem>,
}

pub type Id = String;

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExpressionItem {
    pub id: Id,
    #[serde(default, skip_serializing_if = "is_default")]
    pub folder_id: Option<Id>,
    #[serde(flatten)]
    pub kind: ExpressionItemKind,
}

pub type Latex = String;

#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
#[serde(rename_all = "camelCase")]
pub enum ExpressionItemKind {
    Expression(Expression),
    Image {},
    Text {
        #[serde(default, skip_serializing_if = "is_default")]
        text: String,
    },
    Table {},
    Folder {
        #[serde(default, skip_serializing_if = "is_default")]
        hidden: bool,
        #[serde(default, skip_serializing_if = "is_default")]
        collapsed: bool,
        #[serde(default, skip_serializing_if = "is_default")]
        title: String,
    },
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Expression {
    #[serde(default, skip_serializing_if = "is_default")]
    pub latex: Latex,
    #[serde(default, skip_serializing_if = "is_default")]
    pub hidden: bool,

    pub color: String,
    #[serde(default, skip_serializing_if = "is_default")]
    pub color_latex: Latex,

    #[serde(default, skip_serializing_if = "is_default")]
    pub points: Option<bool>,
    #[serde(default, skip_serializing_if = "is_default")]
    #[serde(alias = "__stashed_V12PointStyle")]
    pub point_style: PointStyle,
    #[serde(default, skip_serializing_if = "is_default")]
    pub point_opacity: Latex,
    #[serde(default, skip_serializing_if = "is_default")]
    pub point_size: Latex,

    #[serde(default, skip_serializing_if = "is_default")]
    pub lines: Option<bool>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub line_style: LineStyle,
    #[serde(default, skip_serializing_if = "is_default")]
    pub line_opacity: Latex,
    #[serde(default, skip_serializing_if = "is_default")]
    pub line_width: Latex,

    #[serde(default, skip_serializing_if = "is_default")]
    pub fill: Option<bool>,
    #[serde(default, skip_serializing_if = "is_default")]
    pub fill_opacity: Latex,

    #[serde(default, skip_serializing_if = "is_default")]
    pub drag_mode: DragMode,

    #[serde(default, skip_serializing_if = "is_default")]
    pub slider: Slider,
    #[serde(default, skip_serializing_if = "is_default")]
    pub parametric_domain: Domain,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DragMode {
    None,
    X,
    Y,
    XY,
    #[default]
    Auto,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LineStyle {
    #[default]
    Solid,
    Dashed,
    Dotted,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum PointStyle {
    #[default]
    Point,
    Open,
    Cross,
    Square,
    Plus,
    Triangle,
    Diamond,
    Star,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Slider {
    #[serde(default, skip_serializing_if = "is_default")]
    pub hard_min: bool,
    #[serde(default, skip_serializing_if = "is_default")]
    pub hard_max: bool,
    #[serde(default, skip_serializing_if = "is_default")]
    pub animation_period: AnimationPeriod,
    #[serde(default, skip_serializing_if = "is_default")]
    pub loop_mode: LoopMode,
    #[serde(default, skip_serializing_if = "is_default")]
    pub play_direction: PlayDirection,
    #[serde(default, skip_serializing_if = "is_default")]
    pub is_playing: bool,
    #[serde(default, skip_serializing_if = "is_default")]
    pub min: Latex,
    #[serde(default, skip_serializing_if = "is_default")]
    pub max: Latex,
    #[serde(default, skip_serializing_if = "is_default")]
    pub step: Latex,
}

fn is_default<T: Default + PartialEq>(t: &T) -> bool {
    t == &T::default()
}

/// Animation period in milliseconds.
#[derive(Debug, Serialize, Deserialize, PartialEq)]
pub struct AnimationPeriod(f64);

impl Default for AnimationPeriod {
    fn default() -> Self {
        Self(4000.0)
    }
}

impl AnimationPeriod {
    pub fn from_secs(secs: f64) -> AnimationPeriod {
        AnimationPeriod(secs * 1000.0)
    }

    pub fn as_secs(&self) -> f64 {
        self.0 / 1000.0
    }
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Domain {
    pub min: Latex,
    pub max: Latex,
}

#[derive(Debug, Serialize, Deserialize, Default, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum LoopMode {
    #[default]
    LoopForwardReverse,
    LoopForward,
    PlayOnce,
    PlayIndefinitely,
}

#[derive(Debug, Serialize_repr, Deserialize_repr, Default, PartialEq)]
#[repr(i8)]
pub enum PlayDirection {
    #[default]
    Forward = 1,
    Reverse = -1,
}
