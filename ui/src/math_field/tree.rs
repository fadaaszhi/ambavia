use std::{
    iter::zip,
    ops::{Deref, DerefMut},
};

use glam::{DVec2, dvec2};

use crate::katex_font::{Font, Glyph, get_glyph};
use parse::latex_tree::{self, Node as LNode};

const OPERATORNAME_SPACE: f64 = 0.17;
const BINOP_SPACE: f64 = 0.2;
const COMMA_SPACE: f64 = 0.15;
const COLON_SPACE: f64 = 0.2;
const BRACKET_JUT: f64 = 0.072;
const BRACKET_PADDING: f64 = 0.072;
const SCRIPT_LOWER_SCALE: f64 = 0.73;
const SCRIPT_UPPER_SCALE: f64 = 0.91;
const SCRIPT_MIDDLE: f64 = 0.02;
const RADICAL_ROOT_SCALE: f64 = 0.802;
const RADICAL_ROOT_MIDDLE: f64 = -0.526;
const RADICAL_ROOT_RIGHT: f64 = 0.485;
const RADICAL_INNER_TOP_PADDING: f64 = 0.073;
const RADICAL_LINE_JUT: f64 = 0.15;
const RADICAL_OUTER_TOP_PADDING: f64 = 0.073;
const RADICAL_OUTER_RIGHT_PADDING: f64 = 0.098;
const FRAC_SCALE: f64 = 0.91;
const FRAC_NUM_OFFSET: f64 = 0.0;
const FRAC_DEN_OFFSET: f64 = 0.079;
const FRAC_LINE_JUT: f64 = 0.09;
const FRAC_SIDE_PADDING: f64 = 0.18;
const FRAC_TOP_PADDING: f64 = 0.0;
const FRAC_BOTTOM_PADDING: f64 = 0.09;
const SUM_PROD_PADDING: f64 = 0.193;
const SUM_PROD_GLYPH_SCALE: f64 = 1.2;
const SUM_PROD_GLYPH_OFFSET_Y: f64 = -0.1;
const SUM_PROD_SUB_SUP_SCALE: f64 = 0.8;
const SUM_PROD_SUB_OFFSET: f64 = 0.152;
const SUM_PROD_SUP_OFFSET: f64 = 0.144;
const INT_SUB_SUP_SCALE: f64 = 0.8;
const INT_MIDDLE: f64 = 0.55;
const INT_SUB_POSITION: DVec2 = dvec2(0.55, 0.98);
const INT_SUP_POSITION: DVec2 = dvec2(1.04, -1.0);
const INT_RIGHT_PADDING: f64 = 0.1;
const CHAR_CENTER: f64 = -0.249;
const CHAR_HEIGHT: f64 = 0.526;
const CHAR_DEPTH: f64 = 0.462;
const EMPTY_WIDTH: f64 = 0.5;

// Reverse alphabetical order makes it so "tan" doesn't stop "tanh" from being chosen.
pub const OPERATORNAMES: &[&str] = &[
    "with", "width", "varp", "variance", "var", "unique", "tscore", "total", "tone", "tanh", "tan",
    "stdevp", "stdev", "stddevp", "stddev", "stdDevP", "stdDev", "spearman", "sort", "sinh", "sin",
    "signum", "sign", "shuffle", "sgn", "sech", "sec", "round", "rgb", "real", "random",
    "quartile", "quantile", "polygon", "nPr", "nCr", "mod", "min", "midpoint", "median", "mean",
    "mcm", "mcd", "max", "mad", "log", "ln", "length", "lcm", "join", "imag", "hsv", "height",
    "gcf", "gcd", "for", "floor", "exp", "erf", "distance", "csch", "csc", "covp", "cov", "count",
    "coth", "cot", "cosh", "cos", "corr", "conj", "ceil", "artanh", "arsinh", "arsech", "arg",
    "arctanh", "arctan", "arcsinh", "arcsin", "arcsech", "arcsec", "arcsch", "arcoth", "arcosh",
    "arccsch", "arccsc", "arccoth", "arccot", "arccosh", "arccos", "abs", "TScore",
];

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bracket {
    Paren,
    Square,
    Brace,
    Pipe,
}

impl From<char> for Bracket {
    fn from(value: char) -> Self {
        match value {
            '(' | ')' => Bracket::Paren,
            '[' | ']' => Bracket::Square,
            '{' | '}' => Bracket::Brace,
            '|' => Bracket::Pipe,
            _ => panic!("{value:?} is not a bracket"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BigOp {
    Sum,
    Prod,
    Int,
}

impl From<&str> for BigOp {
    fn from(value: &str) -> Self {
        match value {
            "sum" => BigOp::Sum,
            "prod" => BigOp::Prod,
            "int" => BigOp::Int,
            _ => panic!("{value:?} is not a big operator"),
        }
    }
}

#[derive(Debug, Default)]
pub struct TexturedQuad {
    advance: f64,
    pub position: DVec2,
    pub size: DVec2,
    pub uv0: DVec2,
    pub uv1: DVec2,
    pub gray: bool,
}

impl TexturedQuad {
    fn transform(&mut self, position: DVec2, scale: f64) {
        self.advance *= scale;
        self.position = position + scale * self.position;
        self.size *= scale;
    }
}

#[derive(Debug)]
pub struct Bounds {
    pub position: DVec2,
    pub width: f64,
    pub height: f64,
    pub depth: f64,
    pub scale: f64,
}

impl Default for Bounds {
    fn default() -> Self {
        Bounds {
            position: DVec2::ZERO,
            width: 0.0,
            height: CHAR_HEIGHT,
            depth: CHAR_DEPTH,
            scale: 1.0,
        }
    }
}

impl Bounds {
    pub fn left(&self) -> f64 {
        self.position.x
    }

    pub fn top(&self) -> f64 {
        self.position.y - self.height
    }

    pub fn right(&self) -> f64 {
        self.position.x + self.width
    }

    pub fn bottom(&self) -> f64 {
        self.position.y + self.depth
    }

    pub fn top_left(&self) -> DVec2 {
        dvec2(self.left(), self.top())
    }

    pub fn bottom_right(&self) -> DVec2 {
        dvec2(self.right(), self.bottom())
    }

    fn transform(&mut self, position: DVec2, scale: f64) -> (DVec2, f64) {
        self.width *= scale;
        self.height *= scale;
        self.depth *= scale;
        self.scale *= scale;
        self.position = position + scale * self.position;
        (self.position, self.scale)
    }

    fn scale(&mut self, scale: f64) {
        self.width *= scale;
        self.height *= scale;
        self.depth *= scale;
        self.scale *= scale;
    }

    fn union(&mut self, other: &Bounds) {
        assert_eq!(self.position, DVec2::ZERO);
        assert_eq!(self.scale, 1.0);
        self.width = (other.width + other.position.x).max(self.width);
        self.height = (other.height - other.position.y).max(self.height);
        self.depth = (other.depth + other.position.y).max(self.depth);
    }
}

#[derive(Debug, Default)]
pub struct Tree {
    pub bounds: Bounds,
    pub nodes: Vec<(Bounds, Node)>,
    pub has_gray_background: bool,
}

impl From<Vec<(Bounds, Node)>> for Tree {
    fn from(nodes: Vec<(Bounds, Node)>) -> Self {
        Self {
            nodes,
            ..Default::default()
        }
    }
}

#[derive(Debug, Default)]
pub struct Line {
    pub x_min: f64,
    pub x_max: f64,
    pub y: f64,
}

impl Line {
    fn transform(&mut self, position: DVec2, scale: f64) {
        self.x_min = position.x + scale * self.x_min;
        self.x_max = position.x + scale * self.x_max;
        self.y = position.y + scale * self.y;
    }
}

#[derive(Debug)]
pub enum Node {
    Bracket {
        left: Option<Bracket>,
        right: Option<Bracket>,
        inner: Tree,
        left_quad: TexturedQuad,
        right_quad: TexturedQuad,
    },
    Script {
        lower: Option<Tree>,
        upper: Option<Tree>,
    },
    Radical {
        root: Option<Tree>,
        arg: Tree,
        radical: TexturedQuad,
        line: Line,
    },
    Frac {
        num: Tree,
        den: Tree,
        line: Line,
    },
    BigOp {
        op: BigOp,
        lower: Tree,
        upper: Tree,
        op_quad: TexturedQuad,
    },
    Char {
        ch: char,
        quad: TexturedQuad,
    },
}

pub fn ends_in_operatorname(nodes: &[(Bounds, Node)]) -> bool {
    OPERATORNAMES.iter().any(|name| {
        let count = name.chars().count();
        nodes.len() >= count
            && zip(name.chars(), &nodes[nodes.len() - count..]).all(|(c, (_, n))| n.is_char(c))
    })
}

impl Node {
    pub fn is_char(&self, c: char) -> bool {
        match self {
            Node::Char { ch, .. } => *ch == c,
            _ => false,
        }
    }
}

pub fn new_bracket(
    left: impl Into<Option<Bracket>>,
    right: impl Into<Option<Bracket>>,
    inner: impl Into<Tree>,
) -> Node {
    Node::Bracket {
        left: left.into(),
        right: right.into(),
        inner: inner.into(),
        left_quad: Default::default(),
        right_quad: Default::default(),
    }
}

pub fn new_script_lower(lower: impl Into<Tree>) -> Node {
    Node::Script {
        lower: Some(lower.into()),
        upper: None,
    }
}

pub fn new_script_upper(upper: impl Into<Tree>) -> Node {
    Node::Script {
        lower: None,
        upper: Some(upper.into()),
    }
}

pub fn new_script(lower: Option<impl Into<Tree>>, upper: Option<impl Into<Tree>>) -> Node {
    Node::Script {
        lower: lower.map(Into::into),
        upper: upper.map(Into::into),
    }
}

pub fn new_sqrt(arg: impl Into<Tree>) -> Node {
    Node::Radical {
        root: None,
        arg: arg.into(),
        radical: Default::default(),
        line: Default::default(),
    }
}

pub fn new_radical(root: Option<impl Into<Tree>>, arg: impl Into<Tree>) -> Node {
    Node::Radical {
        root: root.map(Into::into),
        arg: arg.into(),
        radical: Default::default(),
        line: Default::default(),
    }
}

pub fn new_frac(num: impl Into<Tree>, den: impl Into<Tree>) -> Node {
    Node::Frac {
        num: num.into(),
        den: den.into(),
        line: Default::default(),
    }
}

pub fn new_big_op(op: impl Into<BigOp>, lower: impl Into<Tree>, upper: impl Into<Tree>) -> Node {
    Node::BigOp {
        op: op.into(),
        lower: lower.into(),
        upper: upper.into(),
        op_quad: Default::default(),
    }
}

pub fn new_char(ch: char) -> Node {
    Node::Char {
        ch,
        quad: Default::default(),
    }
}

pub fn to_latex(tree: &[(Bounds, Node)]) -> latex_tree::Nodes<'static> {
    let mut nodes = vec![];
    let mut i = 0;

    'outer: while i < tree.len() {
        for &name in OPERATORNAMES {
            let count = name.chars().count();
            if tree.len() - i >= count
                && zip(name.chars(), &tree[i..]).all(|(c, (_, n))| n.is_char(c))
            {
                nodes.push(LNode::Operatorname(name.chars().map(LNode::Char).collect()));
                i += count;
                continue 'outer;
            }
        }

        let node = &tree[i].1;
        i += 1;

        match node {
            Node::Bracket {
                left, right, inner, ..
            } => {
                let (left, right) = (
                    match left.or(*right).unwrap() {
                        Bracket::Paren => '(',
                        Bracket::Square => '[',
                        Bracket::Brace => '{',
                        Bracket::Pipe => '|',
                    },
                    match right.or(*left).unwrap() {
                        Bracket::Paren => ')',
                        Bracket::Square => ']',
                        Bracket::Brace => '}',
                        Bracket::Pipe => '|',
                    },
                );
                let mut inner = to_latex(inner);
                inner.insert(0, LNode::Char(left));
                inner.push(LNode::Char(right));
                nodes.push(LNode::DelimitedGroup(inner));
            }
            Node::Script { lower, upper } => {
                nodes.push(LNode::SubSup {
                    sub: lower.as_ref().map(|x| to_latex(x)),
                    sup: upper.as_ref().map(|x| to_latex(x)),
                });
            }
            Node::Radical { root, arg, .. } => {
                nodes.push(LNode::Sqrt {
                    root: root.as_ref().map(|x| to_latex(x)),
                    arg: to_latex(arg),
                });
            }
            Node::Frac { num, den, .. } => {
                nodes.push(LNode::Frac {
                    num: to_latex(num),
                    den: to_latex(den),
                });
            }
            Node::BigOp {
                op, lower, upper, ..
            } => {
                let op = match op {
                    BigOp::Sum => "sum",
                    BigOp::Prod => "prod",
                    BigOp::Int => "int",
                };
                nodes.push(LNode::CtrlSeq(op));
                nodes.push(LNode::SubSup {
                    sub: Some(to_latex(lower)),
                    sup: Some(to_latex(upper)),
                });
            }
            Node::Char { ch, .. } => {
                let seq = match ch {
                    '×' => "times",
                    '÷' => "div",
                    'Γ' => "Gamma",
                    'Δ' => "Delta",
                    'Θ' => "Theta",
                    'Λ' => "Lambda",
                    'Ξ' => "Xi",
                    'Π' => "Pi",
                    'Σ' => "Sigma",
                    'Υ' => "Upsilon",
                    'Φ' => "Phi",
                    'Ψ' => "Psi",
                    'Ω' => "Omega",
                    'α' => "alpha",
                    'β' => "beta",
                    'γ' => "gamma",
                    'δ' => "delta",
                    'ε' => "varepsilon",
                    'ζ' => "zeta",
                    'η' => "eta",
                    'θ' => "theta",
                    'ι' => "iota",
                    'κ' => "kappa",
                    'λ' => "lambda",
                    'μ' => "mu",
                    'ν' => "nu",
                    'ξ' => "xi",
                    'π' => "pi",
                    'ρ' => "rho",
                    'ς' => "varsigma",
                    'σ' => "sigma",
                    'τ' => "tau",
                    'υ' => "upsilon",
                    'φ' => "varphi",
                    'χ' => "chi",
                    'ψ' => "psi",
                    'ω' => "omega",
                    'ϑ' => "vartheta",
                    'ϕ' => "phi",
                    'ϖ' => "varpi",
                    'ϱ' => "varrho",
                    'ϵ' => "epsilon",
                    '→' => "to",
                    '∞' => "infty",
                    '≤' => "le",
                    '≥' => "ge",
                    '⋅' => "cdot",
                    _ => "",
                };

                nodes.push(if seq.is_empty() {
                    LNode::Char(*ch)
                } else {
                    LNode::CtrlSeq(seq)
                })
            }
        }
    }

    nodes
}

impl From<&Vec<LNode<'_>>> for Tree {
    fn from(value: &Vec<LNode<'_>>) -> Self {
        value.as_slice().into()
    }
}

impl From<&[LNode<'_>]> for Tree {
    fn from(lnodes: &[LNode]) -> Self {
        let mut nodes = vec![];
        let mut lnodes = lnodes.iter().peekable();
        while let Some(node) = lnodes.next() {
            let node = match node {
                LNode::DelimitedGroup(nodes) => {
                    let [LNode::Char(left), inner @ .., LNode::Char(right)] = &nodes[..] else {
                        unreachable!();
                    };
                    new_bracket(Bracket::from(*left), Bracket::from(*right), inner)
                }
                LNode::SubSup { sub, sup } => new_script(sub.as_ref(), sup.as_ref()),
                LNode::Sqrt { root, arg } => new_radical(root.as_ref(), arg),
                LNode::Frac { num, den } => new_frac(num, den),
                LNode::Operatorname(inner) => {
                    nodes.extend(Tree::from(inner).nodes);
                    continue;
                }
                LNode::CtrlSeq(op @ ("sum" | "prod" | "int")) => {
                    let (lower, upper) = if let Some(LNode::SubSup { sub, sup }) = lnodes.peek() {
                        lnodes.next();
                        (sub.as_ref(), sup.as_ref())
                    } else {
                        (None, None)
                    };
                    new_big_op(*op, lower.unwrap_or(&vec![]), upper.unwrap_or(&vec![]))
                }
                LNode::CtrlSeq(seq) => Node::Char {
                    ch: match *seq {
                        " " => ' ',
                        "times" => '×',
                        "div" => '÷',
                        "Gamma" => 'Γ',
                        "Delta" => 'Δ',
                        "Theta" => 'Θ',
                        "Lambda" => 'Λ',
                        "Xi" => 'Ξ',
                        "Pi" => 'Π',
                        "Sigma" => 'Σ',
                        "Upsilon" => 'Υ',
                        "Phi" => 'Φ',
                        "Psi" => 'Ψ',
                        "Omega" => 'Ω',
                        "alpha" => 'α',
                        "beta" => 'β',
                        "gamma" => 'γ',
                        "delta" => 'δ',
                        "varepsilon" => 'ε',
                        "zeta" => 'ζ',
                        "eta" => 'η',
                        "theta" => 'θ',
                        "iota" => 'ι',
                        "kappa" => 'κ',
                        "lambda" => 'λ',
                        "mu" => 'μ',
                        "nu" => 'ν',
                        "xi" => 'ξ',
                        "pi" => 'π',
                        "rho" => 'ρ',
                        "varsigma" => 'ς',
                        "sigma" => 'σ',
                        "tau" => 'τ',
                        "upsilon" => 'υ',
                        "varphi" => 'φ',
                        "chi" => 'χ',
                        "psi" => 'ψ',
                        "omega" => 'ω',
                        "vartheta" => 'ϑ',
                        "phi" => 'ϕ',
                        "varpi" => 'ϖ',
                        "varrho" => 'ϱ',
                        "epsilon" => 'ϵ',
                        "to" => '→',
                        "infty" => '∞',
                        "lt" => '<',
                        "gt" => '>',
                        "le" => '≤',
                        "ge" => '≥',
                        "lte" => '≤',
                        "gte" => '≥',
                        "cdot" => '⋅',
                        _ => {
                            nodes.extend(seq.chars().map(|ch| (Bounds::default(), new_char(ch))));
                            continue;
                        }
                    },
                    quad: Default::default(),
                },
                LNode::Char(c) => new_char(*c),
            };
            nodes.push((Bounds::default(), node));
        }
        Tree {
            nodes,
            ..Default::default()
        }
    }
}

fn get_quad(font: Font, ch: char) -> TexturedQuad {
    let Glyph {
        advance,
        plane: p,
        atlas: a,
    } = get_glyph(font, ch);
    TexturedQuad {
        advance,
        position: dvec2(p.left, p.top - CHAR_CENTER),
        size: dvec2(p.right - p.left, p.bottom - p.top),
        uv0: dvec2(a.left, a.top),
        uv1: dvec2(a.right, a.bottom),
        gray: false,
    }
}

impl Deref for Tree {
    type Target = Vec<(Bounds, Node)>;

    fn deref(&self) -> &Self::Target {
        &self.nodes
    }
}

impl DerefMut for Tree {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.nodes
    }
}

impl Tree {
    pub fn insert(&mut self, index: usize, node: Node) {
        self.nodes.insert(index, (Bounds::default(), node));
    }

    pub fn push(&mut self, node: Node) {
        self.nodes.push((Bounds::default(), node));
    }

    fn layout_relative(&mut self) {
        self.bounds = Bounds::default();
        self.has_gray_background = false;

        if self.nodes.is_empty() {
            self.bounds.width = EMPTY_WIDTH;
            self.has_gray_background = true;
            return;
        }

        let mut i = 0;
        let mut previous_was_operatorname = false;

        fn add_space_after_operatorname(next_node: &Node) -> bool {
            match next_node {
                Node::Bracket { .. }
                | Node::Script { .. }
                | Node::BigOp { .. }
                | Node::Char {
                    ch:
                        '.' | '+' | '-' | '*' | '=' | '<' | '>' | ',' | ':' | '×' | '÷' | '→' | '⋅',
                    ..
                } => false,
                _ => true,
            }
        }

        'outer: while i < self.nodes.len() {
            for &name in OPERATORNAMES {
                let count = name.chars().count();
                if self.nodes.len() - i >= count
                    && zip(name.chars(), &self.nodes[i..]).all(|(c, (_, n))| n.is_char(c))
                {
                    let add_space_before = i > 0
                        && !previous_was_operatorname
                        && match &self.nodes[i - 1].1 {
                            Node::BigOp { .. }
                            | Node::Char {
                                ch:
                                    '.' | '+' | '-' | '*' | '=' | '<' | '>' | ',' | ':' | '×' | '÷'
                                    | '→' | '⋅',
                                ..
                            } => false,
                            _ => true,
                        };
                    let add_space_after = i + count < self.nodes.len()
                        && add_space_after_operatorname(&self.nodes[i + count].1);

                    for (j, (c, (bounds, node))) in
                        zip(name.chars(), &mut self.nodes[i..]).enumerate()
                    {
                        let Node::Char { quad, .. } = node else {
                            unreachable!()
                        };

                        *quad = get_quad(Font::MainRegular, c);
                        if j == 0 && add_space_before {
                            quad.advance += OPERATORNAME_SPACE;
                            quad.position.x += OPERATORNAME_SPACE;
                        }
                        if j == count - 1 && add_space_after {
                            quad.advance += OPERATORNAME_SPACE;
                        }

                        *bounds = Bounds {
                            width: quad.advance,
                            height: CHAR_HEIGHT,
                            depth: CHAR_DEPTH,
                            ..Default::default()
                        };
                        bounds.position.x += self.bounds.width;
                        self.bounds.union(bounds);
                    }

                    i += count;
                    previous_was_operatorname = true;
                    continue 'outer;
                }
            }

            let previous_was_operand = i > 0
                && !matches!(
                    &self.nodes[i - 1].1,
                    Node::Char {
                        ch: '.'
                            | '+'
                            | '-'
                            | '*'
                            | '='
                            | '<'
                            | '>'
                            | '≤'
                            | '≥'
                            | ','
                            | ':'
                            | '×'
                            | '÷'
                            | '→'
                            | '⋅',
                        ..
                    },
                );
            let previous_operatorname_requires_space = previous_was_operatorname
                && i + 1 < self.nodes.len()
                && add_space_after_operatorname(&self.nodes[i + 1].1);
            let (bounds, node) = &mut self.nodes[i];
            *bounds = Bounds::default();

            match node {
                Node::Bracket {
                    left,
                    right,
                    inner,
                    left_quad,
                    right_quad,
                } => {
                    *left_quad = match left.or(*right).unwrap() {
                        Bracket::Paren => get_quad(Font::Size1Regular, '('),
                        Bracket::Square => get_quad(Font::Size1Regular, '['),
                        Bracket::Brace => get_quad(Font::Size1Regular, '{'),
                        Bracket::Pipe => get_quad(Font::MainRegular, '|'),
                    };
                    *right_quad = match right.or(*left).unwrap() {
                        Bracket::Paren => get_quad(Font::Size1Regular, ')'),
                        Bracket::Square => get_quad(Font::Size1Regular, ']'),
                        Bracket::Brace => get_quad(Font::Size1Regular, '}'),
                        Bracket::Pipe => get_quad(Font::MainRegular, '|'),
                    };
                    left_quad.gray = left.is_none();
                    right_quad.gray = right.is_none();
                    inner.layout_relative();
                    inner.has_gray_background = false;
                    inner.bounds.position.x = left_quad.advance;
                    let top = inner.bounds.top() - BRACKET_JUT;
                    let bottom = inner.bounds.bottom() + BRACKET_JUT;
                    left_quad.position.y = top;
                    right_quad.position.y = top;
                    left_quad.size.y = bottom - top;
                    right_quad.size.y = bottom - top;
                    right_quad.position.x += inner.bounds.right();
                    bounds.width = inner.bounds.right() + right_quad.advance;
                    bounds.height = -top + BRACKET_PADDING;
                    bounds.depth = bottom + BRACKET_PADDING;
                }
                Node::Script { lower, upper } => {
                    if let Some(lower) = lower {
                        lower.layout_relative();
                        lower.bounds.scale(SCRIPT_LOWER_SCALE);
                        lower.bounds.position.y = SCRIPT_MIDDLE + lower.bounds.height;
                        bounds.union(&lower.bounds);
                    }
                    if let Some(upper) = upper {
                        upper.layout_relative();
                        upper.bounds.scale(SCRIPT_UPPER_SCALE);
                        upper.bounds.position.y = SCRIPT_MIDDLE - upper.bounds.depth;
                        bounds.union(&upper.bounds);
                    }
                    if previous_operatorname_requires_space {
                        bounds.width += OPERATORNAME_SPACE;
                    }
                }
                Node::Radical {
                    root,
                    arg,
                    radical,
                    line,
                } => {
                    arg.layout_relative();
                    *radical = get_quad(Font::Size1Regular, '√');
                    if let Some(root) = root {
                        root.layout_relative();
                        root.bounds.scale(RADICAL_ROOT_SCALE);
                        root.bounds.position.y = RADICAL_ROOT_MIDDLE;
                        let offset = (root.bounds.right() - RADICAL_ROOT_RIGHT).max(0.0);
                        root.bounds.position.x = offset + RADICAL_ROOT_RIGHT - root.bounds.width;
                        radical.position.x += offset;
                        radical.advance += offset;
                        bounds.union(&root.bounds);
                    };
                    // The glyphs have padding on top but we want the true top.
                    // The magic value was found manually by measuring.
                    radical.uv0.y += (radical.uv1.y - radical.uv0.y) * 0.02759835584;
                    radical.position.y = arg.bounds.top() - RADICAL_INNER_TOP_PADDING;
                    radical.size.y = arg.bounds.bottom() - radical.position.y;
                    arg.bounds.position.x = radical.advance + RADICAL_LINE_JUT;
                    line.x_min = radical.advance;
                    line.x_max = arg.bounds.right() + RADICAL_LINE_JUT;
                    line.y = radical.position.y;
                    bounds.union(&Bounds {
                        width: line.x_max + RADICAL_OUTER_RIGHT_PADDING,
                        height: -line.y + RADICAL_OUTER_TOP_PADDING,
                        depth: arg.bounds.bottom(),
                        ..Default::default()
                    });
                }
                Node::Frac { num, den, line } => {
                    num.layout_relative();
                    den.layout_relative();
                    num.bounds.scale(FRAC_SCALE);
                    den.bounds.scale(FRAC_SCALE);
                    let max_width = num.bounds.width.max(den.bounds.width);
                    line.x_min = FRAC_SIDE_PADDING;
                    line.x_max = FRAC_SIDE_PADDING + 2.0 * FRAC_LINE_JUT + max_width;
                    line.y = 0.0;
                    bounds.width = max_width + (FRAC_SIDE_PADDING + FRAC_LINE_JUT) * 2.0;
                    bounds.height =
                        num.bounds.height + num.bounds.depth + FRAC_NUM_OFFSET + FRAC_TOP_PADDING;
                    bounds.depth = den.bounds.height
                        + den.bounds.depth
                        + FRAC_DEN_OFFSET
                        + FRAC_BOTTOM_PADDING;
                    num.bounds.position = dvec2(
                        (bounds.width - num.bounds.width) / 2.0,
                        -num.bounds.depth - FRAC_NUM_OFFSET,
                    );
                    den.bounds.position = dvec2(
                        (bounds.width - den.bounds.width) / 2.0,
                        den.bounds.height + FRAC_DEN_OFFSET,
                    );
                }
                Node::BigOp {
                    op,
                    lower,
                    upper,
                    op_quad,
                } => {
                    let c = match op {
                        BigOp::Sum => '∑',
                        BigOp::Prod => '∏',
                        BigOp::Int => '∫',
                    };
                    *op_quad = get_quad(Font::Size2Regular, c);
                    lower.layout_relative();
                    upper.layout_relative();
                    if *op == BigOp::Int {
                        lower.bounds.scale(INT_SUB_SUP_SCALE);
                        upper.bounds.scale(INT_SUB_SUP_SCALE);
                        lower.bounds.position = INT_SUB_POSITION;
                        upper.bounds.position = INT_SUP_POSITION;
                        lower.bounds.position.y = lower
                            .bounds
                            .position
                            .y
                            .max(INT_MIDDLE + lower.bounds.height);
                        upper.bounds.position.y =
                            upper.bounds.position.y.min(INT_MIDDLE - upper.bounds.depth);
                        bounds.width =
                            lower.bounds.right().max(upper.bounds.right()) + INT_RIGHT_PADDING;
                        bounds.height = -upper.bounds.top();
                        bounds.depth = lower.bounds.bottom();
                    } else {
                        op_quad.advance *= SUM_PROD_GLYPH_SCALE;
                        op_quad.position *= SUM_PROD_GLYPH_SCALE;
                        op_quad.size *= SUM_PROD_GLYPH_SCALE;
                        op_quad.position.y += SUM_PROD_GLYPH_OFFSET_Y;
                        lower.bounds.scale(SUM_PROD_SUB_SUP_SCALE);
                        upper.bounds.scale(SUM_PROD_SUB_SUP_SCALE);
                        bounds.width = op_quad.advance.max(
                            lower.bounds.width.max(upper.bounds.width) + 2.0 * SUM_PROD_PADDING,
                        );
                        op_quad.position.x += (bounds.width - op_quad.advance) / 2.0;
                        lower.bounds.position.x = (bounds.width - lower.bounds.width) / 2.0;
                        upper.bounds.position.x = (bounds.width - upper.bounds.width) / 2.0;
                        lower.bounds.position.y = op_quad.position.y
                            + op_quad.size.y
                            + SUM_PROD_SUB_OFFSET
                            + lower.bounds.height;
                        upper.bounds.position.y =
                            op_quad.position.y - SUM_PROD_SUP_OFFSET - upper.bounds.depth;
                        bounds.height = -upper.bounds.top() + SUM_PROD_PADDING;
                        bounds.depth = lower.bounds.bottom() + SUM_PROD_PADDING;
                    }
                }
                Node::Char { ch, quad } => {
                    let (space_before, space_after) = match *ch {
                        '+' | '-' if previous_was_operand => (BINOP_SPACE, BINOP_SPACE),
                        '*' | '=' | '<' | '>' | '≤' | '≥' | '×' | '÷' | '→' | '⋅' => {
                            (BINOP_SPACE, BINOP_SPACE)
                        }
                        ',' => (0.0, COMMA_SPACE),
                        ':' => (0.0, COLON_SPACE),
                        _ => (0.0, 0.0),
                    };
                    let c = match *ch {
                        '-' => '−',
                        '\'' => '′',
                        c => c,
                    };
                    let font = match c {
                        'A'..='Z' | 'a'..='z' | 'α'..='ω' | 'ϑ' | 'ϕ' | 'ϖ' | 'ϱ' | 'ϵ' => {
                            Font::MathItalic
                        }
                        _ => Font::MainRegular,
                    };
                    *quad = get_quad(font, c);
                    quad.position.x += space_before;
                    quad.advance += space_before + space_after;
                    bounds.width = quad.advance;
                    bounds.height = CHAR_HEIGHT;
                    bounds.depth = CHAR_DEPTH;
                }
            }

            bounds.position.x += self.bounds.width;
            self.bounds.union(bounds);
            i += 1;
            previous_was_operatorname = false;
        }
    }

    fn make_absolute(&mut self, position: DVec2, scale: f64) {
        let (position, scale) = self.bounds.transform(position, scale);
        for (bounds, node) in &mut self.nodes {
            let (position, scale) = bounds.transform(position, scale);
            match node {
                Node::Bracket {
                    inner,
                    left_quad,
                    right_quad,
                    ..
                } => {
                    left_quad.transform(position, scale);
                    right_quad.transform(position, scale);
                    inner.make_absolute(position, scale);
                }
                Node::Script { lower, upper } => {
                    if let Some(lower) = lower {
                        lower.make_absolute(position, scale);
                    }
                    if let Some(upper) = upper {
                        upper.make_absolute(position, scale);
                    }
                }
                Node::Radical {
                    root,
                    arg,
                    radical,
                    line,
                } => {
                    if let Some(root) = root {
                        root.make_absolute(position, scale);
                    }
                    arg.make_absolute(position, scale);
                    radical.transform(position, scale);
                    line.transform(position, scale);
                }
                Node::Frac { num, den, line } => {
                    num.make_absolute(position, scale);
                    den.make_absolute(position, scale);
                    line.transform(position, scale);
                }
                Node::BigOp {
                    lower,
                    upper,
                    op_quad,
                    ..
                } => {
                    lower.make_absolute(position, scale);
                    upper.make_absolute(position, scale);
                    op_quad.transform(position, scale);
                }
                Node::Char { quad, .. } => quad.transform(position, scale),
            }
        }
    }

    pub fn layout(&mut self) {
        self.layout_relative();
        self.has_gray_background = false;
        self.make_absolute(DVec2::ZERO, 1.0);
    }
}
