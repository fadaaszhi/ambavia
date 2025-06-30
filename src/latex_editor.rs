#[derive(Debug, PartialEq)]
pub enum BracketKind {
    Paren,
    Bracket,
    Brace,
    Pipe,
}

impl From<char> for BracketKind {
    fn from(value: char) -> Self {
        match value {
            '(' | ')' => BracketKind::Paren,
            '[' | ']' => BracketKind::Bracket,
            '{' | '}' => BracketKind::Brace,
            '|' => BracketKind::Pipe,
            _ => panic!("'{value}' is not a bracket"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum SumProdKind {
    Sum,
    Prod,
}

// Reverse alphabetical order makes it so "tan" doesn't stop "tanh" from being chosen.
const OPERATORNAMES: &[&str] = &[
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

pub mod editor {
    use std::fmt::Write;

    use ambavia::latex_tree::Node as LNode;

    use super::*;

    pub type Nodes = Vec<Node>;

    #[derive(Debug, PartialEq)]
    pub enum Node {
        DelimitedGroup {
            left: BracketKind,
            right: BracketKind,
            inner: Nodes,
        },
        SubSup {
            sub: Option<Nodes>,
            sup: Option<Nodes>,
        },
        Sqrt {
            root: Option<Nodes>,
            arg: Nodes,
        },
        Frac {
            num: Nodes,
            den: Nodes,
        },
        SumProd {
            kind: SumProdKind,
            sub: Nodes,
            sup: Nodes,
        },
        Char(char),
    }

    pub fn to_latex(tree: &[Node]) -> String {
        let mut latex = String::new();
        let l = &mut latex;
        let mut i = 0;

        'outer: while i < tree.len() {
            for &name in OPERATORNAMES {
                let count = name.chars().count();
                if tree.len() - i >= count
                    && name
                        .chars()
                        .zip(&tree[i..])
                        .all(|(c, n)| *n == Node::Char(c))
                {
                    write!(l, r"\operatorname{{{name}}}").unwrap();
                    i += count;
                    continue 'outer;
                }
            }

            let node = &tree[i];
            i += 1;

            match node {
                Node::DelimitedGroup { left, right, inner } => {
                    let left = match left {
                        BracketKind::Paren => '(',
                        BracketKind::Bracket => '[',
                        BracketKind::Brace => '{',
                        BracketKind::Pipe => '|',
                    };
                    let right = match right {
                        BracketKind::Paren => ')',
                        BracketKind::Bracket => ']',
                        BracketKind::Brace => '}',
                        BracketKind::Pipe => '|',
                    };
                    let inner = to_latex(inner);
                    write!(l, r"\left{left}{inner}\right{right}").unwrap();
                }
                Node::SubSup { sub, sup } => {
                    if let Some(sub) = sub {
                        write!(l, r"_{{{}}}", to_latex(sub)).unwrap();
                    }
                    if let Some(sup) = sup {
                        write!(l, r"^{{{}}}", to_latex(sup)).unwrap();
                    }
                }
                Node::Sqrt { root, arg } => {
                    write!(l, r"\sqrt").unwrap();
                    if let Some(root) = root {
                        write!(l, r"[{}]", to_latex(root)).unwrap();
                    }
                    write!(l, r"{{{}}}", to_latex(arg)).unwrap();
                }
                Node::Frac { num, den } => {
                    write!(l, r"\frac{{{}}}{{{}}}", to_latex(num), to_latex(den)).unwrap();
                }
                Node::SumProd { kind, sub, sup } => {
                    let kind = match kind {
                        SumProdKind::Sum => r"sum",
                        SumProdKind::Prod => r"prod",
                    };
                    write!(l, r"\{kind}_{{{}}}^{{{}}}", to_latex(sub), to_latex(sup)).unwrap();
                }
                Node::Char(c) => {
                    let seq = match c {
                        ' ' => " ",
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

                    if seq.is_empty() {
                        write!(l, r"{}", c).unwrap();
                    } else {
                        write!(l, r"\{} ", seq).unwrap();
                    }
                }
            }
        }

        latex
    }

    pub fn convert(tree: &[LNode]) -> Nodes {
        let mut nodes = vec![];
        for node in tree {
            let node = match node {
                LNode::DelimitedGroup(nodes) => {
                    let [LNode::Char(left), inner @ .., LNode::Char(right)] = &nodes[..] else {
                        unreachable!();
                    };
                    let left = BracketKind::from(*left);
                    let right = BracketKind::from(*right);
                    let inner = convert(inner);
                    Node::DelimitedGroup { left, right, inner }
                }
                LNode::SubSup { sub, sup } => Node::SubSup {
                    sub: sub.as_ref().map(|x| convert(x)),
                    sup: sup.as_ref().map(|x| convert(x)),
                },
                LNode::Sqrt { root, arg } => Node::Sqrt {
                    root: root.as_ref().map(|x| convert(x)),
                    arg: convert(arg),
                },
                LNode::Frac { num, den } => Node::Frac {
                    num: convert(num),
                    den: convert(den),
                },
                LNode::Operatorname(inner) => {
                    nodes.append(&mut convert(inner));
                    continue;
                }
                LNode::CtrlSeq(seq) => Node::Char(match *seq {
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
                        nodes.extend(seq.chars().map(Node::Char));
                        continue;
                    }
                }),
                LNode::Char(c) => Node::Char(*c),
            };
            nodes.push(node);
        }
        nodes
    }
}

pub mod layout {
    use glam::{dvec2, DVec2};

    use crate::katex_font::{get_glyph, Font, Glyph};

    use super::{editor::Node as ENode, *};

    const OPERATORNAME_SPACE: f64 = 0.17;
    const BINOP_SPACE: f64 = 0.2;
    const COMMA_SPACE: f64 = 0.2;
    const COLON_SPACE: f64 = 0.2;
    const CHAR_CENTER: f64 = -0.249;
    const CHAR_HEIGHT: f64 = 0.526;
    const CHAR_DEPTH: f64 = 0.462;

    #[derive(Debug, Clone, Copy)]
    pub struct Bounds {
        pub width: f64,
        pub height: f64,
        pub depth: f64,
        pub scale: f64,
        pub position: DVec2,
    }

    impl Default for Bounds {
        fn default() -> Self {
            Bounds {
                width: 0.0,
                height: CHAR_HEIGHT,
                depth: CHAR_DEPTH,
                scale: 1.0,
                position: DVec2::ZERO,
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

        fn transform(&mut self, position: DVec2, scale: f64) {
            self.width *= scale;
            self.height *= scale;
            self.depth *= scale;
            self.scale *= scale;
            self.position = position + scale * self.position;
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
    pub struct Nodes {
        pub bounds: Bounds,
        pub nodes: Vec<(Bounds, Node)>,
    }

    impl Nodes {
        fn push(&mut self, mut bounds: Bounds, node: Node) {
            bounds.position.x += self.bounds.width;
            self.bounds.union(&bounds);
            self.nodes.push((bounds, node));
        }
    }

    #[derive(Debug)]
    pub enum Node {
        DelimitedGroup {
            left: BracketKind,
            right: BracketKind,
            inner: Nodes,
        },
        SubSup {
            sub: Option<Nodes>,
            sup: Option<Nodes>,
        },
        Sqrt {
            root: Option<Nodes>,
            arg: Nodes,
        },
        Frac {
            num: Nodes,
            den: Nodes,
        },
        SumProd {
            kind: SumProdKind,
            sub: Nodes,
            sup: Nodes,
            body: Nodes,
        },
        Char(Glyph),
    }

    fn layout_relative(tree: &[ENode]) -> Nodes {
        let mut nodes = Nodes::default();
        let mut i = 0;

        'outer: while i < tree.len() {
            for &name in OPERATORNAMES {
                let count = name.chars().count();
                if tree.len() - i >= count
                    && name
                        .chars()
                        .zip(&tree[i..])
                        .all(|(c, n)| *n == ENode::Char(c))
                {
                    for (j, c) in name.chars().enumerate() {
                        let mut g = get_glyph(Font::MainRegular, c);
                        g.plane.top -= CHAR_CENTER;
                        g.plane.bottom -= CHAR_CENTER;

                        if j == 0
                            && i > 0
                            && match &tree[i - 1] {
                                ENode::DelimitedGroup { .. } => true,
                                ENode::SubSup { .. } => true,
                                ENode::Sqrt { .. } => true,
                                ENode::Frac { .. } => true,
                                ENode::SumProd { .. } => false,
                                ENode::Char(
                                    '.' | '+' | '-' | '*' | '=' | '<' | '>' | ',' | ':' | '×' | '÷'
                                    | '→' | '⋅',
                                ) => false,
                                ENode::Char(_) => true,
                            }
                        {
                            g.plane.left += OPERATORNAME_SPACE;
                            g.plane.right += OPERATORNAME_SPACE;
                            g.advance += OPERATORNAME_SPACE;
                        }

                        if j == count - 1
                            && i + count < tree.len()
                            && match &tree[i + count] {
                                ENode::DelimitedGroup { .. } => false,
                                ENode::SubSup { .. } => false,
                                ENode::Sqrt { .. } => true,
                                ENode::Frac { .. } => true,
                                ENode::SumProd { .. } => false,
                                ENode::Char(
                                    '.' | '+' | '-' | '*' | '=' | '<' | '>' | ',' | ':' | '×' | '÷'
                                    | '→' | '⋅',
                                ) => false,
                                ENode::Char(_) => true,
                            }
                        {
                            g.advance += OPERATORNAME_SPACE;
                        }

                        nodes.push(
                            Bounds {
                                width: g.advance,
                                height: CHAR_HEIGHT,
                                depth: CHAR_DEPTH,
                                ..Default::default()
                            },
                            Node::Char(g.clone()),
                        );
                    }

                    i += count;
                    continue 'outer;
                }
            }

            match &tree[i] {
                ENode::DelimitedGroup { .. } => todo!(),
                ENode::SubSup { .. } => todo!(),
                ENode::Sqrt { .. } => todo!(),
                ENode::Frac { .. } => todo!(),
                ENode::SumProd { .. } => todo!(),
                ENode::Char(c) => {
                    let (space_before, space_after) = match *c {
                        '+' | '-'
                            if i > 0
                                && match &tree[i - 1] {
                                    ENode::Char(
                                        '.' | '+' | '-' | '*' | '=' | '<' | '>' | ',' | ':' | '×'
                                        | '÷' | '→' | '⋅',
                                    ) => false,
                                    _ => true,
                                } =>
                        {
                            (BINOP_SPACE, BINOP_SPACE)
                        }
                        '*' | '=' | '<' | '>' | '×' | '÷' | '→' | '⋅' => {
                            (BINOP_SPACE, BINOP_SPACE)
                        }
                        ',' => (0.0, COMMA_SPACE),
                        ':' => (0.0, COLON_SPACE),
                        _ => (0.0, 0.0),
                    };

                    let c = match *c {
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
                    let mut g = get_glyph(font, c);
                    g.plane.top -= CHAR_CENTER;
                    g.plane.bottom -= CHAR_CENTER;
                    g.plane.left += space_before;
                    g.plane.right += space_before;
                    g.advance += space_before + space_after;

                    nodes.push(
                        Bounds {
                            width: g.advance,
                            height: CHAR_HEIGHT,
                            depth: CHAR_DEPTH,
                            ..Default::default()
                        },
                        Node::Char(g.clone()),
                    );
                }
            }

            i += 1;
        }
        nodes
    }

    fn make_absolute(nodes: &mut Nodes, position: DVec2, scale: f64) {
        nodes.bounds.transform(position, scale);
        for (bounds, node) in &mut nodes.nodes {
            bounds.transform(nodes.bounds.position, nodes.bounds.scale);
            match node {
                Node::DelimitedGroup { .. } => todo!(),
                Node::SubSup { .. } => todo!(),
                Node::Sqrt { .. } => todo!(),
                Node::Frac { .. } => todo!(),
                Node::SumProd { .. } => todo!(),
                Node::Char(glyph) => {
                    glyph.advance *= bounds.scale;
                    glyph.plane.left = bounds.position.x + scale * glyph.plane.left;
                    glyph.plane.top = bounds.position.y + scale * glyph.plane.top;
                    glyph.plane.right = bounds.position.x + scale * glyph.plane.right;
                    glyph.plane.bottom = bounds.position.y + scale * glyph.plane.bottom;
                }
            }
        }
    }

    pub fn layout(tree: &[ENode]) -> Nodes {
        let mut nodes = layout_relative(tree);
        make_absolute(&mut nodes, DVec2::ZERO, 1.0);
        nodes
    }
}
