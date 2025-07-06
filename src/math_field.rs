mod tree;

use std::{iter::zip, ops::Range};

use ambavia::{
    latex_parser::parse_latex,
    latex_tree::{self, ToString},
};
use glam::{DVec2, dvec2};
use winit::{
    event::{ElementState, KeyEvent, MouseButton},
    window::CursorIcon,
};

use crate::{
    katex_font::{Font, get_glyph},
    math_field::tree::{
        BigOp as BigOpE, Bracket as BracketE,
        Node::{self, *},
        TexturedQuad, Tree, ends_in_operatorname, new_big_op, new_bracket, new_char, new_frac,
        new_radical, new_script, new_script_lower, new_script_upper, new_sqrt, to_latex,
    },
    ui::{Bounds, Context, Event, QuadKind, Response},
    utility::{mix, snap},
};

/// Specifies which structural component of a node to navigate to. Used for
/// traversing into the different parts of an expression tree.
#[derive(Debug, Clone, PartialEq)]
enum NodeField {
    BracketInner,
    ScriptLower,
    ScriptUpper,
    RadicalRoot,
    RadicalArg,
    FracNum,
    FracDen,
    BigOpLower,
    BigOpUpper,
}

use NodeField::*;

/// Represents a path to a subtree. Each tuple contains the child index and
/// which node field to enter.
type Path = Vec<(usize, NodeField)>;
type PathSlice<'a> = &'a [(usize, NodeField)];

/// Represents a cursor position within an expression tree. The cursor sits
/// between nodes.
#[derive(Debug, Clone, PartialEq)]
struct Cursor {
    /// The path through the expression tree to reach the target node list.
    path: Path,
    /// The position within the target node list. The cursor
    /// sits to the left of the `index`th node.
    index: usize,
}

impl From<(Path, usize)> for Cursor {
    fn from((path, index): (Path, usize)) -> Self {
        Self { path, index }
    }
}

/// Represents the extent of a user selection within a single node list. Can
/// be either a cursor (zero-width) or a range of selected nodes.
#[derive(Debug, PartialEq)]
enum SelectionSpan {
    /// A cursor position between two nodes.
    Cursor(usize),
    /// A range of selected nodes, where `start < end`.
    Range(Range<usize>),
}

impl From<Range<usize>> for SelectionSpan {
    fn from(r: Range<usize>) -> Self {
        if r.is_empty() {
            Self::Cursor(r.start)
        } else {
            Self::Range(r)
        }
    }
}

impl SelectionSpan {
    fn as_range(&self) -> Range<usize> {
        match self {
            Self::Cursor(pos) => *pos..*pos,
            Self::Range(r) => r.clone(),
        }
    }
}

/// Represents the user's actual selection before normalization. The
/// `anchor` and `focus` may be at different tree depths and in any order.
#[derive(Debug)]
struct UserSelection {
    /// Where the selection started (e.g., mouse down position or
    /// `Shift`+`Arrow` start)
    anchor: Cursor,
    /// Where the selection currently ends (e.g., current mouse position or
    /// `Shift`+`Arrow` end)
    focus: Cursor,
}

impl From<Cursor> for UserSelection {
    fn from(c: Cursor) -> Self {
        Self {
            anchor: c.clone(),
            focus: c,
        }
    }
}

impl From<(Cursor, Cursor)> for UserSelection {
    fn from((a, f): (Cursor, Cursor)) -> Self {
        Self {
            anchor: a,
            focus: f,
        }
    }
}

/// Represents a normalized selection within the document tree. Unlike
/// `UserSelection`, this is guaranteed to be within a single node list.
struct Selection {
    /// The path to the node list containing the selection.
    path: Path,
    /// The selected range or cursor position within the node list.
    span: SelectionSpan,
}

impl UserSelection {
    fn normalize(&self) -> (Path, f64, f64) {
        let path: Path = zip(&self.anchor.path, &self.focus.path)
            .take_while(|(a, f)| a == f)
            .map(|(a, _)| a.clone())
            .collect();
        let anchor = self
            .anchor
            .path
            .get(path.len())
            .map_or(self.anchor.index as f64, |(i, _)| *i as f64 + 0.5);
        let focus = self
            .focus
            .path
            .get(path.len())
            .map_or(self.focus.index as f64, |(i, _)| *i as f64 + 0.5);
        (path, anchor, focus)
    }
}

impl From<&UserSelection> for Selection {
    fn from(s: &UserSelection) -> Self {
        let (path, anchor, focus) = s.normalize();
        let span = (anchor.min(focus).floor() as usize..anchor.max(focus).ceil() as usize).into();
        Self { path, span }
    }
}

fn bd(node: Node) -> (tree::Bounds, Node) {
    (Default::default(), node)
}

impl Tree {
    fn walk(&self, path: PathSlice) -> &Tree {
        let mut tree = self;
        for (index, field) in path {
            tree = match (&tree.nodes[*index].1, field) {
                (Bracket { inner, .. }, BracketInner) => inner,
                (Script { lower: Some(l), .. }, ScriptLower) => l,
                (Script { upper: Some(u), .. }, ScriptUpper) => u,
                (Radical { root: Some(r), .. }, RadicalRoot) => r,
                (Radical { arg, .. }, RadicalArg) => arg,
                (Frac { num, .. }, FracNum) => num,
                (Frac { den, .. }, FracDen) => den,
                (BigOp { lower, .. }, BigOpLower) => lower,
                (BigOp { upper, .. }, BigOpUpper) => upper,
                (node, field) => {
                    panic!("mismatched node/field:\n  node = {node:?}\n  field = {field:?}")
                }
            };
        }
        tree
    }

    fn walk_mut(&mut self, path: PathSlice) -> &mut Tree {
        let mut tree = self;
        for (index, field) in path {
            tree = match (&mut tree.nodes[*index].1, field) {
                (Bracket { inner, .. }, BracketInner) => inner,
                (Script { lower: Some(l), .. }, ScriptLower) => l,
                (Script { upper: Some(u), .. }, ScriptUpper) => u,
                (Radical { root: Some(r), .. }, RadicalRoot) => r,
                (Radical { arg, .. }, RadicalArg) => arg,
                (Frac { num, .. }, FracNum) => num,
                (Frac { den, .. }, FracDen) => den,
                (BigOp { lower, .. }, BigOpLower) => lower,
                (BigOp { upper, .. }, BigOpUpper) => upper,
                (node, field) => {
                    panic!("mismatched node/field:\n  node = {node:?}\n  field = {field:?}")
                }
            };
        }
        tree
    }

    fn get_hovered(&self, mut path: Path, position: DVec2) -> Cursor {
        let mut tree = self;
        let index = 'index: loop {
            for (i, (bounds, node)) in tree.nodes.iter().enumerate() {
                if position.x >= bounds.right() {
                    continue;
                }
                match node {
                    Bracket { inner, .. } => {
                        if position.x < (bounds.left() + inner.bounds.left()) / 2.0 {
                            break 'index i;
                        }
                        if position.x >= (inner.bounds.right() + bounds.right()) / 2.0 {
                            break 'index i + 1;
                        }
                        path.push((i, BracketInner));
                        tree = inner;
                        continue 'index;
                    }
                    Script { lower, upper } => {
                        if position.x < bounds.left() {
                            break 'index i;
                        }
                        tree = match (lower, upper) {
                            (None, None) => unreachable!(),
                            (None, Some(upper)) => {
                                path.push((i, ScriptUpper));
                                upper
                            }
                            (Some(lower), None) => {
                                path.push((i, ScriptLower));
                                lower
                            }
                            (Some(lower), Some(upper)) => {
                                if position.y < upper.bounds.bottom() {
                                    path.push((i, ScriptUpper));
                                    upper
                                } else {
                                    path.push((i, ScriptLower));
                                    lower
                                }
                            }
                        };
                        continue 'index;
                    }
                    Radical {
                        root: Some(root),
                        arg,
                        ..
                    } => {
                        if position.x < (bounds.left() + root.bounds.left()) / 2.0 {
                            break 'index i;
                        }
                        if position.x >= (arg.bounds.right() + bounds.right()) / 2.0 {
                            break 'index i + 1;
                        }
                        tree = if position.x < (root.bounds.right() + arg.bounds.left()) / 2.0 {
                            path.push((i, RadicalRoot));
                            root
                        } else {
                            path.push((i, RadicalArg));
                            arg
                        };
                        continue 'index;
                    }
                    Radical { arg, .. } => {
                        if position.x < (bounds.left() + arg.bounds.left()) / 2.0 {
                            break 'index i;
                        }
                        if position.x >= (arg.bounds.right() + bounds.right()) / 2.0 {
                            break 'index i + 1;
                        }
                        path.push((i, RadicalArg));
                        tree = arg;
                        continue 'index;
                    }
                    Frac { num, den, line } => {
                        if position.x < line.x_min {
                            break 'index i;
                        }
                        if position.x >= line.x_max {
                            break 'index i + 1;
                        }
                        tree = if position.y < line.y {
                            path.push((i, FracNum));
                            num
                        } else {
                            path.push((i, FracDen));
                            den
                        };
                        continue 'index;
                    }
                    BigOp { lower, upper, .. } => {
                        let middle = (lower.bounds.top() + upper.bounds.bottom()) / 2.0;
                        let (script, field) = if position.y < middle {
                            (upper, BigOpUpper)
                        } else {
                            (lower, BigOpLower)
                        };
                        if position.x < (bounds.left() + script.bounds.left()) / 2.0 {
                            break 'index i;
                        }
                        if position.x >= (script.bounds.right() + bounds.right()) / 2.0 {
                            break 'index i + 1;
                        }
                        path.push((i, field));
                        tree = script;
                        continue 'index;
                    }
                    Char { .. } => {
                        if position.x < bounds.left() + 0.5 * bounds.width {
                            break 'index i;
                        }
                    }
                }
            }

            break tree.nodes.len();
        };

        Cursor { path, index }
    }

    fn add_char(&mut self, mut path: Path, span: SelectionSpan, ch: char) -> Cursor {
        let i = match span {
            SelectionSpan::Cursor(c) => c,
            SelectionSpan::Range(r) => {
                self.drain(r.clone());
                r.start
            }
        };
        if ch.is_ascii_digit() {
            let is_letter = |n: &Node| {
                matches!(
                    n,
                    Char {
                        ch:
                            'A'..='Z'
                            | 'a'..='z'
                            | 'Γ'
                            | 'Δ'
                            | 'Θ'
                            | 'Λ'
                            | 'Ξ'
                            | 'Π'
                            | 'Σ'
                            | 'Υ'
                            | 'Φ'
                            | 'Ψ'
                            | 'Ω'
                            | 'α'..='ω'
                            | 'ϑ'
                            | 'ϕ'
                            | 'ϖ'
                            | 'ϱ'
                            | 'ϵ',
                        ..
                    }
                )
            };
            if i > 0 && is_letter(&self[i - 1].1) && !ends_in_operatorname(&self[..i])
                || i > 1
                    && is_letter(&self[i - 2].1)
                    && !ends_in_operatorname(&self[..i - 1])
                    && matches!(&self[i - 1].1, Script { upper: None, .. })
            {
                self.insert(i, new_script_lower(vec![bd(new_char(ch))]));
                return (path, i + 1).into();
            }
        }
        self.insert(i, new_char(ch));
        let replacements = [
            ("sqrt", ' '),
            ("cbrt", ' '),
            ("nthroot", ' '),
            ("sum", ' '),
            ("prod", ' '),
            ("int", ' '),
            ("cross", '×'),
            ("Gamma", 'Γ'),
            ("Delta", 'Δ'),
            ("Theta", 'Θ'),
            ("Lambda", 'Λ'),
            ("Xi", 'Ξ'),
            ("Pi", 'Π'),
            ("Sigma", 'Σ'),
            ("Upsilon", 'Υ'),
            ("Uψlon", 'Υ'),
            ("Phi", 'Φ'),
            ("Psi", 'Ψ'),
            ("Omega", 'Ω'),
            ("alpha", 'α'),
            ("beta", 'β'),
            ("gamma", 'γ'),
            ("delta", 'δ'),
            ("varepsilon", 'ε'),
            ("vareψlon", 'ε'),
            ("zeta", 'ζ'),
            ("vartheta", 'ϑ'),
            ("theta", 'θ'),
            ("eta", 'η'),
            ("iota", 'ι'),
            ("kappa", 'κ'),
            ("lambda", 'λ'),
            ("mu", 'μ'),
            ("nu", 'ν'),
            ("xi", 'ξ'),
            ("varpi", 'ϖ'),
            ("pi", 'π'),
            ("varrho", 'ϱ'),
            ("rho", 'ρ'),
            ("varsigma", 'ς'),
            ("sigma", 'σ'),
            ("tau", 'τ'),
            ("upsilon", 'υ'),
            ("uψlon", 'υ'),
            ("varphi", 'φ'),
            ("chi", 'χ'),
            ("psi", 'ψ'),
            ("omega", 'ω'),
            ("phi", 'ϕ'),
            ("epsilon", 'ϵ'),
            ("eψlon", 'ϵ'),
            ("->", '→'),
            ("infty", '∞'),
            ("infinity", '∞'),
            ("<=", '≤'),
            (">=", '≥'),
            ("*", '⋅'),
        ];
        for (find, replace) in replacements {
            let char_count = find.chars().count();
            if i + 1 >= char_count
                && find
                    .chars()
                    .rev()
                    .enumerate()
                    .all(|(j, c)| self[i - j].1.is_char(c))
            {
                let index = i + 1 - char_count;
                self.drain(index..i + 1);

                return match find {
                    "sqrt" => {
                        self.insert(index, new_sqrt(vec![]));
                        path.push((index, RadicalArg));
                        (path, 0)
                    }
                    "cbrt" => {
                        self.insert(index, new_radical(Some(vec![bd(new_char('3'))]), vec![]));
                        path.push((index, RadicalArg));
                        (path, 0)
                    }
                    "nthroot" => {
                        self.insert(index, new_radical(Some(vec![]), vec![]));
                        path.push((index, RadicalRoot));
                        (path, 0)
                    }
                    kind @ ("sum" | "prod" | "int") => {
                        let lower = if kind == "int" {
                            vec![]
                        } else {
                            vec![bd(new_char('n')), bd(new_char('='))]
                        };
                        let i = lower.len();
                        self.insert(index, new_big_op(kind, lower, vec![]));
                        path.push((index, BigOpLower));
                        (path, i)
                    }
                    _ => {
                        self.insert(index, new_char(replace));
                        (path, index + 1)
                    }
                }
                .into();
            }
        }
        (path, i + 1).into()
    }

    fn get(&self, index: usize) -> Option<&Node> {
        self.nodes.get(index).map(|(_, n)| n)
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut Node> {
        self.nodes.get_mut(index).map(|(_, n)| n)
    }

    fn get_bracket_mut(
        &mut self,
        index: usize,
    ) -> (&mut Option<BracketE>, &mut Option<BracketE>, &mut Tree) {
        match self.get_mut(index) {
            Some(Bracket {
                left, right, inner, ..
            }) => (left, right, inner),
            other => panic!("expected Bracket, found {other:?}"),
        }
    }

    fn get_script(&self, index: usize) -> (&Option<Tree>, &Option<Tree>) {
        match self.get(index) {
            Some(Script { lower, upper }) => (lower, upper),
            other => panic!("expected Script, found {other:?}"),
        }
    }

    fn get_radical(&self, index: usize) -> (&Option<Tree>, &Tree) {
        match self.get(index) {
            Some(Radical { root, arg, .. }) => (root, arg),
            other => panic!("expected Radical, found {other:?}"),
        }
    }

    fn remove_bracket(&mut self, index: usize) -> (Option<BracketE>, Option<BracketE>, Tree) {
        match self.remove(index).1 {
            Bracket {
                left, right, inner, ..
            } => (left, right, inner),
            other => panic!("expected Bracket, found {other:?}"),
        }
    }

    fn remove_script(&mut self, index: usize) -> (Option<Tree>, Option<Tree>) {
        match self.remove(index).1 {
            Script { lower, upper } => (lower, upper),
            other => panic!("expected Script, found {other:?}"),
        }
    }

    fn remove_radical(&mut self, index: usize) -> (Option<Tree>, Tree) {
        match self.remove(index).1 {
            Radical { root, arg, .. } => (root, arg),
            other => panic!("expected Radical, found {other:?}"),
        }
    }

    fn remove_frac(&mut self, index: usize) -> (Tree, Tree) {
        match self.remove(index).1 {
            Frac { num, den, .. } => (num, den),
            other => panic!("expected Frac, found {other:?}"),
        }
    }

    fn remove_big_op(&mut self, index: usize) -> (BigOpE, Tree, Tree) {
        match self.remove(index).1 {
            BigOp {
                op, lower, upper, ..
            } => (op, lower, upper),
            other => panic!("expected BigOp, found {other:?}"),
        }
    }

    fn join_consecutive_scripts(&mut self, mut cursor: Option<(&mut Cursor, usize)>) {
        let mut i = 0;
        while i < self.len() {
            if let Script { .. } = &self[i].1 {
                let (lower, upper) = self.remove_script(i);
                let mut lowers: Vec<Vec<_>> = vec![];
                let mut uppers: Vec<Vec<_>> = vec![];
                lowers.extend(lower.map(|x| x.nodes));
                uppers.extend(upper.map(|x| x.nodes));

                while let Some(Script { .. }) = self.get(i) {
                    let (lower, upper) = self.remove_script(i);
                    if let Some((cursor, offset)) = &mut cursor {
                        if let Some((index, field)) = cursor.path.get_mut(*offset) {
                            if *index > i {
                                *index -= 1;
                                if *index == i {
                                    let field = field.clone();
                                    let index = cursor
                                        .path
                                        .get_mut(*offset + 1)
                                        .map_or(&mut cursor.index, |(index, _)| index);
                                    *index += if field == ScriptLower {
                                        &lowers
                                    } else {
                                        assert_eq!(field, ScriptUpper);
                                        &uppers
                                    }
                                    .iter()
                                    .map(|s| s.len())
                                    .sum::<usize>();
                                }
                            }
                        } else if cursor.index > i {
                            cursor.index -= 1;
                            if cursor.index == i {
                                if !uppers.is_empty() {
                                    cursor.path.push((cursor.index, ScriptUpper));
                                    cursor.index = uppers.iter().map(|s| s.len()).sum();
                                } else {
                                    assert!(!lowers.is_empty());
                                    cursor.path.push((cursor.index, ScriptLower));
                                    cursor.index = lowers.iter().map(|s| s.len()).sum();
                                }
                            }
                        }
                    }
                    lowers.extend(lower.map(|x| x.nodes));
                    uppers.extend(upper.map(|x| x.nodes));
                }

                let lower = lowers.into_iter().reduce(|mut x, mut y| {
                    x.append(&mut y);
                    x
                });
                let upper = uppers.into_iter().reduce(|mut x, mut y| {
                    x.append(&mut y);
                    x
                });
                assert!(lower.is_some() || upper.is_some());
                self.insert(i, new_script(lower, upper));
            }

            i += 1;
        }

        for (i, (_, node)) in self.iter_mut().enumerate() {
            let field_cursor = if let Some((cursor, offset)) = &mut cursor
                && let Some((index, field)) = cursor.path.get(*offset)
                && *index == i
            {
                Some((field.clone(), (&mut **cursor, *offset + 1)))
            } else {
                None
            };

            match node {
                Bracket { inner, .. } => {
                    let cursor = match field_cursor {
                        Some((BracketInner, cursor)) => Some(cursor),
                        _ => None,
                    };
                    inner.join_consecutive_scripts(cursor);
                }
                Script { lower, upper } => {
                    let (lower_cursor, upper_cursor) = match field_cursor {
                        Some((ScriptLower, cursor)) => (Some(cursor), None),
                        Some((ScriptUpper, cursor)) => (None, Some(cursor)),
                        _ => (None, None),
                    };
                    if let Some(lower) = lower {
                        lower.join_consecutive_scripts(lower_cursor);
                    }
                    if let Some(upper) = upper {
                        upper.join_consecutive_scripts(upper_cursor);
                    }
                }
                Radical { root, arg, .. } => {
                    let (root_cursor, arg_cursor) = match field_cursor {
                        Some((RadicalRoot, cursor)) => (Some(cursor), None),
                        Some((RadicalArg, cursor)) => (None, Some(cursor)),
                        _ => (None, None),
                    };
                    if let Some(root) = root {
                        root.join_consecutive_scripts(root_cursor);
                    }
                    arg.join_consecutive_scripts(arg_cursor);
                }
                Frac { num, den, .. } => {
                    let (num_cursor, den_cursor) = match field_cursor {
                        Some((FracNum, cursor)) => (Some(cursor), None),
                        Some((FracDen, cursor)) => (None, Some(cursor)),
                        _ => (None, None),
                    };
                    num.join_consecutive_scripts(num_cursor);
                    den.join_consecutive_scripts(den_cursor);
                }
                BigOp { lower, upper, .. } => {
                    let (lower_cursor, upper_cursor) = match field_cursor {
                        Some((BigOpLower, cursor)) => (Some(cursor), None),
                        Some((BigOpUpper, cursor)) => (None, Some(cursor)),
                        _ => (None, None),
                    };
                    lower.join_consecutive_scripts(lower_cursor);
                    upper.join_consecutive_scripts(upper_cursor);
                }
                Char { .. } => {}
            }
        }
    }

    fn close_parentheses(&mut self, close_all: bool) {
        let n = self.len();
        for (i, (_, node)) in self.iter_mut().enumerate() {
            match node {
                Bracket {
                    left, right, inner, ..
                } => {
                    if left.is_none() && (i != 0 || close_all) {
                        *left = *right;
                    }
                    if right.is_none() && (i != n - 1 || close_all) {
                        *right = *left;
                    }
                    inner.close_parentheses(close_all);
                }
                Script { lower, upper } => {
                    if let Some(lower) = lower {
                        lower.close_parentheses(close_all);
                    }
                    if let Some(upper) = upper {
                        upper.close_parentheses(close_all);
                    }
                }
                Radical { root, arg, .. } => {
                    if let Some(root) = root {
                        root.close_parentheses(close_all);
                    }
                    arg.close_parentheses(close_all);
                }
                Frac { num, den, .. } => {
                    num.close_parentheses(close_all);
                    den.close_parentheses(close_all);
                }
                BigOp { lower, upper, .. } => {
                    lower.close_parentheses(close_all);
                    upper.close_parentheses(close_all);
                }
                Char { .. } => {}
            }
        }
    }

    fn render_selection(
        &self,
        ctx: &Context,
        selection: &Selection,
        transform: &impl Fn(DVec2) -> DVec2,
        draw_quad: &mut impl FnMut(DVec2, DVec2, QuadKind),
    ) {
        let tree = self.walk(&selection.path);
        match &selection.span {
            SelectionSpan::Cursor(index) => {
                let position = if tree.is_empty() {
                    tree.bounds.position
                } else {
                    tree.nodes.get(*index).map_or(
                        tree.bounds.position + dvec2(tree.bounds.width, 0.0),
                        |(b, _)| b.position,
                    )
                };
                let b = tree::Bounds::default();
                let p0 = transform(position - dvec2(0.0, tree.bounds.scale * b.height));
                let p1 = transform(position + dvec2(0.0, tree.bounds.scale * b.depth));
                let w = ctx.round_nonzero_as_physical(1.0);
                let x = snap(p0.x, w);
                let p0 = dvec2(x - w as f64 / 2.0, p0.y.floor());
                let p1 = dvec2(x + w as f64 / 2.0, p1.y.ceil());
                draw_quad(p0, p1, QuadKind::BlackBox);
            }
            SelectionSpan::Range(r) => {
                for (b, _) in &tree.nodes[r.clone()] {
                    let p0 = transform(b.top_left());
                    let p1 = transform(b.bottom_right());
                    draw_quad(p0, p1, QuadKind::HighlightBox);
                }
            }
        }
    }

    fn render(
        &self,
        ctx: &Context,
        transform: &impl Fn(DVec2) -> DVec2,
        draw_quad: &mut impl FnMut(DVec2, DVec2, QuadKind),
    ) {
        if self.has_gray_background {
            draw_quad(
                transform(self.bounds.top_left()),
                transform(self.bounds.bottom_right()),
                QuadKind::TranslucentBlackBox,
            );
        }
        for (_, node) in &self.nodes {
            match node {
                Bracket {
                    left_quad,
                    right_quad,
                    inner,
                    ..
                } => {
                    left_quad.render(transform, draw_quad);
                    right_quad.render(transform, draw_quad);
                    inner.render(ctx, transform, draw_quad);
                }
                Script { lower, upper } => {
                    if let Some(lower) = lower {
                        lower.render(ctx, transform, draw_quad);
                    }
                    if let Some(upper) = upper {
                        upper.render(ctx, transform, draw_quad);
                    }
                }
                Radical {
                    root,
                    arg,
                    radical,
                    line,
                } => {
                    if let Some(root) = root {
                        root.render(ctx, transform, draw_quad);
                    }
                    arg.render(ctx, transform, draw_quad);
                    radical.render(transform, draw_quad);
                    let mut l0 = transform(dvec2(line.x_min, line.y));
                    let mut l1 = transform(dvec2(line.x_max, line.y));
                    l0.y = l0.y.floor();
                    l1.y = l0.y + ctx.round_nonzero_as_physical(1.0) as f64;
                    draw_quad(l0, l1, QuadKind::BlackBox);
                }
                Frac { num, den, line } => {
                    num.render(ctx, transform, draw_quad);
                    den.render(ctx, transform, draw_quad);
                    let l0 = transform(dvec2(line.x_min, line.y));
                    let l1 = transform(dvec2(line.x_max, line.y));
                    let w = ctx.round_nonzero_as_physical(1.0);
                    let y = snap(l0.y, w);
                    draw_quad(
                        dvec2(l0.x.floor(), y - w as f64 / 2.0),
                        dvec2(l1.x.ceil(), y + w as f64 / 2.0),
                        QuadKind::BlackBox,
                    );
                }
                BigOp {
                    lower,
                    upper,
                    op_quad,
                    ..
                } => {
                    lower.render(ctx, transform, draw_quad);
                    upper.render(ctx, transform, draw_quad);
                    op_quad.render(transform, draw_quad);
                }
                Char { quad, .. } => quad.render(transform, draw_quad),
            }
        }
    }
}

impl TexturedQuad {
    fn render(
        &self,
        transform: &impl Fn(DVec2) -> DVec2,
        draw_quad: &mut impl FnMut(DVec2, DVec2, QuadKind),
    ) {
        draw_quad(
            transform(self.position),
            transform(self.position + self.size),
            if self.gray {
                QuadKind::TranslucentMsdfGlyph
            } else {
                QuadKind::MsdfGlyph
            }(self.uv0, self.uv1),
        );
    }
}

#[derive(Debug)]
pub struct MathField {
    tree: Tree,
    /// Applied before scaling
    left_padding: f64,
    /// Applied before scaling
    right_padding: f64,
    /// Applied before scaling
    overflow_gradient_width: f64,
    scale: f64,
    /// Logical
    scroll: f64,
    dragging: bool,
    selection: Option<UserSelection>,
    selection_was_set: bool,
}

impl Default for MathField {
    fn default() -> Self {
        // These characters escape their bounds so draw extra by that much to
        // avoid them getting clipped off
        let j_glyph = get_glyph(Font::MainRegular, 'j');
        let v_glyph = get_glyph(Font::MathItalic, 'V');
        Self {
            tree: Default::default(),
            left_padding: -j_glyph.plane.left,
            right_padding: v_glyph.plane.right - v_glyph.advance,
            overflow_gradient_width: 0.5,
            scale: 20.0,
            scroll: 0.0,
            dragging: false,
            selection: None,

            selection_was_set: false,
        }
    }
}

impl From<&[latex_tree::Node<'_>]> for MathField {
    fn from(value: &[latex_tree::Node<'_>]) -> Self {
        let mut tree = Tree::from(value);
        tree.layout();
        Self {
            tree,
            ..Default::default()
        }
    }
}

impl From<&latex_tree::Nodes<'_>> for MathField {
    fn from(value: &latex_tree::Nodes<'_>) -> Self {
        value.as_slice().into()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Message {
    Up,
    Down,
    Add,
    Remove,
}

impl MathField {
    /// Logical
    pub fn expression_size(&self) -> DVec2 {
        let b = &self.tree.bounds;
        self.scale
            * dvec2(
                self.left_padding + b.width + self.right_padding,
                b.height + b.depth,
            )
    }

    fn set_selection(&mut self, selection: impl Into<UserSelection>) {
        self.selection = Some(selection.into());
        self.selection_was_set = true;
    }

    fn set_cursor(&mut self, cursor: impl Into<Cursor>) {
        self.set_selection(cursor.into());
    }

    fn tree_updated(&mut self, cursor: impl Into<Cursor>) {
        let mut cursor = cursor.into();
        self.tree.join_consecutive_scripts(Some((&mut cursor, 0)));
        self.tree.close_parentheses(false);
        self.tree.layout();
        self.set_cursor(cursor);
    }

    pub fn unfocus(&mut self) -> Response {
        let mut response = Response::default();
        if self.selection.is_some() {
            self.selection = None;
            self.tree.close_parentheses(true);
            self.tree.layout();
            self.scroll = 0.0;
            response.request_redraw();
        }
        response
    }

    pub fn focus(&mut self) -> Response {
        let mut response = Response::default();
        if !self.has_focus() {
            self.set_cursor((vec![], self.tree.len()));
            response.request_redraw();
        }
        response
    }

    pub fn has_focus(&self) -> bool {
        self.selection.is_some()
    }

    pub fn update(
        &mut self,
        ctx: &Context,
        event: &Event,
        bounds: Bounds,
    ) -> (Response, Option<Message>) {
        let mut response = Response::default();
        let mut message = None;

        let hovered = (bounds.contains(ctx.cursor) || self.dragging).then(|| {
            let position = (ctx.cursor - bounds.pos + dvec2(self.scroll, 0.0)) / self.scale
                - dvec2(self.left_padding, self.tree.bounds.height);
            self.tree.get_hovered(vec![], position)
        });

        if hovered.is_some() {
            response.cursor_icon = CursorIcon::Text;
        }

        match event {
            Event::KeyboardInput(KeyEvent {
                logical_key,
                state: ElementState::Pressed,
                ..
            }) if self.has_focus() => {
                use winit::keyboard::{Key, NamedKey};
                let Selection { mut path, span } = self.selection.as_ref().unwrap().into();

                match &logical_key {
                    Key::Named(NamedKey::Enter) => {
                        message = Some(Message::Add);
                        response.consume_event();
                    }
                    Key::Named(NamedKey::Space) => {
                        let cursor = self.tree.walk_mut(&path).add_char(path, span, ' ');
                        self.tree_updated(cursor);
                        response.request_redraw();
                        response.consume_event();
                    }
                    Key::Named(NamedKey::ArrowLeft) => {
                        if ctx.modifiers.shift_key() {
                            let s = self.selection.take().unwrap();
                            let (mut path, anchor, focus) = s.normalize();
                            let index = if focus % 1.0 == 0.5 {
                                let mut index = focus.floor() as usize;
                                if focus <= anchor && index > 0 {
                                    index -= 1
                                }
                                index
                            } else if anchor + 0.5 == focus {
                                path.push(s.anchor.path[path.len()].clone());
                                self.tree.walk(&path).len()
                            } else if focus > 0.0 {
                                focus as usize - 1
                            } else if let Some((index, _)) = path.pop() {
                                index
                            } else {
                                0
                            };
                            self.set_selection((s.anchor, Cursor { path, index }));
                        } else {
                            match span {
                                SelectionSpan::Cursor(mut i) => {
                                    if i > 0 {
                                        i -= 1;
                                        match &self.tree.walk(&path)[i].1 {
                                            Bracket { inner, .. } => {
                                                path.push((i, BracketInner));
                                                let index = inner.len();
                                                self.set_cursor((path, index));
                                            }
                                            Script {
                                                upper: Some(upper), ..
                                            } => {
                                                path.push((i, ScriptUpper));
                                                let index = upper.len();
                                                self.set_cursor((path, index));
                                            }
                                            Radical { arg, .. } => {
                                                path.push((i, RadicalArg));
                                                let index = arg.len();
                                                self.set_cursor((path, index));
                                            }
                                            Frac { num, .. } => {
                                                path.push((i, FracNum));
                                                let index = num.len();
                                                self.set_cursor((path, index));
                                            }
                                            BigOp { upper, .. } => {
                                                path.push((i, BigOpUpper));
                                                let index = upper.len();
                                                self.set_cursor((path, index));
                                            }
                                            Script { .. } | Char { .. } => {
                                                self.set_cursor((path, i))
                                            }
                                        }
                                    } else if let Some((index, field)) = path.pop() {
                                        match field {
                                            RadicalArg => {
                                                let (root, _) =
                                                    self.tree.walk(&path).get_radical(index);
                                                if let Some(root) = root {
                                                    path.push((index, RadicalRoot));
                                                    let index = root.len();
                                                    self.set_cursor((path, index));
                                                } else {
                                                    self.set_cursor((path, index));
                                                }
                                            }
                                            BracketInner | ScriptLower | ScriptUpper
                                            | RadicalRoot | FracNum | FracDen | BigOpLower
                                            | BigOpUpper => {
                                                self.set_cursor((path, index));
                                            }
                                        }
                                    }
                                }
                                SelectionSpan::Range(r) => self.set_cursor((path, r.start)),
                            }
                        }
                        response.request_redraw();
                        response.consume_event();
                    }
                    Key::Named(NamedKey::ArrowRight) => {
                        if ctx.modifiers.shift_key() {
                            let s = self.selection.take().unwrap();
                            let (mut path, anchor, focus) = s.normalize();
                            let index = if focus % 1.0 == 0.5 {
                                let mut index = focus.ceil() as usize;
                                if focus >= anchor && index < self.tree.walk(&path).len() {
                                    index += 1
                                }
                                index
                            } else if anchor - 0.5 == focus {
                                path.push(s.anchor.path[path.len()].clone());
                                0
                            } else {
                                let index = focus as usize;
                                if index < self.tree.walk(&path).len() {
                                    index + 1
                                } else if let Some((index, _)) = path.pop() {
                                    index + 1
                                } else {
                                    self.tree.len()
                                }
                            };
                            self.set_selection((s.anchor, Cursor { path, index }));
                        } else {
                            match span {
                                SelectionSpan::Cursor(i) => {
                                    let nodes = self.tree.walk(&path);
                                    if i < nodes.len() {
                                        match &nodes[i].1 {
                                            Bracket { .. } => {
                                                path.push((i, BracketInner));
                                                self.set_cursor((path, 0));
                                            }
                                            Script { upper: Some(_), .. } => {
                                                path.push((i, ScriptUpper));
                                                self.set_cursor((path, 0));
                                            }
                                            Radical { root: Some(_), .. } => {
                                                path.push((i, RadicalRoot));
                                                self.set_cursor((path, 0));
                                            }
                                            Radical { .. } => {
                                                path.push((i, RadicalArg));
                                                self.set_cursor((path, 0));
                                            }
                                            Frac { .. } => {
                                                path.push((i, FracNum));
                                                self.set_cursor((path, 0));
                                            }
                                            BigOp { .. } => {
                                                path.push((i, BigOpUpper));
                                                self.set_cursor((path, 0));
                                            }
                                            Script { .. } | Char { .. } => {
                                                self.set_cursor((path, i + 1))
                                            }
                                        }
                                    } else if let Some((index, field)) = path.pop() {
                                        match field {
                                            RadicalRoot => {
                                                path.push((index, RadicalArg));
                                                self.set_cursor((path, 0));
                                            }
                                            BracketInner | ScriptLower | ScriptUpper
                                            | RadicalArg | FracNum | FracDen | BigOpLower
                                            | BigOpUpper => {
                                                self.set_cursor((path, index + 1));
                                            }
                                        }
                                    }
                                }
                                SelectionSpan::Range(r) => self.set_cursor((path, r.end)),
                            }
                        }
                        response.consume_event();
                        response.request_redraw();
                    }
                    Key::Named(NamedKey::ArrowDown) => {
                        if ctx.modifiers.shift_key() {
                            let s = self.selection.take().unwrap();
                            let (mut path, anchor, focus) = s.normalize();
                            let index = if focus > anchor
                                || anchor % 1.0 == 0.0
                                || focus == anchor && focus % 1.0 == 0.0
                            {
                                let index = self.tree.walk(&path).len();
                                if focus.ceil() < index as f64 || path.is_empty() {
                                    index
                                } else {
                                    path.pop().unwrap().0 + 1
                                }
                            } else if focus + 0.5 == anchor {
                                path.push(s.anchor.path[path.len()].clone());
                                0
                            } else {
                                anchor.floor() as usize
                            };
                            self.set_selection((s.anchor, Cursor { path, index }));
                        } else {
                            let i = span.as_range().end;

                            'stuff: {
                                let nodes = self.tree.walk(&path);
                                if i < nodes.len() {
                                    match nodes[i].1 {
                                        Bracket { .. } => {}
                                        Script { lower: Some(_), .. } => {
                                            path.push((i, ScriptLower));
                                            self.set_cursor((path, 0));
                                            break 'stuff;
                                        }
                                        Script { .. } => {}
                                        Radical { .. } => {}
                                        Frac { .. } => {
                                            path.push((i, FracDen));
                                            self.set_cursor((path, 0));
                                            break 'stuff;
                                        }
                                        BigOp { .. } => {
                                            path.push((i, BigOpLower));
                                            self.set_cursor((path, 0));
                                            break 'stuff;
                                        }
                                        Char { .. } => {}
                                    }
                                }

                                if i > 0 {
                                    match &nodes[i - 1].1 {
                                        Bracket { .. } => {}
                                        Script {
                                            lower: Some(lower), ..
                                        } => {
                                            path.push((i - 1, ScriptLower));
                                            let index = lower.len();
                                            self.set_cursor((path, index));
                                            break 'stuff;
                                        }
                                        Script { .. } => {}
                                        Radical { .. } => {}
                                        Frac { den, .. } => {
                                            path.push((i - 1, FracDen));
                                            let index = den.len();
                                            self.set_cursor((path, index));
                                            break 'stuff;
                                        }
                                        BigOp { lower, .. } => {
                                            path.push((i - 1, BigOpLower));
                                            let index = lower.len();
                                            self.set_cursor((path, index));
                                            break 'stuff;
                                        }
                                        Char { .. } => {}
                                    }
                                }

                                let nodes = self.tree.walk(&path);
                                let x = nodes
                                    .nodes
                                    .get(i)
                                    .map_or(nodes.bounds.right(), |(b, _)| b.left());

                                loop {
                                    if let Some((index, field)) = path.pop() {
                                        match field {
                                            BracketInner => {}
                                            ScriptLower => {}
                                            ScriptUpper => {
                                                let (lower, _) =
                                                    self.tree.walk(&path).get_script(index);
                                                if lower.is_some() {
                                                    path.push((index, ScriptLower));
                                                    break;
                                                } else {
                                                    self.set_cursor((path, index));
                                                    break 'stuff;
                                                }
                                            }
                                            RadicalRoot => {}
                                            RadicalArg => {}
                                            FracNum => {
                                                path.push((index, FracDen));
                                                break;
                                            }
                                            FracDen => {}
                                            BigOpLower => {}
                                            BigOpUpper => {
                                                path.push((index, BigOpLower));
                                                break;
                                            }
                                        }
                                    } else {
                                        message = Some(Message::Down);
                                        break 'stuff;
                                    }
                                }

                                self.set_cursor(
                                    self.tree
                                        .walk(&path)
                                        .get_hovered(path, dvec2(x, -f64::INFINITY)),
                                );
                            }
                        }
                        response.request_redraw();
                        response.consume_event();
                    }
                    Key::Named(NamedKey::ArrowUp) => {
                        if ctx.modifiers.shift_key() {
                            let s = self.selection.take().unwrap();
                            let (mut path, anchor, focus) = s.normalize();
                            let index = if focus < anchor
                                || anchor % 1.0 == 0.0
                                || focus == anchor && focus % 1.0 == 0.0
                            {
                                let index = focus.floor() as usize;
                                if index > 0 || path.is_empty() {
                                    0
                                } else {
                                    path.pop().unwrap().0
                                }
                            } else if focus - 0.5 == anchor {
                                path.push(s.anchor.path[path.len()].clone());
                                self.tree.walk(&path).len()
                            } else {
                                anchor.ceil() as usize
                            };
                            self.set_selection((s.anchor, Cursor { path, index }));
                        } else {
                            let i = span.as_range().end;

                            'stuff: {
                                let nodes = self.tree.walk(&path);
                                if i < nodes.len() {
                                    match nodes[i].1 {
                                        Bracket { .. } => {}
                                        Script { upper: Some(_), .. } => {
                                            path.push((i, ScriptUpper));
                                            self.set_cursor((path, 0));
                                            break 'stuff;
                                        }
                                        Script { .. } => {}
                                        Radical { root: Some(_), .. } => {
                                            path.push((i, RadicalRoot));
                                            self.set_cursor((path, 0));
                                            break 'stuff;
                                        }
                                        Radical { .. } => {}
                                        Frac { .. } => {
                                            path.push((i, FracNum));
                                            self.set_cursor((path, 0));
                                            break 'stuff;
                                        }
                                        BigOp { .. } => {
                                            path.push((i, BigOpUpper));
                                            self.set_cursor((path, 0));
                                            break 'stuff;
                                        }
                                        Char { .. } => {}
                                    }
                                }

                                if i > 0 {
                                    match &nodes[i - 1].1 {
                                        Bracket { .. } => {}
                                        Script {
                                            upper: Some(upper), ..
                                        } => {
                                            path.push((i - 1, ScriptUpper));
                                            let index = upper.len();
                                            self.set_cursor((path, index));
                                            break 'stuff;
                                        }
                                        Script { .. } => {}
                                        Radical { .. } => {}
                                        Frac { num, .. } => {
                                            path.push((i - 1, FracNum));
                                            let index = num.len();
                                            self.set_cursor((path, index));
                                            break 'stuff;
                                        }
                                        BigOp { upper, .. } => {
                                            path.push((i - 1, BigOpUpper));
                                            let index = upper.len();
                                            self.set_cursor((path, index));
                                            break 'stuff;
                                        }
                                        Char { .. } => {}
                                    }
                                }

                                let nodes = self.tree.walk(&path);
                                let x = nodes
                                    .nodes
                                    .get(i)
                                    .map_or(nodes.bounds.right(), |(b, _)| b.left());

                                loop {
                                    if let Some((index, field)) = path.pop() {
                                        match field {
                                            BracketInner => {}
                                            ScriptLower => {
                                                let (_, upper) =
                                                    self.tree.walk(&path).get_script(index);
                                                if upper.is_some() {
                                                    path.push((index, ScriptUpper));
                                                    break;
                                                } else {
                                                    self.set_cursor((path, index));
                                                    break 'stuff;
                                                }
                                            }
                                            ScriptUpper => {}
                                            RadicalRoot => {}
                                            RadicalArg => {}
                                            FracNum => {}
                                            FracDen => {
                                                path.push((index, FracNum));
                                                break;
                                            }
                                            BigOpLower => {
                                                path.push((index, BigOpUpper));
                                                break;
                                            }
                                            BigOpUpper => {}
                                        }
                                    } else {
                                        message = Some(Message::Up);
                                        break 'stuff;
                                    }
                                }

                                self.set_cursor(
                                    self.tree
                                        .walk(&path)
                                        .get_hovered(path, dvec2(x, f64::INFINITY)),
                                );
                            }
                        }
                        response.request_redraw();
                        response.consume_event();
                    }
                    Key::Named(NamedKey::Backspace) => {
                        match span {
                            SelectionSpan::Cursor(mut i) => {
                                let nodes = self.tree.walk_mut(&path);
                                if i > 0 {
                                    i -= 1;
                                    match &mut nodes[i].1 {
                                        Bracket { left, .. } => {
                                            if left.is_none() {
                                                let (_, _, inner) = nodes.remove_bracket(i);
                                                let index = i + inner.len();
                                                nodes.splice(i..i, inner.nodes);
                                                self.tree_updated((path, index));
                                            } else {
                                                let rest = nodes.drain(i + 1..).collect::<Vec<_>>();
                                                let (_, right, inner) = nodes.get_bracket_mut(i);
                                                let index = inner.len();
                                                inner.extend(rest);
                                                *right = None;
                                                path.push((i, BracketInner));
                                                self.tree_updated((path, index));
                                            }
                                        }
                                        Script {
                                            upper: Some(upper), ..
                                        } => {
                                            path.push((i, ScriptUpper));
                                            let index = upper.len();
                                            self.set_cursor((path, index));
                                        }
                                        Script {
                                            lower: Some(lower), ..
                                        } => {
                                            path.push((i, ScriptLower));
                                            let index = lower.len();
                                            self.set_cursor((path, index));
                                        }
                                        Script { .. } => unreachable!(),
                                        Radical { arg, .. } => {
                                            path.push((i, RadicalArg));
                                            let index = arg.len();
                                            self.set_cursor((path, index));
                                        }
                                        Frac { den, .. } => {
                                            path.push((i, FracDen));
                                            let index = den.len();
                                            self.set_cursor((path, index));
                                        }
                                        BigOp { upper, .. } => {
                                            path.push((i, BigOpUpper));
                                            let index = upper.len();
                                            self.set_cursor((path, index));
                                        }
                                        Char { .. } => {
                                            nodes.remove(i);
                                            self.tree_updated((path, i));
                                        }
                                    }
                                } else if let Some((index, field)) = path.pop() {
                                    let nodes = self.tree.walk_mut(&path);
                                    match field {
                                        BracketInner => {
                                            if let Bracket { left: None, .. } = &nodes[index].1 {
                                                self.set_cursor((path, index));
                                            } else {
                                                let (_, _, inner) = nodes.remove_bracket(index);
                                                nodes.splice(index..index, inner.nodes);
                                                self.tree_updated((path, index));
                                            }
                                        }
                                        ScriptLower => {
                                            let (Some(lower), upper) = nodes.remove_script(index)
                                            else {
                                                unreachable!()
                                            };
                                            nodes.splice(
                                                index..index,
                                                lower.nodes.into_iter().chain(upper.map(|upper| {
                                                    bd({
                                                        Script {
                                                            lower: None,
                                                            upper: Some(upper),
                                                        }
                                                    })
                                                })),
                                            );
                                            self.tree_updated((path, index));
                                        }
                                        ScriptUpper => {
                                            let (lower, Some(upper)) = nodes.remove_script(index)
                                            else {
                                                unreachable!()
                                            };
                                            let i = if lower.is_some() { index + 1 } else { index };
                                            nodes.splice(
                                                index..index,
                                                lower
                                                    .map(|lower| {
                                                        bd(Script {
                                                            lower: Some(lower),
                                                            upper: None,
                                                        })
                                                    })
                                                    .into_iter()
                                                    .chain(upper.nodes),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                        RadicalRoot | RadicalArg => {
                                            let (root, arg) = nodes.remove_radical(index);
                                            let i = if field == RadicalRoot {
                                                index
                                            } else {
                                                index + root.as_ref().map_or(0, |root| root.len())
                                            };
                                            nodes.splice(
                                                index..index,
                                                root.map(|root| root.nodes)
                                                    .into_iter()
                                                    .flatten()
                                                    .chain(arg.nodes),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                        FracNum | FracDen => {
                                            let (num, den) = nodes.remove_frac(index);
                                            let i = if field == FracNum {
                                                index
                                            } else {
                                                index + num.len()
                                            };
                                            nodes.splice(
                                                index..index,
                                                num.nodes.into_iter().chain(den.nodes),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                        BigOpLower | BigOpUpper => {
                                            let (_, lower, upper) = nodes.remove_big_op(index);
                                            let i = if field == BigOpLower {
                                                index
                                            } else {
                                                index + lower.len()
                                            };
                                            nodes.splice(
                                                index..index,
                                                lower.nodes.into_iter().chain(upper.nodes),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                    }
                                } else if self.tree.is_empty() {
                                    message = Some(Message::Remove);
                                }
                            }
                            SelectionSpan::Range(r) => {
                                self.tree.walk_mut(&path).drain(r.clone());
                                self.tree_updated((path, r.start));
                            }
                        }

                        response.consume_event();
                        response.request_redraw();
                    }
                    Key::Named(NamedKey::Delete) => {
                        match span {
                            SelectionSpan::Cursor(i) => {
                                let nodes = self.tree.walk_mut(&path);
                                if i < nodes.len() {
                                    match &nodes[i].1 {
                                        Bracket { right, .. } => {
                                            if right.is_none() {
                                                let (_, _, inner) = nodes.remove_bracket(i);
                                                nodes.splice(i..i, inner.nodes);
                                                self.tree_updated((path, i));
                                            } else {
                                                let mut rest =
                                                    nodes.drain(0..i).collect::<Vec<_>>();
                                                let (left, _, inner) = nodes.get_bracket_mut(0);
                                                std::mem::swap(&mut rest, inner);
                                                inner.extend(rest);
                                                *left = None;
                                                path.push((0, BracketInner));
                                                self.tree_updated((path, i));
                                            }
                                        }
                                        Script { lower: Some(_), .. } => {
                                            path.push((i, ScriptLower));
                                            self.set_cursor((path, 0));
                                        }
                                        Script { upper: Some(_), .. } => {
                                            path.push((i, ScriptUpper));
                                            self.set_cursor((path, 0));
                                        }
                                        Script { .. } => unreachable!(),
                                        Radical { root: Some(_), .. } => {
                                            path.push((i, RadicalRoot));
                                            self.set_cursor((path, 0));
                                        }
                                        Radical { .. } => {
                                            let (_, arg) = nodes.remove_radical(i);
                                            nodes.splice(i..i, arg.nodes);
                                            self.tree_updated((path, i));
                                        }
                                        Frac { .. } => {
                                            path.push((i, FracNum));
                                            self.set_cursor((path, 0));
                                        }
                                        BigOp { .. } => {
                                            path.push((i, BigOpLower));
                                            self.set_cursor((path, 0));
                                        }
                                        Char { .. } => {
                                            nodes.remove(i);
                                            self.tree_updated((path, i));
                                        }
                                    }
                                } else if let Some((index, field)) = path.pop() {
                                    let nodes = self.tree.walk_mut(&path);
                                    match field {
                                        BracketInner => {
                                            if let Bracket { right: None, .. } = &nodes[index].1 {
                                                self.set_cursor((path, index + 1));
                                            } else {
                                                let (_, _, inner) = nodes.remove_bracket(index);
                                                let i = index + inner.len();
                                                nodes.splice(index..index, inner.nodes);
                                                self.tree_updated((path, i));
                                            }
                                        }
                                        ScriptLower => {
                                            let (Some(lower), upper) = nodes.remove_script(index)
                                            else {
                                                unreachable!()
                                            };
                                            let i = index + lower.len();
                                            nodes.splice(
                                                index..index,
                                                lower.nodes.into_iter().chain(upper.map(|upper| {
                                                    bd(Script {
                                                        lower: None,
                                                        upper: Some(upper),
                                                    })
                                                })),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                        ScriptUpper => {
                                            let (lower, Some(upper)) = nodes.remove_script(index)
                                            else {
                                                unreachable!()
                                            };
                                            let i = if lower.is_some() {
                                                index + 1 + upper.len()
                                            } else {
                                                index + upper.len()
                                            };
                                            nodes.splice(
                                                index..index,
                                                lower
                                                    .map(|lower| {
                                                        bd(Script {
                                                            lower: Some(lower),
                                                            upper: None,
                                                        })
                                                    })
                                                    .into_iter()
                                                    .chain(upper.nodes),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                        RadicalRoot | RadicalArg => {
                                            let (root, arg) = nodes.remove_radical(index);
                                            let i = index
                                                + root.as_ref().map_or(0, |root| root.len())
                                                + if field == RadicalArg { arg.len() } else { 0 };
                                            nodes.splice(
                                                index..index,
                                                root.map(|root| root.nodes)
                                                    .into_iter()
                                                    .flatten()
                                                    .chain(arg.nodes),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                        FracNum | FracDen => {
                                            let (num, den) = nodes.remove_frac(index);
                                            let i = if field == FracNum {
                                                index + num.len()
                                            } else {
                                                index + num.len() + den.len()
                                            };
                                            nodes.splice(
                                                index..index,
                                                num.nodes.into_iter().chain(den.nodes),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                        BigOpLower | BigOpUpper => {
                                            let (_, lower, upper) = nodes.remove_big_op(index);
                                            let i = if field == BigOpLower {
                                                index + lower.len()
                                            } else {
                                                index + lower.len() + upper.len()
                                            };
                                            nodes.splice(
                                                index..index,
                                                lower.nodes.into_iter().chain(upper.nodes),
                                            );
                                            self.tree_updated((path, i));
                                        }
                                    }
                                }
                            }
                            SelectionSpan::Range(r) => {
                                self.tree.walk_mut(&path).drain(r.clone());
                                self.tree_updated((path, r.start));
                            }
                        }
                        response.consume_event();
                        response.request_redraw();
                    }
                    Key::Character(c) => match c.as_str().chars().next() {
                        Some('a') if ctx.modifiers.control_key() || ctx.modifiers.super_key() => {
                            self.set_selection((
                                Cursor {
                                    path: vec![],
                                    index: 0,
                                },
                                Cursor {
                                    path: vec![],
                                    index: self.tree.len(),
                                },
                            ));
                            response.consume_event();
                            response.request_redraw();
                        }
                        Some('c') if ctx.modifiers.control_key() || ctx.modifiers.super_key() => {
                            if let SelectionSpan::Range(r) = span {
                                let latex = to_latex(&self.tree.walk(&path)[r]).to_string();
                                if let Err(e) = ctx.set_clipboard_text(latex) {
                                    eprintln!("failed to set clipboard contents: {e}");
                                }
                            }
                            response.consume_event();
                        }
                        Some('x') if ctx.modifiers.control_key() || ctx.modifiers.super_key() => {
                            if let SelectionSpan::Range(r) = span {
                                let nodes = self.tree.walk_mut(&path);
                                let latex = to_latex(&nodes[r.clone()]).to_string();
                                if let Err(e) = ctx.set_clipboard_text(latex) {
                                    eprintln!("failed to set clipboard contents: {e}");
                                } else {
                                    nodes.drain(r.clone());
                                    self.tree_updated((path, r.start));
                                    response.request_redraw();
                                }
                            }
                            response.consume_event();
                        }
                        Some('v') if ctx.modifiers.control_key() || ctx.modifiers.super_key() => {
                            match ctx.get_clipboard_text().as_ref().map(|s| parse_latex(s)) {
                                Ok(Ok(latex)) => {
                                    let nodes = Tree::from(&latex);
                                    let r = span.as_range();
                                    let pasted_len = nodes.len();
                                    self.tree.walk_mut(&path).splice(r.clone(), nodes.nodes);
                                    self.tree_updated((path, r.start + pasted_len));
                                    response.request_redraw();
                                }
                                Ok(Err(e)) => eprintln!("parse_latex error: {e:?}"),
                                Err(e) => eprintln!("failed to get clipboard contents: {e}"),
                            }
                            response.consume_event();
                        }
                        Some(b @ ('(' | '[' | '{')) => {
                            let b = Some(BracketE::from(b));
                            match span {
                                SelectionSpan::Cursor(i) => {
                                    let nodes = self.tree.walk_mut(&path);
                                    if let Some(Bracket {
                                        left: left @ None,
                                        right,
                                        ..
                                    }) = nodes.get_mut(i)
                                        && *right == b
                                    {
                                        *left = b;
                                        path.push((i, BracketInner));
                                    } else if let Some((index, BracketInner)) = path.last().cloned()
                                        && let nodes = self.tree.walk_mut(&path[0..path.len() - 1])
                                        && let (left @ None, right, inner) =
                                            nodes.get_bracket_mut(index)
                                        && *right == b
                                    {
                                        *left = b;
                                        let rest = inner.drain(..i).collect::<Vec<_>>();
                                        nodes.splice(index..index, rest);
                                        path.last_mut().unwrap().0 += i;
                                    } else {
                                        let nodes = self.tree.walk_mut(&path);
                                        let inner = nodes.drain(i..).collect::<Vec<_>>();
                                        nodes.push(new_bracket(b, None, inner));
                                        path.push((i, BracketInner));
                                    }
                                    self.tree_updated((path, 0));
                                }
                                SelectionSpan::Range(r) => {
                                    let nodes = self.tree.walk_mut(&path);
                                    let inner = nodes.drain(r.clone()).collect::<Vec<_>>();
                                    nodes.insert(r.start, new_bracket(b, b, inner));
                                    path.push((r.start, BracketInner));
                                    self.tree_updated((path, 0));
                                }
                            }
                            response.request_redraw();
                            response.consume_event();
                        }
                        Some(b @ (')' | ']' | '}')) => {
                            let b = Some(BracketE::from(b));
                            match span {
                                SelectionSpan::Cursor(i) => {
                                    if i > 0
                                        && let nodes = self.tree.walk_mut(&path)
                                        && let Some(Bracket {
                                            left,
                                            right: right @ None,
                                            ..
                                        }) = nodes.get_mut(i - 1)
                                        && *left == b
                                    {
                                        *right = b;
                                        self.tree_updated((path, i));
                                    } else if let Some((index, BracketInner)) = path.last().cloned()
                                        && let nodes = self.tree.walk_mut(&path[0..path.len() - 1])
                                        && let (left, right @ None, inner) =
                                            nodes.get_bracket_mut(index)
                                        && *left == b
                                    {
                                        *right = b;
                                        let rest = inner.drain(i..).collect::<Vec<_>>();
                                        path.pop();
                                        nodes.splice(index + 1..index + 1, rest);
                                        self.tree_updated((path, index + 1));
                                    } else {
                                        let nodes = self.tree.walk_mut(&path);
                                        let inner = nodes.drain(..i).collect::<Vec<_>>();
                                        nodes.insert(0, new_bracket(None, b, inner));
                                        self.tree_updated((path, 1));
                                    }
                                }
                                SelectionSpan::Range(r) => {
                                    let nodes = self.tree.walk_mut(&path);
                                    let inner = nodes.drain(r.clone()).collect::<Vec<_>>();
                                    nodes.insert(r.start, new_bracket(b, b, inner));
                                    self.tree_updated((path, r.start + 1));
                                }
                            }
                            response.request_redraw();
                            response.consume_event();
                        }
                        Some('|') => {
                            let b = Some(BracketE::Pipe);
                            match span {
                                SelectionSpan::Cursor(i) => {
                                    let nodes = self.tree.walk_mut(&path);
                                    if let Some(Bracket {
                                        left: left @ None,
                                        right,
                                        ..
                                    }) = nodes.get_mut(i)
                                        && *right == b
                                    {
                                        *left = b;
                                        path.push((i, BracketInner));
                                        self.tree_updated((path, 0));
                                    } else if i > 0
                                        && let Some(Bracket {
                                            left,
                                            right: right @ None,
                                            ..
                                        }) = nodes.get_mut(i - 1)
                                        && *left == b
                                    {
                                        *right = b;
                                        self.tree_updated((path, i));
                                    } else if let Some((index, BracketInner)) = path.last().cloned()
                                        && let nodes = self.tree.walk_mut(&path[0..path.len() - 1])
                                        && let (left, right, inner) = nodes.get_bracket_mut(index)
                                        && (left.is_none() && *right == b
                                            || *left == b && right.is_none())
                                    {
                                        if left.is_none() {
                                            *left = b;
                                            let rest = inner.drain(..i).collect::<Vec<_>>();
                                            nodes.splice(index..index, rest);
                                            path.last_mut().unwrap().0 += i;
                                            self.tree_updated((path, 0));
                                        } else {
                                            *right = b;
                                            let rest = inner.drain(i..).collect::<Vec<_>>();
                                            path.pop();
                                            nodes.splice(index + 1..index + 1, rest);
                                            self.tree_updated((path, index + 1));
                                        }
                                    } else {
                                        let nodes = self.tree.walk_mut(&path);
                                        let inner = nodes.drain(i..).collect::<Vec<_>>();
                                        nodes.push(new_bracket(b, None, inner));
                                        path.push((i, BracketInner));
                                        self.tree_updated((path, 0));
                                    }
                                }
                                SelectionSpan::Range(r) => {
                                    let nodes = self.tree.walk_mut(&path);
                                    let inner = nodes.drain(r.clone()).collect::<Vec<_>>();
                                    nodes.insert(r.start, new_bracket(b, b, inner));
                                    path.push((r.start, BracketInner));
                                    self.tree_updated((path, 0));
                                }
                            }
                            response.request_redraw();
                            response.consume_event();
                        }
                        Some('_') => {
                            let nodes = self.tree.walk_mut(&path);
                            let r = span.as_range();
                            if r.end > 0 {
                                let lower = nodes.drain(r.clone()).collect::<Vec<_>>();
                                let sub_len = lower.len();
                                nodes.insert(r.start, new_script_lower(lower));
                                path.push((r.start, ScriptLower));
                                self.tree_updated((path, sub_len));
                                response.request_redraw();
                            }
                            response.consume_event();
                        }
                        Some('^') => {
                            let nodes = self.tree.walk_mut(&path);
                            let r = span.as_range();
                            if r.end > 0 {
                                let upper = nodes.drain(r.clone()).collect::<Vec<_>>();
                                let sup_len = upper.len();
                                nodes.insert(r.start, new_script_upper(upper));
                                path.push((r.start, ScriptUpper));
                                self.tree_updated((path, sup_len));
                                response.request_redraw();
                            }
                            response.consume_event();
                        }
                        Some('/') => {
                            let nodes = self.tree.walk_mut(&path);
                            let r = match span {
                                SelectionSpan::Cursor(c) => {
                                    nodes[..c]
                                        .iter()
                                        .enumerate()
                                        .rev()
                                        .find_map(|(i, (_, n))| {
                                            matches!(
                                                n,
                                                BigOp { .. }
                                                    | Char {
                                                        ch: '+'
                                                            | '-'
                                                            | '*'
                                                            | '='
                                                            | '<'
                                                            | '>'
                                                            | ','
                                                            | ':'
                                                            | '×'
                                                            | '→'
                                                            | '≤'
                                                            | '≥'
                                                            | '⋅',
                                                        ..
                                                    }
                                            )
                                            .then(|| i + 1)
                                        })
                                        .unwrap_or(0)..c
                                }
                                SelectionSpan::Range(r) => r,
                            };

                            let num = nodes.drain(r.clone()).collect::<Vec<_>>();
                            nodes.insert(r.start, new_frac(num, vec![]));
                            path.push((r.start, if r.is_empty() { FracNum } else { FracDen }));
                            self.tree_updated((path, 0));
                            response.consume_event();
                            response.request_redraw();
                        }
                        Some(
                            c @ ('0'..='9'
                            | 'A'..='Z'
                            | 'a'..='z'
                            | '.'
                            | '+'
                            | '-'
                            | '*'
                            | '='
                            | '<'
                            | '>'
                            | ','
                            | ':'
                            | '!'
                            | '%'
                            | '\''),
                        ) => {
                            let cursor = self.tree.walk_mut(&path).add_char(path, span, c);
                            self.tree_updated(cursor);
                            response.request_redraw();
                            response.consume_event();
                        }
                        _ => {}
                    },
                    _ => {}
                }
            }
            Event::CursorMoved { .. } if self.dragging => {
                if let Some(hovered) = hovered {
                    let anchor = self.selection.take().unwrap().anchor;
                    self.set_selection((anchor, hovered));
                } else {
                    eprintln!("how did we get here?");
                }
                response.consume_event();
                response.request_redraw();
            }
            Event::MouseWheel(DVec2 { x, y }) if hovered.is_some() => {
                if x.abs() > y.abs() {
                    self.scroll -= x;
                    response.consume_event();
                    response.request_redraw();
                }
            }
            Event::MouseInput(ElementState::Pressed, MouseButton::Left) => {
                if let Some(hovered) = hovered {
                    self.dragging = true;
                    self.set_cursor(hovered);
                    response.consume_event();
                    response.request_redraw();
                } else if self.has_focus() {
                    response = response.or(self.unfocus());
                }
            }
            Event::MouseInput(ElementState::Released, MouseButton::Left) if self.dragging => {
                self.dragging = false;
                response.consume_event();
            }
            _ => {}
        }

        if self.selection_was_set {
            self.selection_was_set = false;
            let focus = &self.selection.as_ref().unwrap().focus;
            let tree = self.tree.walk(&focus.path);
            let x = if tree.is_empty() {
                tree.bounds.left()
            } else {
                tree.nodes
                    .get(focus.index)
                    .map_or(tree.bounds.right(), |(b, _)| b.left())
            };
            let x = -self.scroll + self.scale * (self.left_padding + x);
            const CURSOR_EDGE: f64 = 0.8;
            let cursor_edge = self.scale * CURSOR_EDGE;
            let x1 = if bounds.size.x < 2.0 * cursor_edge {
                bounds.size.x / 2.0
            } else {
                x.clamp(cursor_edge, bounds.size.x - cursor_edge)
            };
            self.scroll += x - x1;
        }

        self.scroll = self
            .scroll
            .min(self.expression_size().x - bounds.size.x)
            .max(0.0);

        (response, message)
    }

    pub fn render(
        &mut self,
        ctx: &Context,
        bounds: Bounds,
        draw_quad: &mut impl FnMut(DVec2, DVec2, QuadKind),
    ) {
        let top_left = bounds.pos * ctx.scale_factor;
        let bottom_right = (bounds.pos + bounds.size) * ctx.scale_factor;
        let draw_quad = &mut |p0: DVec2, p1: DVec2, mut kind: QuadKind| {
            let q0 = p0.clamp(top_left, bottom_right);
            let q1 = p1.clamp(top_left, bottom_right);
            match &mut kind {
                QuadKind::MsdfGlyph(uv0, uv1) | QuadKind::TranslucentMsdfGlyph(uv0, uv1) => {
                    *uv0 = mix(*uv0, *uv1, (q0 - p0) / (p1 - p0));
                    *uv1 = mix(*uv0, *uv1, (q1 - p0) / (p1 - p0));
                }
                _ => {}
            }
            draw_quad(q0, q1, kind)
        };
        let height = self.tree.bounds.height;
        let transform = &|p| {
            (bounds.pos - dvec2(self.scroll, 0.0)
                + self.scale * (p + dvec2(self.left_padding, height)))
                * ctx.scale_factor
        };
        match &self.selection {
            Some(selection) => {
                let selection: Selection = selection.into();
                let nodes = self.tree.walk_mut(&selection.path);
                let original_gray = nodes.has_gray_background;
                nodes.has_gray_background = false;
                self.tree
                    .render_selection(ctx, &selection, transform, draw_quad);
                self.tree.render(ctx, transform, draw_quad);
                self.tree.walk_mut(&selection.path).has_gray_background = original_gray;
            }
            None => {
                self.tree.render(ctx, transform, draw_quad);
            }
        }

        if self.scroll > 0.0 {
            // The order of the first two arguments determines which the direction of the gradient
            draw_quad(
                dvec2(
                    top_left.x + ctx.scale_factor * self.scale * self.overflow_gradient_width,
                    bottom_right.y,
                ),
                top_left,
                QuadKind::TransparentToWhiteGradient,
            );
        }

        if self.scroll < self.expression_size().x - bounds.size.x {
            // The order of the first two arguments determines which the direction of the gradient
            draw_quad(
                dvec2(
                    bottom_right.x - ctx.scale_factor * self.scale * self.overflow_gradient_width,
                    top_left.y,
                ),
                bottom_right,
                QuadKind::TransparentToWhiteGradient,
            );
        }
    }
}
