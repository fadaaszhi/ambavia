use std::fmt;

pub type Nodes<'a> = Vec<Node<'a>>;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Bracket {
    Paren,
    Square,
    Brace,
    Pipe,
}

impl TryFrom<char> for Bracket {
    type Error = ();

    fn try_from(value: char) -> Result<Self, Self::Error> {
        Ok(match value {
            '(' | ')' => Bracket::Paren,
            '[' | ']' => Bracket::Square,
            '{' | '}' => Bracket::Brace,
            '|' => Bracket::Pipe,
            _ => return Err(()),
        })
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Node<'a> {
    DelimitedGroup {
        left: Bracket,
        right: Bracket,
        inner: Nodes<'a>,
    },
    SubSup {
        sub: Option<Nodes<'a>>,
        sup: Option<Nodes<'a>>,
    },
    Sqrt {
        root: Option<Nodes<'a>>,
        arg: Nodes<'a>,
    },
    Frac {
        num: Nodes<'a>,
        den: Nodes<'a>,
    },
    Operatorname(Nodes<'a>),
    CtrlSeq(&'a str),
    Char(char),
}

impl<'a> Node<'a> {
    pub fn to_small_string(&self) -> String {
        match self {
            Node::DelimitedGroup { .. } => r"'\left'".into(),
            Node::SubSup { sub: Some(_), .. } => r"'_'".into(),
            Node::SubSup { .. } => r"'^'".into(),
            Node::Sqrt { .. } => r"'\sqrt'".into(),
            Node::Frac { .. } => r"'\frac'".into(),
            Node::Operatorname(_) => r"'\operatorname'".into(),
            Node::CtrlSeq(word) => format!(r"'\{word}'"),
            Node::Char(c) => format!("'{c}'"),
        }
    }
}

pub struct NodesDisplayer<'a>(pub &'a [Node<'a>]);

impl<'a> fmt::Display for NodesDisplayer<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for node in self.0 {
            write!(f, "{node}")?;
        }
        Ok(())
    }
}

impl<'a> fmt::Display for Node<'a> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Node::DelimitedGroup { left, right, inner } => {
                write!(
                    f,
                    r"\left{}{}\right{}",
                    match left {
                        Bracket::Paren => '(',
                        Bracket::Square => '[',
                        Bracket::Brace => '{',
                        Bracket::Pipe => '|',
                    },
                    NodesDisplayer(inner),
                    match right {
                        Bracket::Paren => ')',
                        Bracket::Square => ']',
                        Bracket::Brace => '}',
                        Bracket::Pipe => '|',
                    },
                )?;
            }
            Node::SubSup {
                sub: subscript,
                sup: superscript,
            } => {
                if let Some(subscript) = subscript {
                    write!(f, "_{{{}}}", NodesDisplayer(subscript))?;
                }
                if let Some(superscript) = superscript {
                    write!(f, "^{{{}}}", NodesDisplayer(superscript))?;
                }
            }
            Node::Sqrt { root, arg } => {
                write!(f, r"\sqrt")?;
                if let Some(root) = root {
                    write!(f, "[{{{}}}]", NodesDisplayer(root))?;
                }
                write!(f, "{{{}}}", NodesDisplayer(arg))?;
            }
            Node::Frac { num, den } => write!(
                f,
                r"\frac{{{}}}{{{}}}",
                NodesDisplayer(num),
                NodesDisplayer(den)
            )?,
            Node::Operatorname(name) => write!(f, r"\operatorname{{{}}}", NodesDisplayer(name),)?,
            Node::CtrlSeq(word) => write!(f, r"\{word} ")?,
            Node::Char(c) => match c {
                '{' | '}' | '%' => write!(f, r"\{c}")?,
                _ => write!(f, "{c}")?,
            },
        }
        Ok(())
    }
}

pub trait ToString {
    fn to_string(self) -> String;
}

impl ToString for &[Node<'_>] {
    fn to_string(self) -> String {
        NodesDisplayer(self).to_string()
    }
}
