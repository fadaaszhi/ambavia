use std::fmt;

pub type Nodes<'a> = Vec<Node<'a>>;

#[derive(Debug, PartialEq)]
pub enum Node<'a> {
    DelimitedGroup(Nodes<'a>),
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
            Node::DelimitedGroup(_) => r"'\left'".into(),
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
            Node::DelimitedGroup(nodes) => {
                write!(
                    f,
                    r"\left{}{}\right{}",
                    nodes[0],
                    NodesDisplayer(&nodes[1..nodes.len() - 1]),
                    nodes[nodes.len() - 1]
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
                '{' | '}' | '%' => write!(f, r"\{}", c)?,
                _ => write!(f, "{}", c)?,
            },
        }
        Ok(())
    }
}
