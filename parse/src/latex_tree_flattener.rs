use crate::latex_tree::Node;

#[derive(Debug, PartialEq, Clone)]
pub enum Token<'a> {
    SubSup {
        sub: Option<&'a [Node<'a>]>,
        sup: Option<&'a [Node<'a>]>,
    },
    Sqrt {
        root: Option<&'a [Node<'a>]>,
        arg: &'a [Node<'a>],
    },
    Frac {
        num: &'a [Node<'a>],
        den: &'a [Node<'a>],
    },
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    LPipe,
    RPipe,
    For,
    With,
    Sum,
    Prod,
    Int,
    Log,
    IdentFrag(String),
    Number(String),
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Plus,
    Minus,
    Asterisk,
    Div,
    Cdot,
    Times,
    Dot,
    Ellipsis,
    Comma,
    Colon,
    Exclamation,
    EndOfGroup,
    EndOfInput,
}

impl<'a> Token<'a> {
    pub fn to_small_string(&self) -> String {
        match self {
            Token::IdentFrag(string) | Token::Number(string) => format!("'{string}'"),
            other => match other {
                Token::IdentFrag(_) | Token::Number(_) => unreachable!(),
                Token::SubSup { sub: Some(_), .. } => r"'_'",
                Token::SubSup { .. } => r"'^'",
                Token::Sqrt { .. } => r"'\sqrt'",
                Token::Frac { .. } => r"'\frac'",
                Token::LParen => "'('",
                Token::RParen => "')'",
                Token::LBracket => "'['",
                Token::RBracket => "']'",
                Token::LBrace => "'{'",
                Token::RBrace => "'}'",
                Token::LPipe => r"'\left|'",
                Token::RPipe => r"'\right|'",
                Token::For => "'for'",
                Token::With => "'with'",
                Token::Sum => r"'\sum'",
                Token::Prod => r"'\prod'",
                Token::Int => r"'\int'",
                Token::Log => r"'\log'",
                Token::Equal => "'='",
                Token::Less => "'<'",
                Token::LessEqual => "'≤'",
                Token::Greater => "'>'",
                Token::GreaterEqual => "'≥'",
                Token::Plus => "'+'",
                Token::Minus => "'-'",
                Token::Asterisk => "'*'",
                Token::Div => "'/'",
                Token::Cdot => r"'\cdot'",
                Token::Times => r"'\times'",
                Token::Dot => "'.'",
                Token::Ellipsis => "'...'",
                Token::Comma => "','",
                Token::Colon => "':'",
                Token::Exclamation => "'!'",
                Token::EndOfGroup => "end of group",
                Token::EndOfInput => "end of input",
            }
            .into(),
        }
    }
}

fn flatten_helper<'a>(
    nodes: &'a [Node],
    in_delimited_group: bool,
    tokens: &mut Vec<Token<'a>>,
) -> Result<(), String> {
    fn skip_space(nodes: &[Node], index: &mut usize) {
        while nodes.get(*index) == Some(&Node::Char(' ')) {
            *index += 1;
        }
    }

    let mut index = 0;

    loop {
        skip_space(nodes, &mut index);
        let Some(node) = nodes.get(index) else {
            break;
        };

        match node {
            Node::DelimitedGroup(nodes) => {
                index += 1;
                flatten_helper(nodes, true, tokens)?
            }
            Node::SubSup { sub, sup } => {
                index += 1;
                tokens.push(Token::SubSup {
                    sub: sub.as_deref(),
                    sup: sup.as_deref(),
                })
            }
            Node::Sqrt { root, arg } => {
                index += 1;
                tokens.push(Token::Sqrt {
                    root: root.as_deref(),
                    arg,
                })
            }
            Node::Frac { num, den } => {
                index += 1;
                tokens.push(Token::Frac { num, den })
            }
            Node::Operatorname(nodes) => {
                index += 1;
                let mut name = "".to_string();

                for node in nodes {
                    match node {
                        Node::Char(c @ ('a'..='z' | 'A'..='Z')) => name.push(*c),
                        _ => {
                            return Err(format!(
                                r"'\operatorname' expected letters, found {}",
                                node.to_small_string()
                            ));
                        }
                    }
                }

                tokens.push(Token::IdentFrag(name));
            }
            Node::CtrlSeq(word) => {
                index += 1;
                tokens.push(Token::IdentFrag(word.to_string()));
            }
            Node::Char('0'..='9' | '.') => {
                let mut string = "".to_string();

                while let Some(Node::Char(digit @ '0'..='9')) = nodes.get(index) {
                    index += 1;
                    string.push(*digit);
                    skip_space(nodes, &mut index);
                }

                if nodes.get(index) == Some(&Node::Char('.')) {
                    if nodes.get(index + 1) == Some(&Node::Char('.'))
                        && nodes.get(index + 2) == Some(&Node::Char('.'))
                    {
                        if string.is_empty() {
                            index += 3;
                            string = "...".into();
                        }
                    } else {
                        index += 1;
                        string.push('.');

                        while let Some(Node::Char(digit @ '0'..='9')) = nodes.get(index) {
                            index += 1;
                            string.push(*digit);
                            skip_space(nodes, &mut index);
                        }
                    }
                }

                tokens.push(match string.as_str() {
                    "." => Token::Dot,
                    "..." => Token::Ellipsis,
                    _ => Token::Number(string),
                });
            }
            Node::Char(c @ ('<' | '>')) => {
                index += 1;
                skip_space(nodes, &mut index);
                tokens.push(if nodes.get(index) == Some(&Node::Char('=')) {
                    index += 1;
                    if *c == '<' {
                        Token::LessEqual
                    } else {
                        Token::GreaterEqual
                    }
                } else if *c == '<' {
                    Token::Less
                } else {
                    Token::Greater
                });
            }
            Node::Char(c) => {
                tokens.push(match c {
                    'a'..='z' | 'A'..='Z' => Token::IdentFrag(c.to_string()),
                    '=' => Token::Equal,
                    '+' => Token::Plus,
                    '-' => Token::Minus,
                    '*' => Token::Asterisk,
                    '/' => Token::Div,
                    '(' => Token::LParen,
                    ')' => Token::RParen,
                    '[' => Token::LBracket,
                    ']' => Token::RBracket,
                    '{' => Token::LBrace,
                    '}' => Token::RBrace,
                    '|' if in_delimited_group && index == 0 => Token::LPipe,
                    '|' if in_delimited_group && index == nodes.len() - 1 => Token::RPipe,
                    '|' => {
                        return Err(r"'|' must be preceeded by '\left' or '\right'".into());
                    }
                    ',' => Token::Comma,
                    ':' => Token::Colon,
                    '!' => Token::Exclamation,
                    _ => return Err(format!("unexpected character '{c}'")),
                });
                index += 1;
            }
        }

        if let Some(Token::IdentFrag(word)) = tokens.last()
            && let Some(new_token) = match word.as_ref() {
                "cdot" => Some(Token::Cdot),
                "times" => Some(Token::Times),
                "div" => Some(Token::Div),
                "for" => Some(Token::For),
                "with" => Some(Token::With),
                "sum" => Some(Token::Sum),
                "prod" => Some(Token::Prod),
                "int" => Some(Token::Int),
                "log" => Some(Token::Log),
                "lt" => Some(Token::Less),
                "le" => Some(Token::LessEqual),
                "leq" => Some(Token::LessEqual),
                "gt" => Some(Token::Greater),
                "ge" => Some(Token::GreaterEqual),
                "geq" => Some(Token::GreaterEqual),
                _ => None,
            }
        {
            *tokens.last_mut().unwrap() = new_token;
        }
    }

    Ok(())
}

pub fn flatten<'a>(nodes: &'a [Node]) -> Result<Vec<Token<'a>>, String> {
    let mut tokens = vec![];
    flatten_helper(nodes, false, &mut tokens)?;
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use Node::*;
    use Token as Tk;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;

    #[test]
    fn number_dot_ellipsis() {
        assert_eq!(flatten(&[Char('1')]), Ok(vec![Tk::Number("1".into())]));

        assert_eq!(
            flatten(&[Char('2'), Char('.')]),
            Ok(vec![Tk::Number("2.".into())])
        );

        assert_eq!(
            flatten(&[Char('2'), Char(' '), Char('.'), Char('.')]),
            Ok(vec![Tk::Number("2.".into()), Tk::Dot])
        );

        assert_eq!(
            flatten(&[Char('2'), Char('8'), Char('.'), Char('5'), Char('0')]),
            Ok(vec![Tk::Number("28.50".into()),])
        );

        assert_eq!(
            flatten(&[Char(' '), Char('.'), Char('5')]),
            Ok(vec![Tk::Number(".5".into()),])
        );

        assert_eq!(
            flatten(&[Char(' '), Char('.'), Char('.')]),
            Ok(vec![Tk::Dot, Tk::Dot,])
        );

        assert_eq!(
            flatten(&[Char('.'), Char('.'), Char('.')]),
            Ok(vec![Tk::Ellipsis,])
        );

        assert_eq!(
            flatten(&[Char('1'), Char('.'), Char('.'), Char('.')]),
            Ok(vec![Tk::Number("1".into()), Tk::Ellipsis,])
        );

        assert_eq!(
            flatten(&[
                Char('1'),
                Char('.'),
                Char('4'),
                Char('.'),
                Char('.'),
                Char('.'),
            ]),
            Ok(vec![Tk::Number("1.4".into()), Tk::Ellipsis,])
        );

        assert_eq!(
            flatten(&[Char('4'), Char('.'), Char('.'), Char('.'), Char('6')]),
            Ok(vec![
                Tk::Number("4".into()),
                Tk::Ellipsis,
                Tk::Number("6".into()),
            ])
        );

        assert_eq!(
            flatten(&[
                Char('4'),
                Char('.'),
                Char('.'),
                Char('.'),
                Char('.'),
                Char('6'),
            ]),
            Ok(vec![
                Tk::Number("4".into()),
                Tk::Ellipsis,
                Tk::Number(".6".into()),
            ])
        );

        assert_eq!(
            flatten(&[
                Char('4'),
                Char('.'),
                Char(' '),
                Char('.'),
                Char('.'),
                Char('6'),
            ]),
            Ok(vec![
                Tk::Number("4.".into()),
                Tk::Dot,
                Tk::Number(".6".into()),
            ])
        );
    }

    #[test]
    fn punctuation() {
        assert_eq!(
            flatten(&[Char(','), Char(':'), Char('!')]),
            Ok(vec![Tk::Comma, Tk::Colon, Tk::Exclamation])
        );
    }

    #[test]
    fn ops() {
        assert_eq!(
            flatten(&[
                Char('+'),
                Char('-'),
                Char('*'),
                Char('/'),
                CtrlSeq(r"cdot"),
                CtrlSeq(r"div"),
                CtrlSeq(r"times"),
                Operatorname(vec![Char('c'), Char('d'), Char('o'), Char('t')]),
                Char('='),
                Char('<'),
                Char('<'),
                Char(' '),
                Char('='),
                Char('>'),
                Char('>'),
                Char('='),
                CtrlSeq(r"le"),
                CtrlSeq(r"lt"),
                CtrlSeq(r"gt"),
                CtrlSeq(r"ge"),
                CtrlSeq(r"leq"),
                CtrlSeq(r"geq"),
            ]),
            Ok(vec![
                Tk::Plus,
                Tk::Minus,
                Tk::Asterisk,
                Tk::Div,
                Tk::Cdot,
                Tk::Div,
                Tk::Times,
                Tk::Cdot,
                Tk::Equal,
                Tk::Less,
                Tk::LessEqual,
                Tk::Greater,
                Tk::GreaterEqual,
                Tk::LessEqual,
                Tk::Less,
                Tk::Greater,
                Tk::GreaterEqual,
                Tk::LessEqual,
                Tk::GreaterEqual,
            ])
        );
    }

    #[test]
    fn keywords() {
        assert_eq!(
            flatten(&[
                CtrlSeq("sum"),
                CtrlSeq("for"),
                CtrlSeq("with"),
                CtrlSeq("prod"),
                CtrlSeq("int"),
                CtrlSeq("log"),
                Operatorname(vec![Char('w'), Char('i'), Char('t'), Char('h')]),
            ]),
            Ok(vec![
                Tk::Sum,
                Tk::For,
                Tk::With,
                Tk::Prod,
                Tk::Int,
                Tk::Log,
                Tk::With,
            ])
        );
    }

    #[test]
    fn ident_frag() {
        assert_eq!(
            flatten(&[
                Char('a'),
                CtrlSeq("pi"),
                Operatorname(vec![Char('p'), Char('i')])
            ]),
            Ok(vec![
                Tk::IdentFrag("a".into()),
                Tk::IdentFrag("pi".into()),
                Tk::IdentFrag("pi".into()),
            ])
        );

        assert_eq!(
            flatten(&[Operatorname(vec![Char('p'), Char(' '), Char('i')])]),
            Err(r"'\operatorname' expected letters, found ' '".into())
        );

        assert_eq!(
            flatten(&[Operatorname(vec![CtrlSeq("pi")])]),
            Err(r"'\operatorname' expected letters, found '\pi'".into())
        );
    }

    #[test]
    fn brackets() {
        assert_eq!(
            flatten(&[
                Char('('),
                Char(')'),
                Char('}'),
                Char('{'),
                Char(']'),
                Char('['),
            ]),
            Ok(vec![
                Tk::LParen,
                Tk::RParen,
                Tk::RBrace,
                Tk::LBrace,
                Tk::RBracket,
                Tk::LBracket,
            ])
        );

        assert_eq!(
            flatten(&[Char('('), Char(')'), Char('|')]),
            Err(r"'|' must be preceeded by '\left' or '\right'".into())
        );
    }

    #[test]
    fn flatten_delimited_groups() {
        assert_eq!(
            flatten(&[
                Char('5'),
                DelimitedGroup(vec![
                    Char('|'),
                    DelimitedGroup(vec![
                        Char('['),
                        CtrlSeq("pi"),
                        Char('3'),
                        Char('.'),
                        Char('7'),
                        Char('.'),
                        Char('|'),
                    ]),
                    Char(')'),
                ]),
                Char('4'),
                Char('.'),
                Char('.'),
                Char('.'),
                DelimitedGroup(vec![Char('|'), Char('a'), Char('.'), Char(')')]),
                Char('6'),
            ]),
            Ok(vec![
                Tk::Number("5".into()),
                Tk::LPipe,
                Tk::LBracket,
                Tk::IdentFrag("pi".into()),
                Tk::Number("3.7".into()),
                Tk::Dot,
                Tk::RPipe,
                Tk::RParen,
                Tk::Number("4".into()),
                Tk::Ellipsis,
                Tk::LPipe,
                Tk::IdentFrag("a".into()),
                Tk::Dot,
                Tk::RParen,
                Tk::Number("6".into()),
            ])
        );
    }

    #[test]
    fn sub_sup() {
        let nodes = [
            SubSup {
                sub: Some(vec![Char('1')]),
                sup: Some(vec![CtrlSeq("pi")]),
            },
            Char('j'),
        ];
        let tokens = flatten(&nodes);
        assert_matches!(
            tokens.as_deref(),
            Ok([
                Tk::SubSup {
                    sub: Some(..),
                    sup: Some(..)
                },
                Tk::IdentFrag(name)
            ]) if name == "j"
        );

        let Ok(
            [
                Tk::SubSup {
                    sub: Some(sub),
                    sup: Some(sup),
                },
                ..,
            ],
        ) = tokens.as_deref()
        else {
            unreachable!();
        };

        assert_eq!(flatten(sub), Ok(vec![Tk::Number("1".into())]));
        assert_eq!(flatten(sup), Ok(vec![Tk::IdentFrag("pi".into())]));
    }

    #[test]
    fn sqrt() {
        let nodes = [
            Sqrt {
                root: Some(vec![Char('1')]),
                arg: vec![CtrlSeq("pi")],
            },
            Char('j'),
        ];
        let tokens = flatten(&nodes);
        assert_matches!(tokens.as_deref(), Ok([Tk::Sqrt { root: Some(..), .. },
                Tk::IdentFrag(name)
            ]) if name == "j"
        );

        let Ok(
            [
                Tk::Sqrt {
                    root: Some(root),
                    arg,
                },
                ..,
            ],
        ) = tokens.as_deref()
        else {
            unreachable!();
        };

        assert_eq!(flatten(root), Ok(vec![Tk::Number("1".into())]));
        assert_eq!(flatten(arg), Ok(vec![Tk::IdentFrag("pi".into())]));
    }

    #[test]
    fn frac() {
        let nodes = [
            Frac {
                num: vec![Char('1')],
                den: vec![CtrlSeq("pi")],
            },
            Char('j'),
        ];

        let tokens = flatten(&nodes);
        assert_matches!(tokens.as_deref(), Ok([Tk::Frac { .. },
                Tk::IdentFrag(name)
            ]) if name == "j"
        );

        let Ok([Tk::Frac { num, den }, ..]) = tokens.as_deref() else {
            unreachable!();
        };

        assert_eq!(flatten(num), Ok(vec![Tk::Number("1".into())]));
        assert_eq!(flatten(den), Ok(vec![Tk::IdentFrag("pi".into())]));
    }
}
