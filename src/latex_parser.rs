use std::ops::Range;

use logos::Logos;

use crate::latex_tree::*;

#[derive(Logos, Debug, PartialEq, Clone)]
#[logos(skip r"[ \t\n\f]+")]
pub enum TokenKind<'a> {
    #[token("{")]
    BeginGroup,

    #[token("}")]
    EndGroup,

    #[token(r"\left")]
    Left,

    #[token(r"\right")]
    Right,

    #[token("_")]
    Subscript,

    #[token("^")]
    Superscript,

    #[token(r"\sqrt")]
    Sqrt,

    #[token(r"\frac")]
    Frac,

    #[token(r"\operatorname")]
    Operatorname,

    #[regex(r"\\[a-zA-Z]+", |lex| &lex.slice()[1..])]
    CtrlSeq(&'a str),

    #[regex(
        r"[a-zA-Z0-9()\[\]|.+\-*\/=<>,:!]|\\[{}% ]",
        |lex| lex.slice().chars().find(|c| *c != '\\').expect(r"'\' is not a Char")
    )]
    Char(char),

    EndOfInput,
}

pub type Span = Range<usize>;

type Tk<'a> = TokenKind<'a>;

#[derive(Debug, PartialEq, Clone)]
pub enum ParseError<'a> {
    UnknownToken {
        token: &'a str,
        span: Span,
    },
    UnexpectedToken {
        expected: TokenKind<'a>,
        found: Token<'a>,
    },
    ExpectedBracket {
        after: Token<'a>,
        found: Token<'a>,
    },
    DoubleSubscript {
        first: Token<'a>,
        second: Token<'a>,
    },
    DoubleSuperscript {
        first: Token<'a>,
        second: Token<'a>,
    },
    ExpectedArgument {
        expecter: Token<'a>,
        arg_index: usize,
        found: Token<'a>,
    },
}

#[derive(Debug, PartialEq, Clone)]
pub struct Token<'a> {
    kind: TokenKind<'a>,
    span: Span,
}

struct Lexer<'a> {
    logos: logos::Lexer<'a, Tk<'a>>,
    peeked: Option<Result<Token<'a>, ParseError<'a>>>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            logos: Tk::lexer(source),
            peeked: None,
        }
    }

    fn peek(&mut self) -> Result<&Token<'a>, ParseError<'a>> {
        self.peeked
            .get_or_insert_with(|| {
                let kind = self.logos.next().unwrap_or(Ok(Tk::EndOfInput));
                let span = self.logos.span();
                match kind {
                    Ok(kind) => Ok(Token { kind, span }),
                    Err(()) => Err(ParseError::UnknownToken {
                        token: self.logos.slice(),
                        span,
                    }),
                }
            })
            .as_ref()
            .map_err(|e| e.clone())
    }

    fn next(&mut self) -> Result<Token<'a>, ParseError<'a>> {
        self.peek()?;
        self.peeked
            .take()
            .expect("`peeked` should not be `None` after calling `peek()`")
    }

    fn expect(&mut self, kind: Tk<'a>) -> Result<Token<'a>, ParseError<'a>> {
        let next = self.next()?;

        if next.kind != kind {
            Err(ParseError::UnexpectedToken {
                expected: kind,
                found: next,
            })
        } else {
            Ok(next)
        }
    }
}

#[derive(PartialEq)]
enum NodesKind<'a> {
    Content,
    Argument {
        expecter: Token<'a>,
        arg_index: usize,
    },
    Optional,
}

fn parse_nodes<'a>(
    lexer: &mut Lexer<'a>,
    contents: &mut Nodes<'a>,
    kind: NodesKind<'a>,
) -> Result<(), ParseError<'a>> {
    let parse = |lexer: &mut Lexer<'a>, kind: NodesKind<'a>| {
        let mut nodes = vec![];
        parse_nodes(lexer, &mut nodes, kind).map(|_| nodes)
    };
    loop {
        match lexer.peek()?.kind {
            Tk::BeginGroup => {
                lexer.next()?;
                parse_nodes(lexer, contents, NodesKind::Content)?;
                lexer.expect(Tk::EndGroup)?;
            }
            Tk::Left => {
                let left = lexer.next()?;
                let mut inner = vec![];
                let token = lexer.next()?;
                match token.kind {
                    Tk::Char(c @ ('(' | '[' | '{' | '|')) => inner.push(Node::Char(c)),
                    _ => {
                        return Err(ParseError::ExpectedBracket {
                            after: left,
                            found: token,
                        });
                    }
                };
                parse_nodes(lexer, &mut inner, NodesKind::Content)?;
                let right = lexer.expect(Tk::Right)?;
                let token = lexer.next()?;
                match token.kind {
                    TokenKind::Char(c @ (')' | ']' | '}' | '|')) => inner.push(Node::Char(c)),
                    _ => {
                        return Err(ParseError::ExpectedBracket {
                            after: right,
                            found: token,
                        });
                    }
                }
                contents.push(Node::DelimitedGroup(inner));
            }
            Tk::EndGroup | Tk::Right | Tk::EndOfInput => match kind {
                NodesKind::Argument {
                    expecter,
                    arg_index,
                } => {
                    return Err(ParseError::ExpectedArgument {
                        expecter,
                        arg_index,
                        found: lexer.next()?,
                    });
                }
                _ => break,
            },
            Tk::Subscript | Tk::Superscript => {
                let mut subscript = None;
                let mut superscript = None;

                loop {
                    match lexer.peek()?.kind {
                        Tk::Subscript => {
                            let token = lexer.next()?;
                            if let Some((first, _)) = subscript {
                                return Err(ParseError::DoubleSubscript {
                                    first,
                                    second: token,
                                });
                            }
                            subscript = Some((
                                token.clone(),
                                parse(
                                    lexer,
                                    NodesKind::Argument {
                                        expecter: token,
                                        arg_index: 0,
                                    },
                                )?,
                            ));
                        }
                        Tk::Superscript => {
                            let token = lexer.next()?;
                            if let Some((first, _)) = superscript {
                                return Err(ParseError::DoubleSuperscript {
                                    first,
                                    second: token,
                                });
                            }

                            superscript = Some((
                                token.clone(),
                                parse(
                                    lexer,
                                    NodesKind::Argument {
                                        expecter: token,
                                        arg_index: 0,
                                    },
                                )?,
                            ));
                        }
                        _ => break,
                    }
                }

                contents.push(Node::SubSup {
                    sub: subscript.map(|s| s.1),
                    sup: superscript.map(|s| s.1),
                });
            }
            Tk::Sqrt => {
                let token = lexer.next()?;
                contents.push(Node::Sqrt {
                    root: match lexer.peek()?.kind {
                        Tk::Char('[') => {
                            lexer.next()?;
                            let nodes = parse(lexer, NodesKind::Optional)?;
                            lexer.expect(Tk::Char(']'))?;
                            Some(nodes)
                        }
                        _ => None,
                    },
                    arg: {
                        parse(
                            lexer,
                            NodesKind::Argument {
                                expecter: token,
                                arg_index: 0,
                            },
                        )?
                    },
                });
            }
            Tk::Frac => {
                let token = lexer.next()?;
                contents.push(Node::Frac {
                    num: parse(
                        lexer,
                        NodesKind::Argument {
                            expecter: token.clone(),
                            arg_index: 0,
                        },
                    )?,
                    den: parse(
                        lexer,
                        NodesKind::Argument {
                            expecter: token.clone(),
                            arg_index: 1,
                        },
                    )?,
                });
            }
            Tk::Operatorname => {
                let token = lexer.next()?;
                contents.push(Node::Operatorname(parse(
                    lexer,
                    NodesKind::Argument {
                        expecter: token.clone(),
                        arg_index: 0,
                    },
                )?));
            }
            Tk::CtrlSeq(word) => {
                lexer.next()?;
                contents.push(Node::CtrlSeq(word));
            }
            Tk::Char(']') if kind == NodesKind::Optional => break,
            Tk::Char(c) => {
                lexer.next()?;
                contents.push(Node::Char(c));
            }
        };

        if matches!(kind, NodesKind::Argument { .. }) {
            break;
        }
    }

    Ok(())
}

pub fn parse_latex(source: &str) -> Result<Nodes, ParseError> {
    let mut lexer = Lexer::new(source);
    let mut nodes = vec![];
    parse_nodes(&mut lexer, &mut nodes, NodesKind::Content)?;
    lexer.expect(Tk::EndOfInput)?;
    Ok(nodes)
}

#[cfg(test)]
mod tests {
    use std::vec;

    use super::*;

    fn test_parse(source: &str, expected: Result<Nodes, ParseError>) {
        let result = parse_latex(source);

        if result.is_ok() == expected.is_ok() {
            pretty_assertions::assert_eq!(result, expected, "failed parse r\"{source}\"");
        } else {
            assert_eq!(result, expected, "failed parse r\"{source}\"");
        }
    }

    #[test]
    fn test_parse_latex() {
        use Node::*;
        use ParseError::*;

        test_parse(
            r"[a.b \pi( \%5 \tau\}-,",
            Ok(vec![
                Char('['),
                Char('a'),
                Char('.'),
                Char('b'),
                CtrlSeq("pi"),
                Char('('),
                Char('%'),
                Char('5'),
                CtrlSeq("tau"),
                Char('}'),
                Char('-'),
                Char(','),
            ]),
        );
        test_parse(
            "172#83",
            Err(UnknownToken {
                token: "#",
                span: 3..4,
            }),
        );
        test_parse(
            r"\2",
            Err(UnknownToken {
                token: r"\",
                span: 0..1,
            }),
        );
        test_parse(
            r"3 4{a\pi{\phi}}6",
            Ok(vec![
                Char('3'),
                Char('4'),
                Char('a'),
                CtrlSeq("pi"),
                CtrlSeq("phi"),
                Char('6'),
            ]),
        );
        test_parse(
            "34 {4{ }",
            Err(UnexpectedToken {
                expected: Tk::EndGroup,
                found: Token {
                    kind: Tk::EndOfInput,
                    span: 8..8,
                },
            }),
        );
        test_parse(
            "5}",
            Err(UnexpectedToken {
                expected: Tk::EndOfInput,
                found: Token {
                    kind: Tk::EndGroup,
                    span: 1..2,
                },
            }),
        );
        test_parse(
            "1{2(3}4)5",
            Ok(vec![
                Char('1'),
                Char('2'),
                Char('('),
                Char('3'),
                Char('4'),
                Char(')'),
                Char('5'),
            ]),
        );
        test_parse(
            r"23 \left(4 5{0\pi}\right \} {}",
            Ok(vec![
                Char('2'),
                Char('3'),
                DelimitedGroup(vec![
                    Char('('),
                    Char('4'),
                    Char('5'),
                    Char('0'),
                    CtrlSeq("pi"),
                    Char('}'),
                ]),
            ]),
        );
        test_parse(
            r"23 \left34 5{0\pi}\right \} {}",
            Err(ExpectedBracket {
                after: Token {
                    kind: Tk::Left,
                    span: 3..8,
                },
                found: Token {
                    kind: Tk::Char('3'),
                    span: 8..9,
                },
            }),
        );
        test_parse(
            r"23 \left(4 5{0\pi}\right \pi \} {}",
            Err(ExpectedBracket {
                after: Token {
                    kind: Tk::Right,
                    span: 18..24,
                },
                found: Token {
                    kind: Tk::CtrlSeq("pi"),
                    span: 25..28,
                },
            }),
        );
        test_parse(
            r"23 \left(4 5{0\pi}}\right \} {}",
            Err(UnexpectedToken {
                expected: Tk::Right,
                found: Token {
                    kind: Tk::EndGroup,
                    span: 18..19,
                },
            }),
        );
        test_parse(
            r"\left({\left(\right)}\right)",
            Ok(vec![DelimitedGroup(vec![
                Char('('),
                DelimitedGroup(vec![Char('('), Char(')')]),
                Char(')'),
            ])]),
        );
        test_parse(
            "2^7",
            Ok(vec![
                Char('2'),
                SubSup {
                    sub: None,
                    sup: Some(vec![Char('7')]),
                },
            ]),
        );
        test_parse(
            "j^1_ 23 5",
            Ok(vec![
                Char('j'),
                SubSup {
                    sub: Some(vec![Char('2')]),
                    sup: Some(vec![Char('1')]),
                },
                Char('3'),
                Char('5'),
            ]),
        );
        test_parse(
            "j_{      2 }",
            Ok(vec![
                Char('j'),
                SubSup {
                    sub: Some(vec![Char('2')]),
                    sup: None,
                },
            ]),
        );
        test_parse(
            "a^{1+2}3",
            Ok(vec![
                Char('a'),
                SubSup {
                    sub: None,
                    sup: Some(vec![Char('1'), Char('+'), Char('2')]),
                },
                Char('3'),
            ]),
        );
        test_parse(
            "a^5_ 23 {1 +4}5",
            Ok(vec![
                Char('a'),
                SubSup {
                    sub: Some(vec![Char('2')]),
                    sup: Some(vec![Char('5')]),
                },
                Char('3'),
                Char('1'),
                Char('+'),
                Char('4'),
                Char('5'),
            ]),
        );
        test_parse(r"\sin{2}3", Ok(vec![CtrlSeq("sin"), Char('2'), Char('3')]));
        test_parse(r"\sin s", Ok(vec![CtrlSeq("sin"), Char('s')]));
        test_parse(r"\sin2", Ok(vec![CtrlSeq("sin"), Char('2')]));
        test_parse(
            "a^5_7_a^6_b",
            Err(DoubleSubscript {
                first: Token {
                    kind: Tk::Subscript,
                    span: 3..4,
                },
                second: Token {
                    kind: Tk::Subscript,
                    span: 5..6,
                },
            }),
        );
        test_parse(
            "a^5_7^a^6_b",
            Err(DoubleSuperscript {
                first: Token {
                    kind: Tk::Superscript,
                    span: 1..2,
                },
                second: Token {
                    kind: Tk::Superscript,
                    span: 5..6,
                },
            }),
        );
        test_parse(
            "f_{o{ob }ar}",
            Ok(vec![
                Char('f'),
                SubSup {
                    sub: Some(vec![Char('o'), Char('o'), Char('b'), Char('a'), Char('r')]),
                    sup: None,
                },
            ]),
        );
        test_parse(
            r"1^\left(2\right)3",
            Ok(vec![
                Char('1'),
                SubSup {
                    sub: None,
                    sup: Some(vec![DelimitedGroup(vec![Char('('), Char('2'), Char(')')])]),
                },
                Char('3'),
            ]),
        );
        test_parse(
            "5^(1)",
            Ok(vec![
                Char('5'),
                SubSup {
                    sub: None,
                    sup: Some(vec![Char('(')]),
                },
                Char('1'),
                Char(')'),
            ]),
        );
        test_parse(
            "{5^}",
            Err(ExpectedArgument {
                expecter: Token {
                    kind: Tk::Superscript,
                    span: 2..3,
                },
                arg_index: 0,
                found: Token {
                    kind: Tk::EndGroup,
                    span: 3..4,
                },
            }),
        );
        test_parse(
            r"\left(5_\right)",
            Err(ExpectedArgument {
                expecter: Token {
                    kind: Tk::Subscript,
                    span: 7..8,
                },
                arg_index: 0,
                found: Token {
                    kind: Tk::Right,
                    span: 8..14,
                },
            }),
        );
        test_parse(
            r"5_",
            Err(ExpectedArgument {
                expecter: Token {
                    kind: Tk::Subscript,
                    span: 1..2,
                },
                arg_index: 0,
                found: Token {
                    kind: Tk::EndOfInput,
                    span: 2..2,
                },
            }),
        );
        test_parse(
            r"5_\pi",
            Ok(vec![
                Char('5'),
                SubSup {
                    sub: Some(vec![CtrlSeq("pi")]),
                    sup: None,
                },
            ]),
        );
        test_parse(
            r"\frac{2+4} 56",
            Ok(vec![
                Frac {
                    num: vec![Char('2'), Char('+'), Char('4')],
                    den: vec![Char('5')],
                },
                Char('6'),
            ]),
        );
        test_parse(
            r"\left(\frac\right)",
            Err(ExpectedArgument {
                expecter: Token {
                    kind: Tk::Frac,
                    span: 6..11,
                },
                arg_index: 0,
                found: Token {
                    kind: Tk::Right,
                    span: 11..17,
                },
            }),
        );
        test_parse(
            r"{\frac5}",
            Err(ExpectedArgument {
                expecter: Token {
                    kind: Tk::Frac,
                    span: 1..6,
                },
                arg_index: 1,
                found: Token {
                    kind: Tk::EndGroup,
                    span: 7..8,
                },
            }),
        );
        test_parse(
            r"\frac{}",
            Err(ExpectedArgument {
                expecter: Token {
                    kind: Tk::Frac,
                    span: 0..5,
                },
                arg_index: 1,
                found: Token {
                    kind: Tk::EndOfInput,
                    span: 7..7,
                },
            }),
        );
        test_parse(
            r"\frac\left(1+2\right)\left(3\right)",
            Ok(vec![Frac {
                num: vec![DelimitedGroup(vec![
                    Char('('),
                    Char('1'),
                    Char('+'),
                    Char('2'),
                    Char(')'),
                ])],
                den: vec![DelimitedGroup(vec![Char('('), Char('3'), Char(')')])],
            }]),
        );
        test_parse(
            r"\sqrt{2}3",
            Ok(vec![
                Sqrt {
                    root: None,
                    arg: vec![Char('2')],
                },
                Char('3'),
            ]),
        );
        test_parse(
            r"\sqrt[2]{2}",
            Ok(vec![Sqrt {
                root: Some(vec![Char('2')]),
                arg: vec![Char('2')],
            }]),
        );
        test_parse(
            r"\sqrt[2]34",
            Ok(vec![
                Sqrt {
                    root: Some(vec![Char('2')]),
                    arg: vec![Char('3')],
                },
                Char('4'),
            ]),
        );
        test_parse(
            r"\sqrt\left[n\right]{x}",
            Ok(vec![
                Sqrt {
                    root: None,
                    arg: vec![DelimitedGroup(vec![Char('['), Char('n'), Char(']')])],
                },
                Char('x'),
            ]),
        );
        test_parse(
            r"\sqrt{a}{b}",
            Ok(vec![
                Sqrt {
                    root: None,
                    arg: vec![Char('a')],
                },
                Char('b'),
            ]),
        );
        test_parse(
            r"\sqrt[1]{2\left(3}4\right)5",
            Err(UnexpectedToken {
                expected: Tk::Right,
                found: Token {
                    kind: Tk::EndGroup,
                    span: 17..18,
                },
            }),
        );
        test_parse(
            r"\sqrt[\left(1]{23\right)}45",
            Err(UnexpectedToken {
                expected: Tk::EndGroup,
                found: Token {
                    kind: Tk::Right,
                    span: 17..23,
                },
            }),
        );
        test_parse(
            r"\sqrt[\left[1\right]]2",
            Ok(vec![Sqrt {
                root: Some(vec![DelimitedGroup(vec![Char('['), Char('1'), Char(']')])]),
                arg: vec![Char('2')],
            }]),
        );
        test_parse(
            r"\sqrt[[{]}]1",
            Ok(vec![Sqrt {
                root: Some(vec![Char('['), Char(']')]),
                arg: vec![Char('1')],
            }]),
        );
        test_parse(
            r"\sqrt\frac123",
            Ok(vec![
                Sqrt {
                    root: None,
                    arg: vec![Frac {
                        num: vec![Char('1')],
                        den: vec![Char('2')],
                    }],
                },
                Char('3'),
            ]),
        );
        test_parse(
            r"\log_\frac123",
            Ok(vec![
                CtrlSeq("log"),
                SubSup {
                    sub: Some(vec![Frac {
                        num: vec![Char('1')],
                        den: vec![Char('2')],
                    }]),
                    sup: None,
                },
                Char('3'),
            ]),
        );
        test_parse(
            r"\sqrt\left\{1\right\}",
            Ok(vec![Sqrt {
                root: None,
                arg: vec![DelimitedGroup(vec![Char('{'), Char('1'), Char('}')])],
            }]),
        );
        test_parse(
            r"\sqrt\left1\right)",
            Err(ExpectedBracket {
                after: Token {
                    kind: Tk::Left,
                    span: 5..10,
                },
                found: Token {
                    kind: Tk::Char('1'),
                    span: 10..11,
                },
            }),
        );
        test_parse(
            r"\sqrt(1)",
            Ok(vec![
                Sqrt {
                    root: None,
                    arg: vec![Char('(')],
                },
                Char('1'),
                Char(')'),
            ]),
        );
        test_parse(
            r"\sqrt[\left([\right])]1",
            Ok(vec![Sqrt {
                root: Some(vec![
                    DelimitedGroup(vec![Char('('), Char('['), Char(']')]),
                    Char(')'),
                ]),
                arg: vec![Char('1')],
            }]),
        );
        test_parse(
            r"\sqrt[[]]1",
            Ok(vec![
                Sqrt {
                    root: Some(vec![Char('[')]),
                    arg: vec![Char(']')],
                },
                Char('1'),
            ]),
        );
        test_parse(
            r"\left[(1\right)]",
            Ok(vec![
                DelimitedGroup(vec![Char('['), Char('('), Char('1'), Char(')')]),
                Char(']'),
            ]),
        );
        test_parse(
            r"\left(\sqrt\right)",
            Err(ExpectedArgument {
                expecter: Token {
                    kind: Tk::Sqrt,
                    span: 6..11,
                },
                arg_index: 0,
                found: Token {
                    kind: Tk::Right,
                    span: 11..17,
                },
            }),
        );
        test_parse(
            r"\sqrt[\left()]2",
            Err(UnexpectedToken {
                expected: Tk::Right,
                found: Token {
                    kind: Tk::EndOfInput,
                    span: 15..15,
                },
            }),
        );
        test_parse(
            r"\left(\sqrt\left((1\right)\right))",
            Ok(vec![
                DelimitedGroup(vec![
                    Char('('),
                    Sqrt {
                        root: None,
                        arg: vec![DelimitedGroup(vec![
                            Char('('),
                            Char('('),
                            Char('1'),
                            Char(')'),
                        ])],
                    },
                    Char(')'),
                ]),
                Char(')'),
            ]),
        );
        test_parse(
            r"\operatorname{hello}",
            Ok(vec![Operatorname(vec![
                Char('h'),
                Char('e'),
                Char('l'),
                Char('l'),
                Char('o'),
            ])]),
        );
    }
}
