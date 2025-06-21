use crate::{
    ast::*,
    latex_tree::Node,
    latex_tree_flattener::{flatten, Token},
};

struct Tokens<'a> {
    tokens: &'a [Token<'a>],
    index: usize,
    end_token: Token<'a>,
}

impl<'a> Tokens<'a> {
    fn new(tokens: &'a [Token], end_token: Token<'a>) -> Self {
        Self {
            tokens,
            index: 0,
            end_token,
        }
    }

    fn peek(&self) -> &Token<'a> {
        self.tokens.get(self.index).unwrap_or(&self.end_token)
    }

    fn next(&mut self) -> &Token<'a> {
        self.tokens
            .get(self.index)
            .inspect(|_| {
                self.index += 1;
            })
            .unwrap_or(&self.end_token)
    }

    fn expect(&mut self, token: Token) -> Result<&Token<'a>, String> {
        let next = self.next();

        if next == &token {
            Ok(next)
        } else {
            Err(format!(
                "expected {}, found {}",
                token.to_small_string(),
                next.to_small_string()
            ))
        }
    }
}

fn parse_number(tokens: &mut Tokens) -> Result<Expression, String> {
    match tokens.next() {
        Token::Number(x) => Ok(Expression::Number(x.parse().unwrap_or_else(|s| {
            panic!("flattener should have produced valid number, got '{x}' ({s})")
        }))),
        other => Err(format!(
            "expected number, found {}",
            other.to_small_string()
        )),
    }
}

fn parse_ident_frag(tokens: &mut Tokens) -> Result<String, String> {
    match tokens.next() {
        Token::IdentFrag(x) => Ok(x.clone()),
        other => Err(format!(
            "expected identifier, found {}",
            other.to_small_string()
        )),
    }
}

fn parse_nodes_into_name_subscript(nodes: &[Node]) -> Result<String, String> {
    let mut subscript = "".to_string();

    for node in nodes.iter() {
        match node {
            Node::Char(c @ ('0'..='9' | 'a'..='z' | 'A'..='Z')) => subscript.push(*c),
            other => {
                return Err(format!(
                    "name subscript expected letters and digits, found {}",
                    other.to_small_string(),
                ))
            }
        }
    }

    Ok(subscript)
}

fn parse_assignment(tokens: &mut Tokens, min_bp: u8) -> Result<(String, Expression), String> {
    let mut identifier = parse_ident_frag(tokens)?;
    if let Token::SubSup {
        sub: Some(sub),
        sup,
    } = tokens.peek()
    {
        identifier.push('_');
        identifier += &parse_nodes_into_name_subscript(sub)?;

        if sup.is_some() {
            return Err(format!(
                "expected {}, found '^'",
                Token::Equal.to_small_string()
            ));
        }

        tokens.next();
    }
    tokens.expect(Token::Equal)?;
    let expression = parse_expression(tokens, min_bp)?;
    Ok((identifier, expression))
}

const FOR_PRECEDENCE: (u8, u8) = (1, 2);
const WITH_PRECEDENCE: (u8, u8) = (3, 4);

fn get_prefix_op(token: &Token) -> Option<(UnaryOperator, (), u8)> {
    Some(match token {
        Token::Minus => (UnaryOperator::Neg, (), 7),
        _ => return None,
    })
}

fn get_postfix_op(token: &Token) -> Option<(UnaryOperator, u8, ())> {
    Some(match token {
        Token::Exclamation => (UnaryOperator::Fac, 11, ()),
        _ => return None,
    })
}

fn get_infix_op(token: &Token) -> Option<(BinaryOperator, u8, u8)> {
    Some(match token {
        Token::Plus => (BinaryOperator::Add, 5, 6),
        Token::Minus => (BinaryOperator::Sub, 5, 6),
        Token::Asterisk => (BinaryOperator::Mul, 9, 10),
        Token::Div => (BinaryOperator::Div, 9, 10),
        Token::Cdot => (BinaryOperator::Dot, 9, 10),
        Token::Times => (BinaryOperator::Cross, 9, 10),
        _ => return None,
    })
}

pub fn parse_nodes_into_expression(nodes: &[Node], end_token: Token) -> Result<Expression, String> {
    let tokens = flatten(nodes)?;
    let mut tokens = Tokens::new(&tokens, end_token.clone());
    let expression = parse_expression(&mut tokens, 0)?;
    tokens.expect(end_token)?;
    Ok(expression)
}

fn parse_args(
    tokens: &mut Tokens,
    existing_args: Vec<Expression>,
) -> Result<Vec<Expression>, String> {
    tokens.expect(Token::LParen)?;
    let mut args = existing_args;
    let mut first = true;

    while tokens.peek() != &Token::RParen {
        if !first {
            if tokens.peek() != &Token::Comma {
                break;
            }
            tokens.next();
        }
        args.push(parse_expression(tokens, 0)?);
        first = false;
    }

    tokens.expect(Token::RParen)?;
    Ok(args)
}

fn get_comparison_op(token: &Token) -> Option<ComparisonOperator> {
    Some(match token {
        Token::Equal => ComparisonOperator::Equal,
        Token::Less => ComparisonOperator::Less,
        Token::LessEqual => ComparisonOperator::LessEqual,
        Token::Greater => ComparisonOperator::Greater,
        Token::GreaterEqual => ComparisonOperator::GreaterEqual,
        _ => return None,
    })
}

fn parse_list(tokens: &mut Tokens, as_index: bool) -> Result<Expression, String> {
    tokens.expect(Token::LBracket)?;
    let mut list = vec![];
    let mut first = true;
    while !matches!(tokens.peek(), Token::RBracket | Token::Ellipsis) {
        if !first {
            if tokens.peek() != &Token::Comma {
                break;
            }
            tokens.next();
            if tokens.peek() == &Token::Ellipsis {
                break;
            }
        }
        list.push(parse_expression(tokens, FOR_PRECEDENCE.0 + 1)?);
        first = false;
    }

    if as_index && list.len() == 1 && get_comparison_op(tokens.peek()).is_some() {
        let chain = parse_chained_comparison(tokens, list.pop())?;
        tokens.expect(Token::RBracket)?;
        return Ok(Expression::ChainedComparison(chain));
    }

    if list.len() == 1 && tokens.peek() == &Token::For {
        let list_comp = parse_list_comprehension(tokens, list.pop().unwrap(), FOR_PRECEDENCE.1)?;
        tokens.expect(Token::RBracket)?;
        return Ok(list_comp);
    }

    if tokens.peek() == &Token::Ellipsis {
        tokens.next();
        if tokens.peek() == &Token::Comma {
            tokens.next();
        }

        let before_ellipsis = list;
        if before_ellipsis.is_empty() {
            return Err(format!(
                "expected expression, found {}",
                Token::Ellipsis.to_small_string()
            ));
        }

        let mut after_ellipsis = vec![];
        let mut first = true;
        while !matches!(tokens.peek(), Token::RBracket) {
            if !first {
                if tokens.peek() != &Token::Comma {
                    break;
                }
                tokens.next();
            }
            after_ellipsis.push(parse_expression(tokens, 0)?);
            first = false;
        }

        if !as_index && after_ellipsis.is_empty() {
            return Err(format!(
                "expected expression after {}, found {}",
                Token::Ellipsis.to_small_string(),
                tokens.peek().to_small_string()
            ));
        }

        tokens.expect(Token::RBracket)?;
        Ok(Expression::ListRange {
            before_ellipsis,
            after_ellipsis,
        })
    } else {
        tokens.expect(Token::RBracket)?;
        Ok(Expression::List(list))
    }
}

fn parse_assignment_list(
    tokens: &mut Tokens,
    min_bp: u8,
) -> Result<Vec<(String, Expression)>, String> {
    let mut assignments = vec![];
    loop {
        assignments.push(parse_assignment(tokens, min_bp)?);
        if tokens.peek() != &Token::Comma {
            break;
        }
        tokens.next();
    }
    Ok(assignments)
}

fn parse_list_comprehension(
    tokens: &mut Tokens,
    body: Expression,
    min_bp: u8,
) -> Result<Expression, String> {
    tokens.expect(Token::For)?;
    Ok(Expression::For {
        body: Box::new(body),
        lists: parse_assignment_list(tokens, min_bp)?,
    })
}

fn parse_piecewise_case(tokens: &mut Tokens, first: Expression) -> Result<Expression, String> {
    let test = Box::new(Expression::ChainedComparison(parse_chained_comparison(
        tokens,
        Some(first),
    )?));
    let consequent = Box::new(if tokens.peek() == &Token::Colon {
        tokens.next();
        parse_expression(tokens, 0)?
    } else {
        Expression::Number(1.0)
    });
    let alternate = if tokens.peek() == &Token::Comma {
        tokens.next();
        let mut expr = parse_expression(tokens, 0)?;
        if get_comparison_op(tokens.peek()).is_some() {
            expr = parse_piecewise_case(tokens, expr)?
        }
        Some(Box::new(expr))
    } else {
        None
    };
    Ok(Expression::Piecewise {
        test,
        consequent,
        alternate,
    })
}

fn parse_expression(tokens: &mut Tokens, min_bp: u8) -> Result<Expression, String> {
    while let &Token::Plus = tokens.peek() {
        tokens.next();
    }

    let mut left = {
        if let Some((op, (), r_bp)) = get_prefix_op(tokens.peek()) {
            tokens.next();
            let arg = parse_expression(tokens, r_bp)?;
            Expression::UnaryOperation {
                operation: op,
                arg: Box::new(arg),
            }
        } else {
            match tokens.peek() {
                Token::Number(_) => parse_number(tokens)?,
                Token::IdentFrag(_) => Expression::Identifier(parse_ident_frag(tokens)?),
                Token::Frac { num, den } => {
                    let frac = Expression::BinaryOperation {
                        operation: BinaryOperator::Div,
                        left: Box::new(parse_nodes_into_expression(num, Token::EndOfGroup)?),
                        right: Box::new(parse_nodes_into_expression(den, Token::EndOfGroup)?),
                    };
                    tokens.next();
                    frac
                }
                Token::Sqrt { root, arg } => {
                    let arg = Box::new(parse_nodes_into_expression(arg, Token::EndOfGroup)?);
                    let expr = if let Some(root) = root {
                        Expression::BinaryOperation {
                            operation: BinaryOperator::Pow,
                            left: arg,
                            right: Box::new(Expression::BinaryOperation {
                                operation: BinaryOperator::Div,
                                left: Box::new(Expression::Number(1.0)),
                                right: Box::new(parse_nodes_into_expression(
                                    root,
                                    Token::EndOfGroup,
                                )?),
                            }),
                        }
                    } else {
                        Expression::UnaryOperation {
                            operation: UnaryOperator::Sqrt,
                            arg,
                        }
                    };
                    tokens.next();
                    expr
                }
                Token::LParen => {
                    let mut list = parse_args(tokens, vec![])?;
                    match list.len() {
                        0 => return Err("parentheses cannot be empty".into()),
                        1 => list.pop().unwrap(),
                        2 => {
                            let y = list.pop().unwrap();
                            let x = list.pop().unwrap();
                            Expression::BinaryOperation {
                                operation: BinaryOperator::Point,
                                left: Box::new(x),
                                right: Box::new(y),
                            }
                        }
                        _ => return Err("points may only have 2 coordinates".into()),
                    }
                }
                Token::LBracket => parse_list(tokens, false)?,
                Token::LPipe => {
                    tokens.next();
                    let arg = Box::new(parse_expression(tokens, 0)?);
                    tokens.expect(Token::RPipe)?;
                    Expression::UnaryOperation {
                        operation: UnaryOperator::Norm,
                        arg,
                    }
                }
                Token::LBrace => {
                    tokens.next();
                    let piecewise = if tokens.peek() == &Token::RBrace {
                        Expression::Number(1.0)
                    } else {
                        let first = parse_expression(tokens, 0)?;
                        parse_piecewise_case(tokens, first)?
                    };
                    tokens.expect(Token::RBrace)?;
                    piecewise
                }
                Token::Sum | Token::Prod => {
                    let kind_token = tokens.next().clone();
                    let kind = match kind_token {
                        Token::Sum => SumProdKind::Sum,
                        Token::Prod => SumProdKind::Prod,
                        _ => unreachable!(),
                    };
                    let sub_sup = tokens.next();
                    let Token::SubSup { sub, sup } = sub_sup else {
                        return Err(format!(
                            r"{} expected lower and upper bounds, found {}",
                            kind_token.to_small_string(),
                            sub_sup.to_small_string()
                        ));
                    };
                    let Some(sub) = sub else {
                        return Err(format!(
                            r"{} expected lower bound",
                            kind_token.to_small_string(),
                        ));
                    };
                    let sub = flatten(sub)?;
                    let mut sub_tokens = Tokens::new(&sub, Token::EndOfGroup);
                    let (variable, lower_bound) = parse_assignment(&mut sub_tokens, 0)?;
                    sub_tokens.expect(Token::EndOfGroup)?;
                    let Some(sup) = sup else {
                        return Err(format!(
                            r"{} expected upper bound",
                            kind_token.to_small_string()
                        ));
                    };
                    let upper_bound =
                        Box::new(parse_nodes_into_expression(sup, Token::EndOfGroup)?);
                    let body = Box::new(parse_expression(
                        tokens,
                        get_infix_op(&Token::Plus)
                            .expect("'+' should be an operator")
                            .2,
                    )?);
                    Expression::SumProd {
                        kind,
                        variable,
                        lower_bound: Box::new(lower_bound),
                        upper_bound,
                        body,
                    }
                }
                token => {
                    return Err(format!(
                        "expected expression, found {}",
                        token.to_small_string()
                    ))
                }
            }
        }
    };

    loop {
        let ((op, l_bp, r_bp), implicit) = if let Some((op, l_bp, ())) =
            get_postfix_op(tokens.peek())
        {
            if l_bp < min_bp {
                break;
            }

            tokens.next();
            left = Expression::UnaryOperation {
                operation: op,
                arg: Box::new(left),
            };
            continue;
        } else if let Some(info) = get_infix_op(tokens.peek()) {
            (info, false)
        } else {
            match tokens.peek() {
                Token::SubSup { sub, sup } => {
                    if let Some(sub) = sub {
                        let subscript = parse_nodes_into_name_subscript(sub)?;
                        match left {
                            Expression::Identifier(ref mut name) => {
                                *name += "_{";
                                name.push_str(&subscript);
                                *name += "}";
                            }
                            _ => return Err("only identifiers may have subscripts".into()),
                        }
                    }

                    if let Some(sup) = sup {
                        left = Expression::BinaryOperation {
                            operation: BinaryOperator::Pow,
                            left: Box::new(left),
                            right: Box::new(parse_nodes_into_expression(sup, Token::EndOfGroup)?),
                        }
                    }

                    tokens.next();
                    continue;
                }
                Token::LBracket => {
                    left = Expression::BinaryOperation {
                        operation: BinaryOperator::Index,
                        left: Box::new(left),
                        right: Box::new(match parse_list(tokens, true)? {
                            Expression::List(list) if list.is_empty() => {
                                return Err("square brackets cannot be empty".into());
                            }
                            Expression::List(mut list) if list.len() == 1 => list.pop().unwrap(),
                            other => other,
                        }),
                    };
                    continue;
                }
                Token::LParen if matches!(left, Expression::Identifier(_)) => {
                    let Expression::Identifier(callee) = left else {
                        unreachable!();
                    };
                    tokens.next();
                    let mut args = vec![];

                    while tokens.peek() != &Token::RParen {
                        args.push(parse_expression(tokens, 0)?);
                        if tokens.peek() != &Token::Comma {
                            break;
                        }
                        tokens.next();
                    }

                    tokens.expect(Token::RParen)?;
                    left = Expression::CallOrMultiply { callee, args };
                    continue;
                }
                Token::Sqrt { .. }
                | Token::Frac { .. }
                | Token::LParen
                | Token::LBrace
                | Token::LPipe
                | Token::Sum
                | Token::Prod
                | Token::Int
                | Token::Log
                | Token::IdentFrag(_)
                | Token::Number(_) => (
                    get_infix_op(&Token::Asterisk).expect("'*' should be an infix operator"),
                    true,
                ),
                Token::Dot => {
                    tokens.next();
                    let callee = match tokens.next() {
                        Token::IdentFrag(callee) => callee.clone(),
                        other => {
                            return Err(format!(
                                "{} expected identifier, found {}",
                                Token::Dot.to_small_string(),
                                other.to_small_string()
                            ))
                        }
                    };
                    left = match callee.as_str() {
                        "x" => Expression::UnaryOperation {
                            operation: UnaryOperator::PointX,
                            arg: Box::new(left),
                        },
                        "y" => Expression::UnaryOperation {
                            operation: UnaryOperator::PointY,
                            arg: Box::new(left),
                        },
                        _ => {
                            let mut args = vec![left];
                            if tokens.peek() == &Token::LParen {
                                args = parse_args(tokens, args)?;
                            }
                            Expression::Call {
                                callee: callee.clone(),
                                args,
                            }
                        }
                    };
                    continue;
                }
                Token::For => {
                    let (l_bp, r_bp) = FOR_PRECEDENCE;

                    if l_bp < min_bp {
                        break;
                    }

                    left = parse_list_comprehension(tokens, left, r_bp)?;
                    continue;
                }
                Token::With => {
                    let (l_bp, r_bp) = WITH_PRECEDENCE;

                    if l_bp < min_bp {
                        break;
                    }

                    tokens.expect(Token::With)?;
                    left = Expression::With {
                        body: Box::new(left),
                        substitutions: parse_assignment_list(tokens, r_bp)?,
                    };
                    continue;
                }
                _ => break,
            }
        };

        if l_bp < min_bp {
            break;
        }

        if !implicit {
            tokens.next();
        }

        let right = parse_expression(tokens, r_bp)?;
        left = Expression::BinaryOperation {
            operation: op,
            left: Box::new(left),
            right: Box::new(right),
        };
    }

    Ok(left)
}

fn parse_chained_comparison(
    tokens: &mut Tokens,
    first_expression: Option<Expression>,
) -> Result<ChainedComparison, String> {
    let mut operands = vec![];
    let mut operators = vec![];
    let mut first = true;

    if let Some(first_expression) = first_expression {
        first = false;
        operands.push(first_expression);
    }

    loop {
        if !first {
            operators.push(match tokens.peek() {
                Token::Equal => ComparisonOperator::Equal,
                Token::Less => ComparisonOperator::Less,
                Token::LessEqual => ComparisonOperator::LessEqual,
                Token::Greater => ComparisonOperator::Greater,
                Token::GreaterEqual => ComparisonOperator::GreaterEqual,
                other => {
                    if operators.is_empty() {
                        return Err(format!(
                            "expected comparison, found {}",
                            other.to_small_string()
                        ));
                    }
                    break;
                }
            });
            tokens.next();
        }
        first = false;
        operands.push(parse_expression(tokens, 0)?);
    }

    assert!(!operators.is_empty());
    assert_eq!(operands.len(), operators.len() + 1);
    Ok(ChainedComparison {
        operands,
        operators,
    })
}

pub fn parse_expression_list_entry(nodes: &[Node]) -> Result<ExpressionListEntry, String> {
    let tokens = flatten(nodes)?;
    let mut tokens = Tokens::new(&tokens, Token::EndOfInput);
    let entry = parse_expression_list_entry_from_tokens(&mut tokens)?;
    tokens.expect(Token::EndOfInput)?;
    Ok(entry)
}

fn parse_expression_list_entry_from_tokens(
    tokens: &mut Tokens,
) -> Result<ExpressionListEntry, String> {
    let expression = parse_expression(tokens, 0)?;

    if get_comparison_op(tokens.peek()).is_none() {
        return Ok(ExpressionListEntry::Expression(expression));
    }

    let mut chain = parse_chained_comparison(tokens, Some(expression))?;

    if chain.operators.len() == 1 && chain.operators[0] == ComparisonOperator::Equal {
        match &chain.operands[0] {
            Expression::Identifier(name) => {
                return Ok(ExpressionListEntry::Assignment {
                    name: name.clone(),
                    value: chain.operands.pop().unwrap(),
                })
            }
            Expression::CallOrMultiply { callee, args } => {
                if let Some(parameters) = args
                    .iter()
                    .map(|a| match a {
                        Expression::Identifier(name) => Some(name),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>()
                {
                    return Ok(ExpressionListEntry::FunctionDeclaration {
                        name: callee.clone(),
                        parameters: parameters.iter().map(|&p| p.clone()).collect(),
                        body: chain.operands.pop().unwrap(),
                    });
                }
            }
            _ => {}
        }
    }

    Ok(ExpressionListEntry::Relation(chain))
}

#[cfg(test)]
mod tests {
    use crate::latex_tree::Node;

    use super::*;
    use pretty_assertions::assert_eq;
    use BinaryOperator::*;
    use ComparisonOperator::*;
    use Expression::{
        BinaryOperation as Bop, Call, CallOrMultiply as CallMull, For, Identifier as Id, List,
        ListRange, Number as Num, Piecewise, SumProd, UnaryOperation as Uop, With,
    };
    use ExpressionListEntry as Ele;
    use SumProdKind::*;
    use Token as T;
    use UnaryOperator::*;

    fn bx<T>(x: T) -> Box<T> {
        Box::new(x)
    }

    #[test]
    fn number() {
        let tokens = [T::Number("5".into())];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(parse_number(&mut tokens), Ok(Num(5.0)));
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn identifier() {
        let tokens = [T::IdentFrag("poo".into())];
        let mut tokens = Tokens::new(&tokens, Token::EndOfGroup);
        assert_eq!(parse_ident_frag(&mut tokens), Ok("poo".into()));
        assert_eq!(tokens.next(), &Token::EndOfGroup);
    }

    #[test]
    fn very_basic_expressions() {
        let tokens = [T::Number("1".into())];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(parse_expression(&mut tokens, 0), Ok(Num(1.0)));
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [T::IdentFrag("yo".into())];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(parse_expression(&mut tokens, 0), Ok(Id("yo".into())));
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn binary_operations() {
        let tokens = [
            T::Number("1".into()),
            T::Plus,
            T::Number("2".into()),
            T::Asterisk,
            T::Number("3".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Add,
                left: bx(Num(1.0)),
                right: bx(Bop {
                    operation: Mul,
                    left: bx(Num(2.0)),
                    right: bx(Num(3.0))
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::Number("1".into()),
            T::Times,
            T::Number("2".into()),
            T::Minus,
            T::Number("3".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Sub,
                left: bx(Bop {
                    operation: Cross,
                    left: bx(Num(1.0)),
                    right: bx(Num(2.0))
                }),
                right: bx(Num(3.0))
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [T::Number("1".into()), T::Plus];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected expression, found end of input".into())
        );
    }

    #[test]
    fn prefix_operations() {
        let tokens = [
            T::Plus,
            T::Minus,
            T::Plus,
            T::Minus,
            T::Plus,
            T::IdentFrag("j".into()),
            T::Asterisk,
            T::Number("3".into()),
            T::Exclamation,
            T::Exclamation,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Uop {
                operation: Neg,
                arg: bx(Uop {
                    operation: Neg,
                    arg: bx(Bop {
                        operation: Mul,
                        left: bx(Id("j".into())),
                        right: bx(Uop {
                            operation: Fac,
                            arg: bx(Uop {
                                operation: Fac,
                                arg: bx(Num(3.0))
                            })
                        })
                    })
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn paren() {
        let tokens = [
            T::Number("1".into()),
            T::Cdot,
            T::LParen,
            T::Number("2".into()),
            T::Minus,
            T::Number("3".into()),
            T::RParen,
            T::Exclamation,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Dot,
                left: bx(Num(1.0)),
                right: bx(Uop {
                    operation: Fac,
                    arg: bx(Bop {
                        operation: Sub,
                        left: bx(Num(2.0)),
                        right: bx(Num(3.0))
                    })
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn mismatched_paren() {
        let tokens = [
            T::Number("1".into()),
            T::Cdot,
            T::LParen,
            T::Number("2".into()),
            T::Minus,
            T::Number("3".into()),
            T::Exclamation,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected ')', found end of input".to_string())
        );
    }

    #[test]
    fn frac() {
        let tokens = [
            T::Number("1".into()),
            T::Cdot,
            T::Frac {
                num: &[Node::Char('2'), Node::Char('.'), Node::Char('3')],
                den: &[Node::Char('4')],
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Dot,
                left: bx(Num(1.0)),
                right: bx(Bop {
                    operation: Div,
                    left: bx(Num(2.3)),
                    right: bx(Num(4.0))
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn sqrt() {
        let tokens = [T::Sqrt {
            root: None,
            arg: &[Node::Char('4')],
        }];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Uop {
                operation: Sqrt,
                arg: bx(Num(4.0))
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [T::Sqrt {
            root: Some(&[Node::Char('3')]),
            arg: &[Node::Char('4')],
        }];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Pow,
                left: bx(Num(4.0)),
                right: bx(Bop {
                    operation: Div,
                    left: bx(Num(1.0)),
                    right: bx(Num(3.0))
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn sup() {
        let sup = [
            Node::Char('2'),
            Node::SubSup {
                sub: None,
                sup: Some(vec![Node::Char('3')]),
            },
        ];
        let tokens = [
            T::Number("1".into()),
            T::SubSup {
                sub: None,
                sup: Some(&sup),
            },
            T::SubSup {
                sub: None,
                sup: Some(&[Node::CtrlSeq("asdf")]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Pow,
                left: bx(Bop {
                    operation: Pow,
                    left: bx(Num(1.0)),
                    right: bx(Bop {
                        operation: Pow,
                        left: bx(Num(2.0)),
                        right: bx(Num(3.0))
                    })
                }),
                right: bx(Id("asdf".into()))
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn sub_sup() {
        let tokens = [
            T::IdentFrag("a".into()),
            T::SubSup {
                sub: Some(&[Node::Char('b'), Node::Char('c')]),
                sup: None,
            },
            T::Asterisk,
            T::IdentFrag("n".into()),
            T::SubSup {
                sub: Some(&[Node::Char('d'), Node::Char('6')]),
                sup: Some(&[Node::Char('e')]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Mul,
                left: bx(Id("a_{bc}".into())),
                right: bx(Bop {
                    operation: Pow,
                    left: bx(Id("n_{d6}".into())),
                    right: bx(Id("e".into()))
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn bad_sub() {
        let tokens = [
            T::IdentFrag("a".into()),
            T::SubSup {
                sub: Some(&[Node::Char('b'), Node::Char(' '), Node::Char('c')]),
                sup: Some(&[]),
            },
            T::Asterisk,
            T::IdentFrag("n".into()),
            T::SubSup {
                sub: Some(&[Node::Char('d'), Node::Char('6')]),
                sup: Some(&[Node::Char('e')]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("name subscript expected letters and digits, found ' '".into())
        );

        let tokens = [
            T::Number("4".into()),
            T::SubSup {
                sub: Some(&[Node::Char('b'), Node::Char('j'), Node::Char('c')]),
                sup: Some(&[]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("only identifiers may have subscripts".into())
        );
    }

    #[test]
    fn implicit_multiplication() {
        let tokens = [
            T::IdentFrag("a".into()),
            T::IdentFrag("b".to_string()),
            T::Plus,
            T::Number("3".into()),
            T::LParen,
            T::IdentFrag("c".into()),
            T::Comma,
            T::IdentFrag("pi".into()),
            T::RParen,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Add,
                left: bx(Bop {
                    operation: Mul,
                    left: bx(Id("a".into())),
                    right: bx(Id("b".into()))
                }),
                right: bx(Bop {
                    operation: Mul,
                    left: bx(Num(3.0)),
                    right: bx(Bop {
                        operation: Point,
                        left: bx(Id("c".into())),
                        right: bx(Id("pi".into()))
                    })
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn call_or_multiply() {
        let tokens = [
            T::IdentFrag("a".into()),
            T::LParen,
            T::Number("6".into()),
            T::Comma,
            T::IdentFrag("b".into()),
            T::RParen,
            T::Number("3".into()),
            T::IdentFrag("k".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Mul,
                left: bx(Bop {
                    operation: Mul,
                    left: bx(CallMull {
                        callee: "a".into(),
                        args: vec![Num(6.0), Id("b".into())]
                    }),
                    right: bx(Num(3.0))
                }),
                right: bx(Id("k".into()))
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn point() {
        let tokens = [
            T::LParen,
            T::Number("6".into()),
            T::Comma,
            T::IdentFrag("b".into()),
            T::RParen,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Point,
                left: bx(Num(6.0)),
                right: bx(Id("b".into()))
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn empty_paren() {
        let tokens = [T::LParen, T::RParen];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("parentheses cannot be empty".into())
        );
    }

    #[test]
    fn too_many_coordinates() {
        let tokens = [
            T::LParen,
            T::Number("6".into()),
            T::Comma,
            T::IdentFrag("b".into()),
            T::Comma,
            T::IdentFrag("b".into()),
            T::RParen,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("points may only have 2 coordinates".into())
        );
    }

    #[test]
    fn list() {
        let tokens = [
            T::LBracket,
            T::Number("6".into()),
            T::Comma,
            T::IdentFrag("b".into()),
            T::LParen,
            T::RParen,
            T::Comma,
            T::IdentFrag("b".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(List(vec![
                Num(6.0),
                CallMull {
                    callee: "b".into(),
                    args: vec![]
                },
                Id("b".into())
            ]))
        );

        let tokens = [T::LBracket, T::RBracket, T::Number("5".into())];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Mul,
                left: bx(List(vec![])),
                right: bx(Num(5.0))
            })
        );
    }

    #[test]
    fn index() {
        let tokens = [
            T::LBracket,
            T::Number("6".into()),
            T::Comma,
            T::IdentFrag("b".into()),
            T::LParen,
            T::RParen,
            T::Comma,
            T::IdentFrag("b".into()),
            T::RBracket,
            T::LBracket,
            T::Number("5".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Index,
                left: bx(List(vec![
                    Num(6.0),
                    CallMull {
                        callee: "b".into(),
                        args: vec![]
                    },
                    Id("b".into())
                ])),
                right: bx(Num(5.0))
            })
        );

        let tokens = [
            T::IdentFrag("L".into()),
            T::LBracket,
            T::Number("5".into()),
            T::Comma,
            T::Number("4".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Index,
                left: bx(Id("L".into())),
                right: bx(List(vec![Num(5.0), Num(4.0)]))
            })
        );

        let tokens = [T::IdentFrag("L".into()), T::LBracket, T::RBracket];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("square brackets cannot be empty".into())
        );
    }

    #[test]
    fn list_range() {
        let tokens = [
            T::LBracket,
            T::Number("1".into()),
            T::Comma,
            T::IdentFrag("a".into()),
            T::SubSup {
                sub: Some(&[Node::Char('2')]),
                sup: None,
            },
            T::Plus,
            T::Number("3".into()),
            T::Ellipsis,
            T::Number("4".into()),
            T::Comma,
            T::Number("5".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(ListRange {
                before_ellipsis: vec![
                    Num(1.0),
                    Bop {
                        operation: Add,
                        left: bx(Id("a_{2}".into())),
                        right: bx(Num(3.0))
                    }
                ],
                after_ellipsis: vec![Num(4.0), Num(5.0)]
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::LBracket,
            T::Number("1".into()),
            T::Comma,
            T::IdentFrag("a".into()),
            T::SubSup {
                sub: Some(&[Node::Char('2')]),
                sup: None,
            },
            T::Plus,
            T::Number("3".into()),
            T::Comma,
            T::Ellipsis,
            T::Number("4".into()),
            T::Comma,
            T::Number("5".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(ListRange {
                before_ellipsis: vec![
                    Num(1.0),
                    Bop {
                        operation: Add,
                        left: bx(Id("a_{2}".into())),
                        right: bx(Num(3.0))
                    }
                ],
                after_ellipsis: vec![Num(4.0), Num(5.0)]
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::IdentFrag("L".into()),
            T::LBracket,
            T::Number("1".into()),
            T::Comma,
            T::IdentFrag("a".into()),
            T::SubSup {
                sub: Some(&[Node::Char('2')]),
                sup: None,
            },
            T::Plus,
            T::Number("3".into()),
            T::Comma,
            T::Ellipsis,
            T::Comma,
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Index,
                left: bx(Id("L".into())),
                right: bx(ListRange {
                    before_ellipsis: vec![
                        Num(1.0),
                        Bop {
                            operation: Add,
                            left: bx(Id("a_{2}".into())),
                            right: bx(Num(3.0))
                        }
                    ],
                    after_ellipsis: vec![]
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::LBracket,
            T::Number("1".into()),
            T::Comma,
            T::IdentFrag("a".into()),
            T::SubSup {
                sub: Some(&[Node::Char('2')]),
                sup: None,
            },
            T::Plus,
            T::Number("3".into()),
            T::Comma,
            T::Ellipsis,
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected expression after '...', found ']'".into())
        );

        let tokens = [
            T::LBracket,
            T::Ellipsis,
            T::Number("1".into()),
            T::Comma,
            T::IdentFrag("a".into()),
            T::SubSup {
                sub: Some(&[Node::Char('2')]),
                sup: None,
            },
            T::Plus,
            T::Number("3".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected expression, found '...'".into())
        );
    }

    #[test]
    fn no_trailing_commas() {
        let tokens = [
            T::LBracket,
            T::Number("6".into()),
            T::Comma,
            T::IdentFrag("b".into()),
            T::LParen,
            T::RParen,
            T::Comma,
            T::IdentFrag("b".into()),
            T::Comma,
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected expression, found ']'".into())
        );

        let tokens = [
            T::LParen,
            T::Number("1".into()),
            T::Comma,
            T::Number("2".into()),
            T::Comma,
            T::RParen,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected expression, found ')'".into())
        );
    }

    #[test]
    fn abs() {
        let tokens = [
            T::Number("6".into()),
            T::LPipe,
            T::IdentFrag("b".into()),
            T::RPipe,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Mul,
                left: bx(Num(6.0)),
                right: bx(Uop {
                    operation: Norm,
                    arg: bx(Id("b".into()))
                })
            })
        );
    }

    #[test]
    fn chained_comparison() {
        let tokens = [
            T::Number("1.0".into()),
            T::LessEqual,
            T::Number("2.0".into()),
            T::GreaterEqual,
            T::Number("3.0".into()),
            T::Less,
            T::Number("5.0".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_chained_comparison(&mut tokens, None),
            Ok(ChainedComparison {
                operands: vec![Num(1.0), Num(2.0), Num(3.0), Num(5.0)],
                operators: vec![LessEqual, GreaterEqual, Less]
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [T::Equal, T::Number("2.0".into())];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_chained_comparison(&mut tokens, Some(Num(1.0))),
            Ok(ChainedComparison {
                operands: vec![Num(1.0), Num(2.0)],
                operators: vec![Equal]
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [T::Number("2.0".into())];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_chained_comparison(&mut tokens, None),
            Err("expected comparison, found end of input".into())
        );
    }

    #[test]
    fn list_filter() {
        let tokens = [
            T::IdentFrag("L".into()),
            T::LBracket,
            T::Number("1".into()),
            T::LessEqual,
            T::Number("2".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Index,
                left: bx(Id("L".into())),
                right: bx(Expression::ChainedComparison(ChainedComparison {
                    operands: vec![Num(1.0), Num(2.0)],
                    operators: vec![LessEqual]
                }))
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::LBracket,
            T::Number("1".into()),
            T::LessEqual,
            T::Number("2".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected ']', found '≤'".into())
        );
    }

    #[test]
    fn piecewise() {
        let tokens = [T::LBrace, T::RBrace];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(parse_expression(&mut tokens, 0), Ok(Num(1.0)));
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [T::LBrace, T::Number("5".into()), T::RBrace];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected comparison, found '}'".into())
        );

        let tokens = [
            T::LBrace,
            T::Number("1".into()),
            T::Less,
            T::Number("2".into()),
            T::RBrace,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Piecewise {
                test: bx(Expression::ChainedComparison(ChainedComparison {
                    operands: vec![Num(1.0), Num(2.0)],
                    operators: vec![Less]
                })),
                consequent: bx(Num(1.0)),
                alternate: None
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::LBrace,
            T::Number("1".into()),
            T::Less,
            T::Number("2".into()),
            T::Colon,
            T::RBrace,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err("expected expression, found '}'".into())
        );

        let tokens = [
            T::LBrace,
            T::Number("1".into()),
            T::Less,
            T::Number("2".into()),
            T::Colon,
            T::Number("3".into()),
            T::RBrace,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Piecewise {
                test: bx(Expression::ChainedComparison(ChainedComparison {
                    operands: vec![Num(1.0), Num(2.0)],
                    operators: vec![Less]
                })),
                consequent: bx(Num(3.0)),
                alternate: None
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::LBrace,
            T::Number("1".into()),
            T::Less,
            T::Number("2".into()),
            T::Comma,
            T::Number("3".into()),
            T::Equal,
            T::Number("4".into()),
            T::Equal,
            T::Number("5".into()),
            T::Colon,
            T::Number("6".into()),
            T::Comma,
            T::Number("7".into()),
            T::RBrace,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Piecewise {
                test: bx(Expression::ChainedComparison(ChainedComparison {
                    operands: vec![Num(1.0), Num(2.0)],
                    operators: vec![Less]
                })),
                consequent: bx(Num(1.0)),
                alternate: Some(bx(Piecewise {
                    test: bx(Expression::ChainedComparison(ChainedComparison {
                        operands: vec![Num(3.0), Num(4.0), Num(5.0)],
                        operators: vec![Equal, Equal]
                    })),
                    consequent: bx(Num(6.0)),
                    alternate: Some(bx(Num(7.0)))
                }))
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn point_coordinate_access() {
        let tokens = [
            T::IdentFrag("p".into()),
            T::Dot,
            T::IdentFrag("x".into()),
            T::Plus,
            T::IdentFrag("p".into()),
            T::Dot,
            T::IdentFrag("y".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Add,
                left: bx(Uop {
                    operation: PointX,
                    arg: bx(Id("p".into()))
                }),
                right: bx(Uop {
                    operation: PointY,
                    arg: bx(Id("p".into()))
                }),
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn dot_call() {
        let tokens = [
            T::IdentFrag("a".into()),
            T::Dot,
            T::IdentFrag("max".into()),
            T::Plus,
            T::IdentFrag("b".into()),
            T::Dot,
            T::IdentFrag("max".into()),
            T::LParen,
            T::RParen,
            T::Plus,
            T::IdentFrag("c".into()),
            T::Dot,
            T::IdentFrag("join".into()),
            T::LParen,
            T::Number("1".into()),
            T::Comma,
            T::Number("2".into()),
            T::RParen,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Add,
                left: bx(Bop {
                    operation: Add,
                    left: bx(Call {
                        callee: "max".into(),
                        args: vec![Id("a".into())]
                    }),
                    right: bx(Call {
                        callee: "max".into(),
                        args: vec![Id("b".into())]
                    })
                }),
                right: bx(Call {
                    callee: "join".into(),
                    args: vec![Id("c".into()), Num(1.0), Num(2.0)]
                })
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn sum_prod() {
        let tokens = [T::Sum];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"'\sum' expected lower and upper bounds, found end of input".into())
        );

        let tokens = [
            T::Sum,
            T::SubSup {
                sub: Some(&[]),
                sup: None,
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"expected identifier, found end of group".into())
        );

        let tokens = [
            T::Prod,
            T::SubSup {
                sub: None,
                sup: Some(&[]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"'\prod' expected lower bound".into())
        );

        let tokens = [
            T::Sum,
            T::SubSup {
                sub: Some(&[Node::Char('5')]),
                sup: Some(&[]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"expected identifier, found '5'".into())
        );

        let tokens = [
            T::Prod,
            T::SubSup {
                sub: Some(&[Node::Char('n')]),
                sup: Some(&[]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"expected '=', found end of group".into())
        );

        let tokens = [
            T::Sum,
            T::SubSup {
                sub: Some(&[Node::Char('n'), Node::Char('=')]),
                sup: Some(&[]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"expected expression, found end of group".into())
        );

        let tokens = [
            T::Prod,
            T::SubSup {
                sub: Some(&[Node::Char('n'), Node::Char('='), Node::Char('1')]),
                sup: Some(&[]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"expected expression, found end of group".into())
        );

        let tokens = [
            T::Sum,
            T::SubSup {
                sub: Some(&[
                    Node::Char('n'),
                    Node::Char('='),
                    Node::Char('1'),
                    Node::Char('='),
                ]),
                sup: Some(&[]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"expected end of group, found '='".into())
        );

        let tokens = [
            T::Prod,
            T::SubSup {
                sub: Some(&[Node::Char('n'), Node::Char('='), Node::Char('1')]),
                sup: Some(&[Node::Char('9')]),
            },
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Err(r"expected expression, found end of input".into())
        );

        let tokens = [
            T::Prod,
            T::SubSup {
                sub: Some(&[Node::Char('n'), Node::Char('='), Node::Char('1')]),
                sup: Some(&[Node::Char('9')]),
            },
            T::Number("5".into()),
            T::IdentFrag("n".into()),
            T::Plus,
            T::Number("6".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(Bop {
                operation: Add,
                left: bx(SumProd {
                    kind: Prod,
                    variable: "n".into(),
                    lower_bound: bx(Num(1.0)),
                    upper_bound: bx(Num(9.0)),
                    body: bx(Bop {
                        operation: Mul,
                        left: bx(Num(5.0)),
                        right: bx(Id("n".into()))
                    })
                }),
                right: bx(Num(6.0))
            })
        );
    }

    #[test]
    fn with_for() {
        let tokens = [
            T::IdentFrag("a".into()),
            T::With,
            T::IdentFrag("a".into()),
            T::Equal,
            T::Number("4".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(With {
                body: bx(Id("a".into())),
                substitutions: vec![("a".into(), Num(4.0))],
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::IdentFrag("a".into()),
            T::Plus,
            T::Number("7".into()),
            T::With,
            T::IdentFrag("a".into()),
            T::Equal,
            T::Number("4".into()),
            T::Plus,
            T::LBracket,
            T::IdentFrag("c".into()),
            T::For,
            T::IdentFrag("c".into()),
            T::SubSup {
                sub: Some(&[Node::Char('4')]),
                sup: None,
            },
            T::Equal,
            T::Number("2".into()),
            T::Comma,
            T::IdentFrag("d".into()),
            T::Equal,
            T::Number("6".into()),
            T::RBracket,
            T::Comma,
            T::IdentFrag("b".into()),
            T::Equal,
            T::Number("5".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(With {
                body: bx(Bop {
                    operation: Add,
                    left: bx(Id("a".into())),
                    right: bx(Num(7.0))
                }),
                substitutions: vec![
                    (
                        "a".into(),
                        Bop {
                            operation: Add,
                            left: bx(Num(4.0)),
                            right: bx(For {
                                body: bx(Id("c".into())),
                                lists: vec![("c_4".into(), Num(2.0)), ("d".into(), Num(6.0))],
                            })
                        }
                    ),
                    ("b".into(), Num(5.0))
                ],
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::IdentFrag("a".into()),
            T::With,
            T::IdentFrag("a".into()),
            T::Equal,
            T::Number("3".into()),
            T::For,
            T::IdentFrag("b".into()),
            T::Equal,
            T::LBracket,
            T::Number("5".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(For {
                body: bx(With {
                    body: bx(Id("a".into())),
                    substitutions: vec![("a".into(), Num(3.0))],
                }),
                lists: vec![("b".into(), List(vec![Num(5.0)]))],
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);

        let tokens = [
            T::IdentFrag("a".into()),
            T::Plus,
            T::IdentFrag("b".into()),
            T::For,
            T::IdentFrag("b".into()),
            T::Equal,
            T::LBracket,
            T::Number("5".into()),
            T::RBracket,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression(&mut tokens, 0),
            Ok(For {
                body: bx(Bop {
                    operation: Add,
                    left: bx(Id("a".into())),
                    right: bx(Id("b".into()))
                }),
                lists: vec![("b".into(), List(vec![Num(5.0)]))],
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn ele_assignment() {
        let tokens = [
            T::IdentFrag("c".into()),
            T::Equal,
            T::LParen,
            T::Number("1.18".into()),
            T::Comma,
            T::Number("3.78".into()),
            T::RParen,
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression_list_entry_from_tokens(&mut tokens),
            Ok(Ele::Assignment {
                name: "c".into(),
                value: Bop {
                    operation: Point,
                    left: bx(Num(1.18)),
                    right: bx(Num(3.78))
                }
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn ele_function_declaration() {
        let tokens = [
            T::IdentFrag("f".into()),
            T::SubSup {
                sub: Some(&[Node::Char('4')]),
                sup: None,
            },
            T::LParen,
            T::IdentFrag("x".into()),
            T::Comma,
            T::IdentFrag("y".into()),
            T::RParen,
            T::Equal,
            T::IdentFrag("x".into()),
            T::SubSup {
                sub: None,
                sup: Some(&[Node::Char('2')]),
            },
            T::IdentFrag("y".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression_list_entry_from_tokens(&mut tokens),
            Ok(Ele::FunctionDeclaration {
                name: "f_{4}".into(),
                parameters: vec!["x".into(), "y".into()],
                body: Bop {
                    operation: Mul,
                    left: bx(Bop {
                        operation: Pow,
                        left: bx(Id("x".into())),
                        right: bx(Num(2.0))
                    }),
                    right: bx(Id("y".into()))
                }
            })
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn ele_relation() {
        let tokens = [
            T::IdentFrag("f".into()),
            T::SubSup {
                sub: Some(&[Node::Char('4')]),
                sup: None,
            },
            T::LParen,
            T::IdentFrag("x".into()),
            T::Comma,
            T::IdentFrag("y".into()),
            T::RParen,
            T::Less,
            T::IdentFrag("x".into()),
            T::SubSup {
                sub: None,
                sup: Some(&[Node::Char('2')]),
            },
            T::IdentFrag("y".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression_list_entry_from_tokens(&mut tokens),
            Ok(Ele::Relation(ChainedComparison {
                operands: vec![
                    CallMull {
                        callee: "f_{4}".into(),
                        args: vec![Id("x".into()), Id("y".into())],
                    },
                    Bop {
                        operation: Mul,
                        left: bx(Bop {
                            operation: Pow,
                            left: bx(Id("x".into())),
                            right: bx(Num(2.0)),
                        }),
                        right: bx(Id("y".into())),
                    },
                ],
                operators: vec![Less],
            })),
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }

    #[test]
    fn ele_expression() {
        let tokens = [
            T::IdentFrag("f".into()),
            T::SubSup {
                sub: Some(&[Node::Char('4')]),
                sup: None,
            },
            T::LParen,
            T::IdentFrag("x".into()),
            T::Comma,
            T::IdentFrag("y".into()),
            T::RParen,
            T::Plus,
            T::IdentFrag("x".into()),
            T::SubSup {
                sub: None,
                sup: Some(&[Node::Char('2')]),
            },
            T::IdentFrag("y".into()),
        ];
        let mut tokens = Tokens::new(&tokens, T::EndOfInput);
        assert_eq!(
            parse_expression_list_entry_from_tokens(&mut tokens),
            Ok(Ele::Expression(Bop {
                operation: Add,
                left: bx(CallMull {
                    callee: "f_{4}".into(),
                    args: vec![Id("x".into()), Id("y".into())],
                }),
                right: bx(Bop {
                    operation: Mul,
                    left: bx(Bop {
                        operation: Pow,
                        left: bx(Id("x".into())),
                        right: bx(Num(2.0)),
                    }),
                    right: bx(Id("y".into())),
                }),
            })),
        );
        assert_eq!(tokens.next(), &T::EndOfInput);
    }
}
