use std::collections::{HashMap, HashSet};

use crate::ast::{self, ExpressionListEntry};
pub use crate::ast::{BinaryOperator, ComparisonOperator, SumProdKind, UnaryOperator};

#[derive(Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    Identifier(usize),
    List(Vec<Expression>),
    ListRange {
        before_ellipsis: Vec<Expression>,
        after_ellipsis: Vec<Expression>,
    },
    UnaryOperation {
        operation: UnaryOperator,
        arg: Box<Expression>,
    },
    BinaryOperation {
        operation: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    ChainedComparison {
        operands: Vec<Expression>,
        operators: Vec<ComparisonOperator>,
    },
    Piecewise {
        test: Box<Expression>,
        consequent: Box<Expression>,
        alternate: Option<Box<Expression>>,
    },
    SumProd {
        kind: SumProdKind,
        variable: usize,
        lower_bound: Box<Expression>,
        upper_bound: Box<Expression>,
        body: Body,
    },
    For {
        body: Body,
        lists: Vec<(usize, Expression)>,
    },
}

#[derive(Debug, PartialEq)]
pub struct Assignment {
    pub name: usize,
    pub value: Expression,
}

#[derive(Debug, PartialEq)]
pub struct Body {
    pub assignments: Vec<Assignment>,
    pub value: Box<Expression>,
}

#[derive(Clone)]
struct Scoped<T> {
    value: T,
    depends: usize,
}

fn scoped<T>(value: T, depends: usize) -> Scoped<T> {
    Scoped { value, depends }
}

#[derive(Default)]
struct Scope<'a> {
    definitions: HashMap<&'a str, usize>,
    derived: HashMap<&'a str, Scoped<usize>>,
}

struct Resolver<'a> {
    globals: HashMap<&'a str, Result<&'a ExpressionListEntry, String>>,
    assignments: Vec<Assignment>,
    next_name: usize,
    scopes: Vec<Scope<'a>>,
    currently_resolving: HashSet<&'a str>,
}

impl<'a> Resolver<'a> {
    fn new(list: &'a [ExpressionListEntry]) -> Self {
        let mut globals = HashMap::new();

        for entry in list {
            match entry {
                ExpressionListEntry::Assignment { name, .. }
                | ExpressionListEntry::FunctionDeclaration { name, .. } => {
                    if let Some(result @ Ok(_)) = globals.get_mut(name.as_str()) {
                        *result = Err(format!("'{name}' defined multiple times"));
                    } else {
                        globals.insert(name.as_str(), Ok(entry));
                    }
                }
                _ => continue,
            };
        }

        Self {
            globals,
            assignments: vec![],
            next_name: 0,
            scopes: vec![Scope::default()],
            currently_resolving: HashSet::new(),
        }
    }

    fn next_name(&mut self) -> usize {
        let next_name = self.next_name;
        self.next_name += 1;
        next_name
    }

    fn calculate_variable(&mut self, name: &'a str) -> Result<Scoped<usize>, String> {
        for (i, scope) in self.scopes.iter().enumerate().rev() {
            if let Some(&index) = scope.definitions.get(name) {
                return Ok(Scoped {
                    value: index,
                    depends: i,
                });
            }
        }

        if let Some(x) = self.scopes.last().unwrap().derived.get(name) {
            return Ok(x.clone());
        }

        let entry = self
            .globals
            .get(name)
            .ok_or_else(|| format!("'{name}' is not defined"))?
            .clone()?;

        match entry {
            ExpressionListEntry::Assignment { value, .. } => {
                if self.currently_resolving.contains(name) {
                    let mut names = self.currently_resolving.iter().cloned().collect::<Vec<_>>();
                    names.sort();
                    let last = names.pop().unwrap();

                    return Err(if names.len() > 0 {
                        let joined = names.join("', '");
                        format!("'{joined}' and '{last}' can't be defined in terms of each other")
                    } else {
                        format!("'{last}' can't be defined in terms of itself")
                    });
                }
                self.currently_resolving.insert(name);
                let value = self.resolve_expression(value);
                self.currently_resolving.remove(name);
                let value = value?;
                let result = if let Some(result) = self.scopes[value.depends].derived.get(name) {
                    result.clone()
                } else {
                    let index = self.next_name();
                    self.assignments.push(Assignment {
                        name: index,
                        value: value.value,
                    });
                    scoped(index, value.depends)
                };

                for i in result.depends..self.scopes.len() {
                    self.scopes[i].derived.insert(name, result.clone());
                }

                Ok(result)
            }
            ExpressionListEntry::FunctionDeclaration { .. } => {
                Err(format!("'{name}' is a function, try using parentheses"))
            }
            _ => unreachable!(),
        }
    }

    fn resolve_expressions(
        &mut self,
        es: &'a [ast::Expression],
    ) -> Result<Scoped<Vec<Expression>>, String> {
        let mut result = vec![];
        let mut depends = 0;

        for e in es {
            let e = self.resolve_expression(e)?;
            result.push(e.value);
            depends = depends.max(e.depends);
        }

        Ok(scoped(result, depends))
    }

    fn resolve_call(
        &mut self,
        callee: &'a str,
        args: &'a [ast::Expression],
    ) -> Result<Scoped<Expression>, String> {
        let Some(entry) = self.globals.get(callee) else {
            return Err(format!("'{callee}' is not defined"));
        };
        let ExpressionListEntry::FunctionDeclaration {
            parameters, body, ..
        } = entry.clone()?
        else {
            return Err(format!("variable '{callee}' can't be used as a function"));
        };

        if parameters.len() != args.len() {
            return Err(format!(
                "function '{callee}' requires {}{}",
                if args.len() > parameters.len() {
                    "only "
                } else {
                    ""
                },
                if parameters.len() == 1 {
                    "1 argument".into()
                } else {
                    format!("{} arguments", parameters.len())
                }
            ));
        }

        let mut scope = Scope::default();
        let mut depends = 0;

        for (name, value) in parameters.iter().zip(args) {
            if scope.definitions.contains_key(name.as_str()) {
                return Err(format!(
                    "cannot use '{name}' for multiple parameters of this function"
                ));
            }

            let value = self.resolve_expression(value)?;
            depends = depends.max(value.depends);
            let index = self.next_name();
            self.assignments.push(Assignment {
                name: index,
                value: value.value,
            });
            scope.definitions.insert(name, index);
        }

        // i think depends might need to be a (bit)set or something
        self.scopes.push(scope);
        let body = self.resolve_expression(body);
        self.scopes.pop();
        let body = body?;
        depends = depends.max(body.depends.min(self.scopes.len() - 1));

        Ok(scoped(body.value, depends))
    }

    fn resolve_expression(&mut self, e: &'a ast::Expression) -> Result<Scoped<Expression>, String> {
        match e {
            ast::Expression::Number(value) => Ok(scoped(Expression::Number(*value), 0)),
            ast::Expression::Identifier(name) => {
                let name = self.calculate_variable(name)?;
                Ok(scoped(Expression::Identifier(name.value), name.depends))
            }
            ast::Expression::List(list) => {
                let list = self.resolve_expressions(list)?;
                Ok(scoped(Expression::List(list.value), list.depends))
            }
            ast::Expression::ListRange {
                before_ellipsis,
                after_ellipsis,
            } => {
                let before_ellipsis = self.resolve_expressions(before_ellipsis)?;
                let after_ellipsis = self.resolve_expressions(after_ellipsis)?;
                Ok(scoped(
                    Expression::ListRange {
                        before_ellipsis: before_ellipsis.value,
                        after_ellipsis: after_ellipsis.value,
                    },
                    before_ellipsis.depends.max(after_ellipsis.depends),
                ))
            }
            ast::Expression::UnaryOperation { operation, arg } => {
                let arg = self.resolve_expression(&arg)?;
                Ok(scoped(
                    Expression::UnaryOperation {
                        operation: *operation,
                        arg: Box::new(arg.value),
                    },
                    arg.depends,
                ))
            }
            ast::Expression::BinaryOperation {
                operation,
                left,
                right,
            } => {
                let left = self.resolve_expression(&left)?;
                let right = self.resolve_expression(&right)?;
                Ok(scoped(
                    Expression::BinaryOperation {
                        operation: *operation,
                        left: Box::new(left.value),
                        right: Box::new(right.value),
                    },
                    left.depends.max(right.depends),
                ))
            }
            ast::Expression::CallOrMultiply { callee, args } => {
                if let Some(Ok(ExpressionListEntry::FunctionDeclaration { .. })) =
                    self.globals.get(callee.as_str())
                {
                    self.resolve_call(callee, args)
                } else {
                    let name = callee;
                    let callee = self.calculate_variable(callee)?;
                    let args = self.resolve_expressions(args)?;
                    let len = args.value.len();

                    if len == 1 || len == 2 {
                        let mut args_iter = args.value.into_iter();
                        Ok(scoped(
                            Expression::BinaryOperation {
                                operation: BinaryOperator::Mul,
                                left: Box::new(Expression::Identifier(callee.value)),
                                right: Box::new(if len == 1 {
                                    args_iter.next().unwrap()
                                } else {
                                    Expression::BinaryOperation {
                                        operation: BinaryOperator::Point,
                                        left: Box::new(args_iter.next().unwrap()),
                                        right: Box::new(args_iter.next().unwrap()),
                                    }
                                }),
                            },
                            callee.depends.max(args.depends),
                        ))
                    } else if len == 0 {
                        Err(format!("variable '{name}' can't be used as a function"))
                    } else {
                        Err("points may only have 2 coordinates".into())
                    }
                }
            }
            ast::Expression::Call { callee, args } => self.resolve_call(callee, args),
            ast::Expression::ChainedComparison(ast::ChainedComparison {
                operands,
                operators,
            }) => {
                let operands = self.resolve_expressions(operands)?;
                Ok(scoped(
                    Expression::ChainedComparison {
                        operands: operands.value,
                        operators: operators.clone(),
                    },
                    operands.depends,
                ))
            }
            ast::Expression::Piecewise {
                test,
                consequent,
                alternate,
            } => {
                let test = self.resolve_expression(&test)?;
                let consequent = self.resolve_expression(&consequent)?;
                let mut depends = test.depends.max(consequent.depends);
                Ok(scoped(
                    Expression::Piecewise {
                        test: Box::new(test.value),
                        consequent: Box::new(consequent.value),
                        alternate: if let Some(e) = alternate {
                            let alternate = self.resolve_expression(&e)?;
                            depends = depends.max(alternate.depends);
                            Some(Box::new(alternate.value))
                        } else {
                            None
                        },
                    },
                    depends,
                ))
            }
            ast::Expression::SumProd { .. } => todo!(),
            ast::Expression::With {
                body,
                substitutions,
            } => {
                let mut scope = Scope::default();
                let mut depends = 0;

                for (name, value) in substitutions {
                    if scope.definitions.contains_key(name.as_str()) {
                        return Err(format!(
                            "a 'with' expression cannot make multiple substitutions for '{name}'"
                        ));
                    }

                    let value = self.resolve_expression(value)?;
                    depends = depends.max(value.depends);
                    let index = self.next_name();
                    self.assignments.push(Assignment {
                        name: index,
                        value: value.value,
                    });
                    scope.definitions.insert(name, index);
                }

                // i think depends might need to be a (bit)set or something
                self.scopes.push(scope);
                let body = self.resolve_expression(body);
                self.scopes.pop();
                let body = body?;
                depends = depends.max(body.depends.min(self.scopes.len() - 1));

                Ok(scoped(body.value, depends))
            }
            ast::Expression::For { .. } => todo!(),
        }
    }
}

pub fn resolve_names(
    list: &[ExpressionListEntry],
) -> (Vec<Assignment>, Vec<Option<Result<usize, String>>>) {
    let mut resolver = Resolver::new(list);

    let assignment_indices: Vec<_> = list
        .iter()
        .map(|e| match e {
            ExpressionListEntry::Assignment { name, value } => {
                if let Some(index) = resolver.scopes[0].derived.get(name.as_str()) {
                    Some(Ok(index.value))
                } else {
                    match resolver.resolve_expression(value) {
                        Ok(value) => {
                            let index = resolver.next_name();
                            resolver.assignments.push(Assignment {
                                name: index,
                                value: value.value,
                            });

                            if resolver.globals.get(name.as_str()).unwrap().is_ok() {
                                resolver.scopes[0].derived.insert(name, scoped(index, 0));
                            }

                            Some(Ok(index))
                        }
                        Err(error) => Some(Err(error)),
                    }
                }
            }
            ExpressionListEntry::FunctionDeclaration { .. } => None,
            ExpressionListEntry::Relation(..) => todo!(),
            ExpressionListEntry::Expression(expression) => {
                Some(match resolver.resolve_expression(expression) {
                    Ok(value) => {
                        let index = resolver.next_name();
                        resolver.assignments.push(Assignment {
                            name: index,
                            value: value.value,
                        });
                        Ok(index)
                    }
                    Err(error) => Err(error),
                })
            }
        })
        .collect();

    (resolver.assignments, assignment_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::{
        BinaryOperator as ABo,
        Expression::{
            // ListRange as AListRange,
            // UnaryOperation as AUop,
            BinaryOperation as ABop,
            CallOrMultiply as ACallMul,
            // Call as ACall,
            Identifier as AId,
            List as AList,
            // For as AFor,
            Number as ANum,
            // ChainedComparison as AComparison,
            // Piecewise as APiecewise,
            // SumProd as ASumProd,
            With as AWith,
        },
    };
    use pretty_assertions::assert_eq;
    use ExpressionListEntry::{
        Assignment as ElAssign, Expression as ElExpr, FunctionDeclaration as ElFunction,
    };

    fn bx<T>(x: T) -> Box<T> {
        Box::new(x)
    }

    #[test]
    fn expressions() {
        assert_eq!(
            resolve_names(&[
                ElExpr(ANum(5.0)),
                ElExpr(ABop {
                    operation: ABo::Add,
                    left: bx(ANum(1.0)),
                    right: bx(ANum(2.0)),
                }),
            ]),
            (
                vec![
                    Assignment {
                        name: 0,
                        value: Expression::Number(5.0),
                    },
                    Assignment {
                        name: 1,
                        value: Expression::BinaryOperation {
                            operation: BinaryOperator::Add,
                            left: bx(Expression::Number(1.0)),
                            right: bx(Expression::Number(2.0)),
                        },
                    },
                ],
                vec![Some(Ok(0)), Some(Ok(1))],
            ),
        );
    }

    #[test]
    fn assignments() {
        assert_eq!(
            resolve_names(&[
                ElAssign {
                    name: "c".into(),
                    value: ANum(1.0),
                },
                ElAssign {
                    name: "b".into(),
                    value: AId("c".into()),
                },
            ]),
            (
                vec![
                    Assignment {
                        name: 0,
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        name: 1,
                        value: Expression::Identifier(0),
                    },
                ],
                vec![Some(Ok(0)), Some(Ok(1))],
            ),
        );
    }

    #[test]
    fn multiple_definitions_error() {
        assert_eq!(
            resolve_names(&[
                ElAssign {
                    name: "a".into(),
                    value: ANum(1.0),
                },
                ElAssign {
                    name: "a".into(),
                    value: ANum(2.0),
                },
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
            ]),
            (
                vec![
                    Assignment {
                        name: 0,
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        name: 1,
                        value: Expression::Number(2.0),
                    },
                ],
                vec![
                    Some(Ok(0)),
                    Some(Ok(1)),
                    Some(Err("'a' defined multiple times".into())),
                ],
            ),
        );
    }

    #[test]
    fn circular_error() {
        assert_eq!(
            resolve_names(&[
                ElAssign {
                    name: "a".into(),
                    value: AId("b".into()),
                },
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
            ]),
            (
                vec![],
                vec![
                    Some(Err(
                        "'a' and 'b' can't be defined in terms of each other".into()
                    )),
                    Some(Err(
                        "'a' and 'b' can't be defined in terms of each other".into()
                    )),
                ],
            ),
        );
    }

    #[test]
    fn dependencies() {
        assert_eq!(
            resolve_names(&[
                ElAssign {
                    name: "c".into(),
                    value: ANum(1.0),
                },
                ElAssign {
                    name: "b".into(),
                    value: AId("c".into()),
                },
                ElAssign {
                    name: "a".into(),
                    value: AWith {
                        body: bx(AId("b".into())),
                        substitutions: vec![("c".into(), ANum(2.0))],
                    },
                },
                ElExpr(AWith {
                    body: bx(AId("a".into())),
                    substitutions: vec![("b".into(), ANum(3.0))],
                }),
                ElExpr(AWith {
                    body: bx(AId("b".into())),
                    substitutions: vec![("c".into(), ANum(4.0))],
                }),
                ElExpr(AWith {
                    body: bx(AId("a".into())),
                    substitutions: vec![("c".into(), ANum(5.0))],
                }),
            ]),
            (
                vec![
                    // c = 1
                    Assignment {
                        name: 0,
                        value: Expression::Number(1.0),
                    },
                    // b = c
                    Assignment {
                        name: 1,
                        value: Expression::Identifier(0),
                    },
                    // with c = 2
                    Assignment {
                        name: 2,
                        value: Expression::Number(2.0),
                    },
                    // b = c
                    Assignment {
                        name: 3,
                        value: Expression::Identifier(2),
                    },
                    // a = b
                    Assignment {
                        name: 4,
                        value: Expression::Identifier(3),
                    },
                    // with b = 3
                    Assignment {
                        name: 5,
                        value: Expression::Number(3.0),
                    },
                    // with c = 2
                    Assignment {
                        name: 6,
                        value: Expression::Number(2.0),
                    },
                    // a = b
                    Assignment {
                        name: 7,
                        value: Expression::Identifier(5),
                    },
                    // a
                    Assignment {
                        name: 8,
                        value: Expression::Identifier(7),
                    },
                    // with c = 4
                    Assignment {
                        name: 9,
                        value: Expression::Number(4.0),
                    },
                    // b = c
                    Assignment {
                        name: 10,
                        value: Expression::Identifier(9),
                    },
                    // b
                    Assignment {
                        name: 11,
                        value: Expression::Identifier(10),
                    },
                    // with c = 5
                    Assignment {
                        name: 12,
                        value: Expression::Number(5.0),
                    },
                    // with c = 2
                    Assignment {
                        name: 13,
                        value: Expression::Number(2.0),
                    },
                    // b = c
                    Assignment {
                        name: 14,
                        value: Expression::Identifier(13),
                    },
                    // a = b
                    Assignment {
                        name: 15,
                        value: Expression::Identifier(14),
                    },
                    // a
                    Assignment {
                        name: 16,
                        value: Expression::Identifier(15),
                    },
                ],
                vec![
                    Some(Ok(0)),
                    Some(Ok(1)),
                    Some(Ok(4)),
                    Some(Ok(8)),
                    Some(Ok(11)),
                    Some(Ok(16)),
                ],
            ),
        );
    }

    #[test]
    fn function_errors() {
        assert_eq!(
            resolve_names(&[
                ElExpr(ACallMul {
                    callee: "a".into(),
                    args: vec![],
                }),
                ElAssign {
                    name: "b".into(),
                    value: ANum(1.0),
                },
                ElExpr(ACallMul {
                    callee: "b".into(),
                    args: vec![],
                }),
                ElFunction {
                    name: "c".into(),
                    parameters: vec![],
                    body: ANum(2.0),
                },
                ElExpr(AId("c".into())),
            ]),
            (
                vec![Assignment {
                    name: 0,
                    value: Expression::Number(1.0),
                }],
                vec![
                    Some(Err("'a' is not defined".into())),
                    Some(Ok(0)),
                    Some(Err("variable 'b' can't be used as a function".into())),
                    None,
                    Some(Err("'c' is a function, try using parentheses".into())),
                ],
            ),
        );
    }

    #[test]
    fn call_mul_disambiguation() {
        assert_eq!(
            resolve_names(&[
                ElAssign {
                    name: "a".into(),
                    value: ANum(1.0),
                },
                ElExpr(ACallMul {
                    callee: "a".into(),
                    args: vec![ANum(2.0)],
                }),
                ElExpr(ACallMul {
                    callee: "a".into(),
                    args: vec![ANum(3.0), ANum(4.0)],
                }),
                ElExpr(ACallMul {
                    callee: "a".into(),
                    args: vec![ANum(5.0), ANum(6.0), ANum(7.0), ANum(8.0)],
                }),
            ]),
            (
                vec![
                    Assignment {
                        name: 0,
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        name: 1,
                        value: Expression::BinaryOperation {
                            operation: BinaryOperator::Mul,
                            left: bx(Expression::Identifier(0)),
                            right: bx(Expression::Number(2.0)),
                        },
                    },
                    Assignment {
                        name: 2,
                        value: Expression::BinaryOperation {
                            operation: BinaryOperator::Mul,
                            left: bx(Expression::Identifier(0)),
                            right: bx(Expression::BinaryOperation {
                                operation: BinaryOperator::Point,
                                left: bx(Expression::Number(3.0)),
                                right: bx(Expression::Number(4.0)),
                            }),
                        },
                    },
                ],
                vec![
                    Some(Ok(0)),
                    Some(Ok(1)),
                    Some(Ok(2)),
                    Some(Err("points may only have 2 coordinates".into())),
                ],
            ),
        );
    }

    #[test]
    fn function_v1_9() {
        assert_eq!(
            resolve_names(&[
                // f(a1, a2, a3, a4) = [a1, b, c, d]
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()],
                    body: AList(vec![
                        AId("a1".into()),
                        AId("b".into()),
                        AId("c".into()),
                        AId("d".into()),
                    ]),
                },
                // b = a2
                ElAssign {
                    name: "b".into(),
                    value: AId("a2".into()),
                },
                // c = a3
                ElAssign {
                    name: "c".into(),
                    value: AId("a3".into()),
                },
                // a3 = 5
                ElAssign {
                    name: "a3".into(),
                    value: ANum(5.0),
                },
                // d = a4
                ElAssign {
                    name: "d".into(),
                    value: AId("a4".into()),
                },
                // f(1, 2, 3, 4) with a1 = 6, a2 = 7
                ElExpr(AWith {
                    body: bx(ACallMul {
                        callee: "f".into(),
                        args: vec![ANum(1.0), ANum(2.0), ANum(3.0), ANum(4.0)],
                    }),
                    substitutions: vec![("a1".into(), ANum(6.0)), ("a2".into(), ANum(7.0))],
                }),
            ]),
            (
                vec![
                    // a3 = 5
                    Assignment {
                        name: 0,
                        value: Expression::Number(5.0),
                    },
                    // c = a3
                    Assignment {
                        name: 1,
                        value: Expression::Identifier(0),
                    },
                    // with a1 = 6
                    Assignment {
                        name: 2,
                        value: Expression::Number(6.0),
                    },
                    // with a2 = 7
                    Assignment {
                        name: 3,
                        value: Expression::Number(7.0),
                    },
                    // a1 = 1
                    Assignment {
                        name: 4,
                        value: Expression::Number(1.0),
                    },
                    // a2 = 2
                    Assignment {
                        name: 5,
                        value: Expression::Number(2.0),
                    },
                    // a3 = 3
                    Assignment {
                        name: 6,
                        value: Expression::Number(3.0),
                    },
                    // a4 = 4
                    Assignment {
                        name: 7,
                        value: Expression::Number(4.0),
                    },
                    // b = a2
                    Assignment {
                        name: 8,
                        value: Expression::Identifier(5),
                    },
                    // c = a3
                    Assignment {
                        name: 9,
                        value: Expression::Identifier(6),
                    },
                    // d = a4
                    Assignment {
                        name: 10,
                        value: Expression::Identifier(7),
                    },
                    // [a1, b, c, d]
                    Assignment {
                        name: 11,
                        value: Expression::List(vec![
                            Expression::Identifier(4),
                            Expression::Identifier(8),
                            Expression::Identifier(9),
                            Expression::Identifier(10),
                        ]),
                    },
                ],
                vec![
                    None,
                    Some(Err("'a2' is not defined".into())),
                    Some(Ok(1)),
                    Some(Ok(0)),
                    Some(Err("'a4' is not defined".into())),
                    Some(Ok(11)),
                ],
            )
        );
    }

    // #[test]
    // fn function_v1_10() {
    //     assert_eq!(
    //         resolve_names(&[
    //             ElFunction {
    //                 name: "f".into(),
    //                 parameters: vec!["a1".into(), "a2".into(), "a3".into(), "a4".into()],
    //                 body: AList(vec![
    //                     AId("a1".into()),
    //                     AId("b".into()),
    //                     AId("c".into()),
    //                     AId("d".into()),
    //                 ]),
    //             },
    //             ElAssign {
    //                 name: "b".into(),
    //                 value: AId("a2".into()),
    //             },
    //             ElAssign {
    //                 name: "c".into(),
    //                 value: AId("a3".into()),
    //             },
    //             ElAssign {
    //                 name: "a3".into(),
    //                 value: ANum(5.0),
    //             },
    //             ElAssign {
    //                 name: "d".into(),
    //                 value: AId("a4".into()),
    //             },
    //             ElExpr(AWith {
    //                 body: bx(ACall {
    //                     callee: "f".into(),
    //                     args: vec![ANum(1.0), ANum(2.0), ANum(3.0), ANum(4.0)],
    //                 }),
    //                 substitutions: vec![("a1".into(), ANum(6.0)), ("a2".into(), ANum(7.0))],
    //             }),
    //         ]),
    //         (
    //             vec![
    //                 // a3 = 5
    //                 Assignment {
    //                     name: 0,
    //                     value: Expression::Number(5.0),
    //                 },
    //                 // c = a3
    //                 Assignment {
    //                     name: 1,
    //                     value: Expression::Identifier(0),
    //                 },
    //                 // with a1 = 6
    //                 Assignment {
    //                     name: 2,
    //                     value: Expression::Number(6.0),
    //                 },
    //                 // with a2 = 7
    //                 Assignment {
    //                     name: 3,
    //                     value: Expression::Number(7.0),
    //                 },
    //                 // a1 = 1
    //                 Assignment {
    //                     name: 4,
    //                     value: Expression::Number(1.0),
    //                 },
    //                 // a2 = 2
    //                 Assignment {
    //                     name: 5,
    //                     value: Expression::Number(2.0),
    //                 },
    //                 // a3 = 3
    //                 Assignment {
    //                     name: 6,
    //                     value: Expression::Number(3.0),
    //                 },
    //                 // a4 = 4
    //                 Assignment {
    //                     name: 7,
    //                     value: Expression::Number(4.0),
    //                 },
    //                 // b = a2
    //                 Assignment {
    //                     name: 8,
    //                     value: Expression::Identifier(3),
    //                 },
    //                 // d = a4
    //                 Assignment {
    //                     name: 9,
    //                     value: Expression::Identifier(7),
    //                 },
    //                 // [a1, b, c, d]
    //                 Assignment {
    //                     name: 10,
    //                     value: Expression::List(vec![
    //                         Expression::Identifier(4),
    //                         Expression::Identifier(8),
    //                         Expression::Identifier(1),
    //                         Expression::Identifier(9),
    //                     ]),
    //                 },
    //             ],
    //             vec![
    //                 None,
    //                 Some(Err("too many variables, try defining 'a2'".into())),
    //                 Some(Ok(1)),
    //                 Some(Ok(0)),
    //                 Some(Err("too many variables, try defining 'a4'".into())),
    //                 Some(Ok(10)),
    //             ],
    //         )
    //     );
    // }

    #[test]
    fn efficiency() {
        // need to fix this ugh
        assert_eq!(
            resolve_names(&[
                ElAssign {
                    name: "a".into(),
                    value: AWith {
                        body: bx(ANum(1.0)),
                        substitutions: vec![("b".into(), ANum(2.0))],
                    },
                },
                ElExpr(AWith {
                    body: bx(AId("a".into())),
                    substitutions: vec![("c".into(), ANum(3.0))],
                }),
            ]),
            (
                vec![
                    // b = 2
                    Assignment {
                        name: 0,
                        value: Expression::Number(2.0)
                    },
                    // a = 1
                    Assignment {
                        name: 1,
                        value: Expression::Number(1.0)
                    },
                    // c = 3
                    Assignment {
                        name: 2,
                        value: Expression::Number(3.0)
                    },
                    // a
                    Assignment {
                        name: 3,
                        value: Expression::Identifier(1)
                    }
                ],
                vec![Some(Ok(1)), Some(Ok(3))]
            ),
        );
    }
}
