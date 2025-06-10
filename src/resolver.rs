use std::collections::HashMap;

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
    pub id: usize,
    pub value: Expression,
}

#[derive(Debug, PartialEq)]
pub struct Body {
    pub assignments: Vec<Assignment>,
    pub value: Box<Expression>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Dependency {
    Assignment(usize),
    Unsubstituted,
}

type Dependencies<'a> = HashMap<&'a str, Dependency>;

trait Merge {
    fn merge(&mut self, other: Self);

    fn merged(mut self, other: Self) -> Self
    where
        Self: Sized,
    {
        self.merge(other);
        self
    }
}

impl<'a> Merge for Dependencies<'a> {
    fn merge(&mut self, other: Self) {
        for (name, dep) in &other {
            let d = self.entry(name).or_insert(Dependency::Unsubstituted);

            if let Dependency::Assignment(id) = dep {
                if let Dependency::Assignment(i) = d {
                    assert_eq!(i, id);
                }

                *d = *dep;
            }
        }

        self.extend(other);
    }
}

struct Resolver<'a> {
    globals: HashMap<&'a str, Result<&'a ExpressionListEntry, String>>,
    assignments: Vec<Assignment>,
    id_counter: usize,
    substitutions: HashMap<&'a str, Vec<usize>>,
    derived: HashMap<&'a str, Vec<(usize, Dependencies<'a>)>>,
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
            id_counter: 0,
            substitutions: HashMap::new(),
            derived: HashMap::new(),
        }
    }

    fn create_assignment(&mut self, value: Expression) -> usize {
        let id = self.id_counter;
        self.id_counter += 1;
        self.assignments.push(Assignment { id, value });
        id
    }

    fn has_substitution(&self, name: &str) -> bool {
        self.substitutions
            .get(name)
            .map(|v| !v.is_empty())
            .unwrap_or_default()
    }

    fn resolve_variable(&mut self, name: &'a str) -> Result<(usize, Dependencies<'a>), String> {
        if let Some(id) = self.substitutions.get(name).and_then(|v| v.last()) {
            let mut deps = Dependencies::new();
            deps.insert(name, Dependency::Assignment(*id));
            return Ok((*id, deps));
        }

        if let Some((id, deps)) = self.derived.get(name).and_then(|v| v.last()) {
            if deps.iter().all(|(n, d)| match d {
                Dependency::Assignment(i) => {
                    self.substitutions
                        .get(n)
                        .and_then(|v| v.last())
                        .cloned()
                        .unwrap_or_else(|| self.derived.get(n).unwrap().last().unwrap().0)
                        == *i
                }
                Dependency::Unsubstituted => !self.has_substitution(n),
            }) {
                let mut deps = deps.clone();
                deps.insert(name, Dependency::Assignment(*id));
                return Ok((*id, deps));
            }
        }

        let entry = self
            .globals
            .get(name)
            .ok_or_else(|| format!("'{name}' is not defined"))?
            .clone()?;

        match entry {
            ExpressionListEntry::Assignment { value, .. } => {
                let (value, mut deps) = self.resolve_expression(value)?;
                let id = self.create_assignment(value);
                self.derived
                    .entry(name)
                    .or_default()
                    .push((id, deps.clone()));
                deps.insert(name, Dependency::Assignment(id));
                Ok((id, deps))
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
    ) -> Result<(Vec<Expression>, Dependencies<'a>), String> {
        let mut result = vec![];
        let mut deps = Dependencies::new();

        for e in es {
            let (e, d) = self.resolve_expression(e)?;
            result.push(e);
            deps.merge(d);
        }

        Ok((result, deps))
    }

    fn resolve_call(
        &mut self,
        callee: &'a str,
        args: &'a [ast::Expression],
    ) -> Result<(Expression, Dependencies<'a>), String> {
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

        let mut seen = HashMap::new();
        let mut argument_deps = Dependencies::new();

        for (name, value) in parameters.iter().zip(args) {
            if seen.contains_key(name.as_str()) {
                return Err(format!(
                    "cannot use '{name}' for multiple parameters of this function"
                ));
            }

            let (value, d) = self.resolve_expression(value)?;
            argument_deps.merge(d);
            let id = self.create_assignment(value);
            self.substitutions.entry(name).or_default().push(id);
            seen.insert(name.as_str(), id);
        }

        // Don't unwrap yet because we want to clean up self.substitutions. But
        // maybe it's actually okay to leave it messy and it can cleaned up once
        // at the end of each expression list entry?
        let body = self.resolve_expression(body);

        for (name, id) in &seen {
            let popped = self.substitutions.get_mut(name).unwrap().pop();
            assert_eq!(popped, Some(*id))
        }

        let (body, mut body_deps) = body?;

        for (name, id) in &seen {
            if let Some(dep) = body_deps.remove(name) {
                assert_eq!(dep, Dependency::Assignment(*id));
            }
        }

        for (name, dep) in &mut body_deps {
            if self.has_substitution(name) {
                continue;
            }

            if *dep == Dependency::Unsubstituted {
                continue;
            }

            let v = self.derived.get_mut(name).unwrap();
            let d = &v.last().unwrap().1;

            for (n, i) in &seen {
                if let Some(x) = d.get(n) {
                    assert_eq!(*x, Dependency::Assignment(*i));
                }
            }

            if seen.iter().any(|(n, _)| d.contains_key(n)) {
                v.pop();
                *dep = Dependency::Unsubstituted;
            }
        }

        Ok((body, argument_deps.merged(body_deps)))
    }

    fn resolve_expression(
        &mut self,
        e: &'a ast::Expression,
    ) -> Result<(Expression, Dependencies<'a>), String> {
        match e {
            ast::Expression::Number(value) => Ok((Expression::Number(*value), Dependencies::new())),
            ast::Expression::Identifier(name) => {
                let (id, d) = self.resolve_variable(name)?;
                Ok((Expression::Identifier(id), d))
            }
            ast::Expression::List(list) => {
                let (list, d) = self.resolve_expressions(list)?;
                Ok((Expression::List(list), d))
            }
            ast::Expression::ListRange {
                before_ellipsis,
                after_ellipsis,
            } => {
                let (before_ellipsis, d0) = self.resolve_expressions(before_ellipsis)?;
                let (after_ellipsis, d1) = self.resolve_expressions(after_ellipsis)?;
                Ok((
                    Expression::ListRange {
                        before_ellipsis,
                        after_ellipsis,
                    },
                    d0.merged(d1),
                ))
            }
            ast::Expression::UnaryOperation { operation, arg } => {
                let (arg, d) = self.resolve_expression(arg)?;
                Ok((
                    Expression::UnaryOperation {
                        operation: *operation,
                        arg: Box::new(arg),
                    },
                    d,
                ))
            }
            ast::Expression::BinaryOperation {
                operation,
                left,
                right,
            } => {
                let (left, d0) = self.resolve_expression(left)?;
                let (right, d1) = self.resolve_expression(right)?;
                Ok((
                    Expression::BinaryOperation {
                        operation: *operation,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    d0.merged(d1),
                ))
            }
            ast::Expression::CallOrMultiply { callee, args } => {
                if let Some(Ok(ExpressionListEntry::FunctionDeclaration { .. })) =
                    self.globals.get(callee.as_str())
                {
                    self.resolve_call(callee, args)
                } else {
                    let (left, right) = (callee, args);
                    let name = left;
                    let (left, d0) = self.resolve_variable(left)?;
                    let (right, d1) = self.resolve_expressions(right)?;
                    let len = right.len();

                    if len == 1 || len == 2 {
                        let mut right_iter = right.into_iter();
                        Ok((
                            Expression::BinaryOperation {
                                operation: BinaryOperator::Mul,
                                left: Box::new(Expression::Identifier(left)),
                                right: Box::new(if len == 1 {
                                    right_iter.next().unwrap()
                                } else {
                                    Expression::BinaryOperation {
                                        operation: BinaryOperator::Point,
                                        left: Box::new(right_iter.next().unwrap()),
                                        right: Box::new(right_iter.next().unwrap()),
                                    }
                                }),
                            },
                            d0.merged(d1),
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
                let (operands, d) = self.resolve_expressions(operands)?;
                Ok((
                    Expression::ChainedComparison {
                        operands,
                        operators: operators.clone(),
                    },
                    d,
                ))
            }
            ast::Expression::Piecewise {
                test,
                consequent,
                alternate,
            } => {
                let (test, d0) = self.resolve_expression(test)?;
                let (consequent, d1) = self.resolve_expression(consequent)?;
                let mut d = d0.merged(d1);
                Ok((
                    Expression::Piecewise {
                        test: Box::new(test),
                        consequent: Box::new(consequent),
                        alternate: if let Some(e) = alternate {
                            let (alternate, d2) = self.resolve_expression(e)?;
                            d.merge(d2);
                            Some(Box::new(alternate))
                        } else {
                            None
                        },
                    },
                    d,
                ))
            }
            ast::Expression::SumProd { .. } => todo!(),
            ast::Expression::With {
                body,
                substitutions,
            } => {
                let mut seen = HashMap::new();
                let mut substitution_deps = Dependencies::new();

                for (name, value) in substitutions {
                    if seen.contains_key(name.as_str()) {
                        return Err(format!(
                            "a 'with' expression cannot make multiple substitutions for '{name}'"
                        ));
                    }

                    let (value, d) = self.resolve_expression(value)?;
                    substitution_deps.merge(d);
                    let id = self.create_assignment(value);
                    self.substitutions.entry(name).or_default().push(id);
                    seen.insert(name.as_str(), id);
                }

                // Don't unwrap yet because we want to clean up
                // self.substitutions. But maybe it's actually okay to leave it
                // messy and it can cleaned up once at the end of each
                // expression list entry?
                let body = self.resolve_expression(body);

                for (name, id) in &seen {
                    let popped = self.substitutions.get_mut(name).unwrap().pop();
                    assert_eq!(popped, Some(*id))
                }

                let (body, mut body_deps) = body?;

                for (name, id) in &seen {
                    if let Some(dep) = body_deps.remove(name) {
                        assert_eq!(dep, Dependency::Assignment(*id));
                    }
                }

                for (name, dep) in &mut body_deps {
                    if self.has_substitution(name) {
                        continue;
                    }

                    if *dep == Dependency::Unsubstituted {
                        continue;
                    }

                    let v = self.derived.get_mut(name).unwrap();
                    let d = &v.last().unwrap().1;

                    for (n, i) in &seen {
                        if let Some(x) = d.get(n) {
                            assert_eq!(*x, Dependency::Assignment(*i));
                        }
                    }

                    if seen.iter().any(|(n, _)| d.contains_key(n)) {
                        v.pop();
                        *dep = Dependency::Unsubstituted;
                    }
                }

                Ok((body, substitution_deps.merged(body_deps)))
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
            // we could add in asserts here to check things like no substitutions or derived is all length 1 or 0
            ExpressionListEntry::Assignment { name, value } => {
                if let Some((id, _)) = resolver
                    .derived
                    .get(name.as_str())
                    .inspect(|v| assert!(v.len() <= 1))
                    .and_then(|v| v.last())
                {
                    Some(Ok(*id))
                } else {
                    match resolver.resolve_expression(value) {
                        Ok((value, deps)) => {
                            let id = resolver.create_assignment(value);

                            if resolver.globals.get(name.as_str()).unwrap().is_ok() {
                                resolver
                                    .derived
                                    .entry(name)
                                    .or_default()
                                    .push((id, deps.clone()));
                            }

                            Some(Ok(id))
                        }
                        Err(error) => Some(Err(error)),
                    }
                }
            }
            ExpressionListEntry::FunctionDeclaration { .. } => None,
            ExpressionListEntry::Relation(..) => todo!(),
            ExpressionListEntry::Expression(expression) => {
                Some(match resolver.resolve_expression(expression) {
                    Ok((value, _)) => {
                        let id = resolver.create_assignment(value);
                        Ok(id)
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
                        id: 0,
                        value: Expression::Number(5.0),
                    },
                    Assignment {
                        id: 1,
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
                        id: 0,
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: 1,
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
                        id: 0,
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: 1,
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
        panic!("this currrently fails by stack overflow");
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
                // c = 1
                ElAssign {
                    name: "c".into(),
                    value: ANum(1.0),
                },
                // b = c
                ElAssign {
                    name: "b".into(),
                    value: AId("c".into()),
                },
                // a = b with c = 2
                ElAssign {
                    name: "a".into(),
                    value: AWith {
                        body: bx(AId("b".into())),
                        substitutions: vec![("c".into(), ANum(2.0))],
                    },
                },
                // a with b = 3
                ElExpr(AWith {
                    body: bx(AId("a".into())),
                    substitutions: vec![("b".into(), ANum(3.0))],
                }),
                // b with c = 4
                ElExpr(AWith {
                    body: bx(AId("b".into())),
                    substitutions: vec![("c".into(), ANum(4.0))],
                }),
                // a with c = 5
                ElExpr(AWith {
                    body: bx(AId("a".into())),
                    substitutions: vec![("c".into(), ANum(5.0))],
                }),
            ]),
            (
                vec![
                    // c = 1
                    Assignment {
                        id: 0,
                        value: Expression::Number(1.0),
                    },
                    // b = c
                    Assignment {
                        id: 1,
                        value: Expression::Identifier(0),
                    },
                    // with c = 2
                    Assignment {
                        id: 2,
                        value: Expression::Number(2.0),
                    },
                    // b = c
                    Assignment {
                        id: 3,
                        value: Expression::Identifier(2),
                    },
                    // a = b
                    Assignment {
                        id: 4,
                        value: Expression::Identifier(3),
                    },
                    // with b = 3
                    Assignment {
                        id: 5,
                        value: Expression::Number(3.0),
                    },
                    // with c = 2
                    Assignment {
                        id: 6,
                        value: Expression::Number(2.0),
                    },
                    // a = b
                    Assignment {
                        id: 7,
                        value: Expression::Identifier(5),
                    },
                    // a
                    Assignment {
                        id: 8,
                        value: Expression::Identifier(7),
                    },
                    // with c = 4
                    Assignment {
                        id: 9,
                        value: Expression::Number(4.0),
                    },
                    // b = c
                    Assignment {
                        id: 10,
                        value: Expression::Identifier(9),
                    },
                    // b
                    Assignment {
                        id: 11,
                        value: Expression::Identifier(10),
                    },
                    // with c = 5
                    Assignment {
                        id: 12,
                        value: Expression::Number(5.0),
                    },
                    // a
                    Assignment {
                        id: 13,
                        value: Expression::Identifier(4),
                    },
                ],
                vec![
                    Some(Ok(0)),
                    Some(Ok(1)),
                    Some(Ok(4)),
                    Some(Ok(8)),
                    Some(Ok(11)),
                    Some(Ok(13)),
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
                    id: 0,
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
                        id: 0,
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: 1,
                        value: Expression::BinaryOperation {
                            operation: BinaryOperator::Mul,
                            left: bx(Expression::Identifier(0)),
                            right: bx(Expression::Number(2.0)),
                        },
                    },
                    Assignment {
                        id: 2,
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
                        id: 0,
                        value: Expression::Number(5.0),
                    },
                    // c = a3
                    Assignment {
                        id: 1,
                        value: Expression::Identifier(0),
                    },
                    // with a1 = 6
                    Assignment {
                        id: 2,
                        value: Expression::Number(6.0),
                    },
                    // with a2 = 7
                    Assignment {
                        id: 3,
                        value: Expression::Number(7.0),
                    },
                    // a1 = 1
                    Assignment {
                        id: 4,
                        value: Expression::Number(1.0),
                    },
                    // a2 = 2
                    Assignment {
                        id: 5,
                        value: Expression::Number(2.0),
                    },
                    // a3 = 3
                    Assignment {
                        id: 6,
                        value: Expression::Number(3.0),
                    },
                    // a4 = 4
                    Assignment {
                        id: 7,
                        value: Expression::Number(4.0),
                    },
                    // b = a2
                    Assignment {
                        id: 8,
                        value: Expression::Identifier(5),
                    },
                    // c = a3
                    Assignment {
                        id: 9,
                        value: Expression::Identifier(6),
                    },
                    // d = a4
                    Assignment {
                        id: 10,
                        value: Expression::Identifier(7),
                    },
                    // [a1, b, c, d]
                    Assignment {
                        id: 11,
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
                        id: 0,
                        value: Expression::Number(2.0)
                    },
                    // a = 1
                    Assignment {
                        id: 1,
                        value: Expression::Number(1.0)
                    },
                    // c = 3
                    Assignment {
                        id: 2,
                        value: Expression::Number(3.0)
                    },
                    // a
                    Assignment {
                        id: 3,
                        value: Expression::Identifier(1)
                    }
                ],
                vec![Some(Ok(1)), Some(Ok(3))]
            ),
        );
    }
}
