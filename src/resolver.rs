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
        lists: Vec<Assignment>,
    },
}

#[derive(Debug, PartialEq)]
pub struct Assignment {
    pub id: usize,
    pub name: String,
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
    NotDynamic,
}

#[derive(Clone)]
struct Dependencies<'a> {
    level: usize,
    map: HashMap<&'a str, Dependency>,
}

impl<'a> Dependencies<'a> {
    fn with_level(level: usize) -> Self {
        Self {
            level,
            map: HashMap::new(),
        }
    }

    fn none() -> Self {
        Dependencies::with_level(0)
    }
}

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
        for (name, dep) in &other.map {
            let d = self.map.entry(name).or_insert(Dependency::NotDynamic);

            if let Dependency::Assignment(id) = dep {
                if let Dependency::Assignment(i) = d {
                    assert_eq!(i, id);
                }

                *d = *dep;
            }
        }

        self.map.extend(other.map);
        self.level = self.level.max(other.level);
    }
}

struct ScopeMap<'a, T>(HashMap<&'a str, Vec<T>>);

impl<'a, T> Default for ScopeMap<'a, T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<'a, T> ScopeMap<'a, T> {
    fn get(&self, k: &str) -> Option<&T> {
        self.0.get(k).and_then(|v| v.last())
    }

    fn contains_key(&self, k: &str) -> bool {
        self.0.get(k).map(|v| !v.is_empty()).unwrap_or_default()
    }

    fn push(&mut self, k: &'a str, v: T) {
        self.0.entry(k).or_default().push(v);
    }

    fn pop(&mut self, k: &str) -> Option<T> {
        self.0.get_mut(k).and_then(|v| v.pop())
    }
}

struct Resolver<'a> {
    globals: HashMap<&'a str, Result<&'a ExpressionListEntry, String>>,
    assignments: Vec<Vec<Assignment>>,
    id_counter: usize,
    dynamic_scope: ScopeMap<'a, (usize, Dependencies<'a>)>,
    global_scope: ScopeMap<'a, (usize, Dependencies<'a>)>,
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
            assignments: vec![vec![]],
            id_counter: 0,
            dynamic_scope: Default::default(),
            global_scope: Default::default(),
        }
    }

    fn create_assignment(&mut self, name: &str, value: Expression) -> Assignment {
        let id = self.id_counter;
        self.id_counter += 1;
        Assignment {
            id,
            name: name.to_string(),
            value,
        }
    }

    fn push_assignment(&mut self, name: &str, level: usize, value: Expression) -> usize {
        let assignment = self.create_assignment(name, value);
        let id = assignment.id;
        self.assignments[level].push(assignment);
        id
    }

    fn resolve_variable(&mut self, name: &'a str) -> (Result<usize, String>, Dependencies<'a>) {
        if let Some((id, deps)) = self
            .dynamic_scope
            .get(name)
            .or_else(|| self.global_scope.get(name))
        {
            if deps.map.iter().all(|(n, d)| match d {
                Dependency::Assignment(i) => {
                    self.dynamic_scope
                        .get(n)
                        .unwrap_or_else(|| self.global_scope.get(n).unwrap())
                        .0
                        == *i
                }
                Dependency::NotDynamic => !self.dynamic_scope.contains_key(n),
            }) {
                let mut deps = deps.clone();
                deps.map.insert(name, Dependency::Assignment(*id));
                return (Ok(*id), deps);
            }
        }

        let entry = match self
            .globals
            .get(name)
            .ok_or_else(|| format!("'{name}' is not defined"))
            .cloned()
            .and_then(|x| x)
        {
            Ok(entry) => entry,
            Err(err) => return (Err(err), Dependencies::none()),
        };

        match entry {
            ExpressionListEntry::Assignment { value, .. } => {
                let (value, mut deps) = match self.resolve_expression(value) {
                    (Ok(v), d) => (v, d),
                    (Err(e), d) => return (Err(e), d),
                };
                let id = self.push_assignment(name, deps.level, value);
                self.global_scope.push(name, (id, deps.clone()));
                deps.map.insert(name, Dependency::Assignment(id));
                (Ok(id), deps)
            }
            ExpressionListEntry::FunctionDeclaration { .. } => (
                Err(format!("'{name}' is a function, try using parentheses")),
                Dependencies::none(),
            ),
            _ => unreachable!(),
        }
    }

    fn resolve_expressions(
        &mut self,
        es: &'a [ast::Expression],
    ) -> (Result<Vec<Expression>, String>, Dependencies<'a>) {
        let mut result = vec![];
        let mut deps = Dependencies::none();

        for e in es {
            let (e, d) = self.resolve_expression(e);
            deps.merge(d);
            match e {
                Ok(e) => result.push(e),
                Err(e) => return (Err(e), deps),
            }
        }

        (Ok(result), deps)
    }

    fn resolve_dynamic(
        &mut self,
        body: &'a ast::Expression,
        bindings: impl Iterator<Item = (&'a String, &'a ast::Expression)>,
        error_string: impl FnOnce(&str) -> String,
    ) -> (Result<Expression, String>, Dependencies<'a>) {
        let mut seen = HashMap::new();
        let mut binding_deps = Dependencies::none();

        for (name, value) in bindings {
            if seen.contains_key(name.as_str()) {
                return (Err(error_string(name)), binding_deps);
            }

            let (value, d) = match self.resolve_expression(value) {
                (Ok(v), d) => (v, d),
                (Err(e), d) => return (Err(e), binding_deps.merged(d)),
            };
            let id = self.push_assignment(name, d.level, value);
            seen.insert(name.as_str(), (id, d.level));
            binding_deps.merge(d);
        }

        for (name, &(id, level)) in &seen {
            self.dynamic_scope
                .push(name, (id, Dependencies::with_level(level)));
        }

        let (body, mut body_deps) = self.resolve_expression(body);

        for (name, (id, _)) in &seen {
            let popped = self.dynamic_scope.pop(name);
            assert_eq!(popped.map(|(i, _)| i), Some(*id))
        }

        for (name, (id, _)) in &seen {
            if let Some(dep) = body_deps.map.remove(name) {
                assert_eq!(dep, Dependency::Assignment(*id));
            }
        }

        for (name, dep) in &mut body_deps.map {
            if self.dynamic_scope.contains_key(name) {
                continue;
            }

            if *dep == Dependency::NotDynamic {
                continue;
            }

            let d = &self.global_scope.get(name).unwrap().1;

            for (n, (i, _)) in &seen {
                if let Some(x) = d.map.get(n) {
                    assert_eq!(*x, Dependency::Assignment(*i));
                }
            }

            if seen.iter().any(|(n, _)| d.map.contains_key(n)) {
                self.global_scope.pop(name);
                *dep = Dependency::NotDynamic;
            }
        }

        (body, binding_deps.merged(body_deps))
    }

    fn resolve_call(
        &mut self,
        callee: &'a str,
        args: &'a [ast::Expression],
    ) -> (Result<Expression, String>, Dependencies<'a>) {
        let err = |s| (Err(s), Dependencies::none());
        let (parameters, body) = match self.globals.get(callee).cloned() {
            Some(Ok(ExpressionListEntry::FunctionDeclaration {
                parameters, body, ..
            })) => (parameters, body),
            Some(Ok(_)) => return err(format!("variable '{callee}' can't be used as a function")),
            Some(Err(e)) => return err(e),
            None => return err(format!("'{callee}' is not defined")),
        };

        if parameters.len() != args.len() {
            return err(format!(
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

        self.resolve_dynamic(body, parameters.iter().zip(args.iter()), |name| {
            format!("cannot use '{name}' for multiple parameters of this function")
        })
    }

    fn resolve_expression(
        &mut self,
        e: &'a ast::Expression,
    ) -> (Result<Expression, String>, Dependencies<'a>) {
        match e {
            ast::Expression::Number(value) => {
                (Ok(Expression::Number(*value)), Dependencies::none())
            }
            ast::Expression::Identifier(name) => {
                let (id, d) = self.resolve_variable(name);
                (id.map(Expression::Identifier), d)
            }
            ast::Expression::List(list) => {
                let (list, d) = self.resolve_expressions(list);
                (list.map(Expression::List), d)
            }
            ast::Expression::ListRange {
                before_ellipsis,
                after_ellipsis,
            } => {
                let (before_ellipsis, d0) = match self.resolve_expressions(before_ellipsis) {
                    (Ok(v), d) => (v, d),
                    (Err(e), d) => return (Err(e), d),
                };
                let (after_ellipsis, d1) = match self.resolve_expressions(after_ellipsis) {
                    (Ok(v), d) => (v, d),
                    (Err(e), d) => return (Err(e), d0.merged(d)),
                };
                (
                    Ok(Expression::ListRange {
                        before_ellipsis,
                        after_ellipsis,
                    }),
                    d0.merged(d1),
                )
            }
            ast::Expression::UnaryOperation { operation, arg } => {
                let (arg, d) = self.resolve_expression(arg);
                (
                    arg.map(|arg| Expression::UnaryOperation {
                        operation: *operation,
                        arg: Box::new(arg),
                    }),
                    d,
                )
            }
            ast::Expression::BinaryOperation {
                operation,
                left,
                right,
            } => {
                let (left, d0) = match self.resolve_expression(left) {
                    (Ok(v), d) => (v, d),
                    (Err(e), d) => return (Err(e), d),
                };
                let (right, d1) = match self.resolve_expression(right) {
                    (Ok(v), d) => (v, d),
                    (Err(e), d) => return (Err(e), d0.merged(d)),
                };
                (
                    Ok(Expression::BinaryOperation {
                        operation: *operation,
                        left: Box::new(left),
                        right: Box::new(right),
                    }),
                    d0.merged(d1),
                )
            }
            ast::Expression::CallOrMultiply { callee, args } => {
                if let Some(Ok(ExpressionListEntry::FunctionDeclaration { .. })) =
                    self.globals.get(callee.as_str())
                {
                    self.resolve_call(callee, args)
                } else {
                    let (left, right) = (callee, args);
                    let name = left;
                    let (left, d0) = match self.resolve_variable(left) {
                        (Ok(v), d) => (v, d),
                        (Err(e), d) => return (Err(e), d),
                    };
                    let len = right.len();
                    let (right, d1) = match self.resolve_expressions(right) {
                        (Ok(v), d) => (v, d),
                        (Err(e), d) => return (Err(e), d0.merged(d)),
                    };
                    (
                        if len == 1 || len == 2 {
                            let mut right_iter = right.into_iter();
                            Ok(Expression::BinaryOperation {
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
                            })
                        } else {
                            Err(if len == 0 {
                                format!("variable '{name}' can't be used as a function")
                            } else {
                                "points may only have 2 coordinates".into()
                            })
                        },
                        d0.merged(d1),
                    )
                }
            }
            ast::Expression::Call { callee, args } => self.resolve_call(callee, args),
            ast::Expression::ChainedComparison(ast::ChainedComparison {
                operands,
                operators,
            }) => {
                let (operands, d) = self.resolve_expressions(operands);
                (
                    operands.map(|operands| Expression::ChainedComparison {
                        operands,
                        operators: operators.clone(),
                    }),
                    d,
                )
            }
            ast::Expression::Piecewise {
                test,
                consequent,
                alternate,
            } => {
                let (test, d0) = match self.resolve_expression(test) {
                    (Ok(v), d) => (v, d),
                    (Err(e), d) => return (Err(e), d),
                };
                let (consequent, d1) = match self.resolve_expression(consequent) {
                    (Ok(v), d) => (v, d),
                    (Err(e), d) => return (Err(e), d0.merged(d)),
                };
                let mut d = d0.merged(d1);
                (
                    Ok(Expression::Piecewise {
                        test: Box::new(test),
                        consequent: Box::new(consequent),
                        alternate: if let Some(e) = alternate {
                            let (alternate, d2) = self.resolve_expression(e);
                            d.merge(d2);
                            match alternate {
                                Ok(a) => Some(Box::new(a)),
                                Err(e) => return (Err(e), d),
                            }
                        } else {
                            None
                        },
                    }),
                    d,
                )
            }
            ast::Expression::SumProd { .. } => todo!(),
            ast::Expression::With {
                body,
                substitutions,
            } => self.resolve_dynamic(body, substitutions.iter().map(|(n, v)| (n, v)), |name| {
                format!("a 'with' expression cannot make multiple substitutions for '{name}'")
            }),
            ast::Expression::For { body, lists } => {
                let mut seen = HashMap::new();
                let mut list_deps = Dependencies::none();
                let mut resolved_lists = vec![];

                for (name, value) in lists {
                    if seen.contains_key(name.as_str()) {
                        return (Err(format!(
                            "you can't define '{name}' more than once on the right-hand side of 'for'"
                        )), list_deps);
                    }

                    let (value, d) = match self.resolve_expression(value) {
                        (Ok(v), d) => (v, d),
                        (Err(e), d) => return (Err(e), list_deps.merged(d)),
                    };
                    let assignment = self.create_assignment(name, value);
                    seen.insert(name.as_str(), assignment.id);
                    list_deps.merge(d);
                    resolved_lists.push(assignment);
                }

                let new_level = self.assignments.len();
                self.assignments.push(vec![]);

                for (name, id) in &seen {
                    self.dynamic_scope
                        .push(name, (*id, Dependencies::with_level(new_level)));
                }

                let (body, mut body_deps) = self.resolve_expression(body);

                for (name, id) in &seen {
                    let popped = self.dynamic_scope.pop(name);
                    assert_eq!(popped.map(|(i, _)| i), Some(*id))
                }

                let assignments = self.assignments.pop().unwrap();

                for (name, id) in &seen {
                    if let Some(dep) = body_deps.map.remove(name) {
                        assert_eq!(dep, Dependency::Assignment(*id));
                    }
                }

                for (name, dep) in &mut body_deps.map {
                    if self.dynamic_scope.contains_key(name) {
                        continue;
                    }

                    if *dep == Dependency::NotDynamic {
                        continue;
                    }

                    let d = &self.global_scope.get(name).unwrap().1;

                    for (n, i) in &seen {
                        if let Some(x) = d.map.get(n) {
                            assert_eq!(*x, Dependency::Assignment(*i));
                        }
                    }

                    if seen.iter().any(|(n, _)| d.map.contains_key(n)) {
                        self.global_scope.pop(name);
                        *dep = Dependency::NotDynamic;
                    }
                }

                body_deps.level = 0;

                for (name, dep) in &mut body_deps.map {
                    if *dep == Dependency::NotDynamic {
                        continue;
                    }

                    body_deps.level = self
                        .dynamic_scope
                        .get(name)
                        .unwrap_or_else(|| self.global_scope.get(name).unwrap())
                        .1
                        .level
                        .max(body_deps.level);
                }

                (
                    body.map(|body| Expression::For {
                        body: Body {
                            assignments,
                            value: Box::new(body),
                        },
                        lists: resolved_lists,
                    }),
                    list_deps.merged(body_deps),
                )
            }
        }
    }
}

pub fn resolve_names(
    list: &[ExpressionListEntry],
) -> (Vec<Assignment>, Vec<Option<Result<usize, String>>>) {
    let mut resolver = Resolver::new(list);

    let assignment_indices: Vec<_> = list
        .iter()
        .map(|e| {
            assert!(resolver.dynamic_scope.0.iter().all(|(_, v)| v.is_empty()));
            assert!(resolver.global_scope.0.iter().all(|(_, v)| v.len() <= 1));

            match e {
                ExpressionListEntry::Assignment { name, value } => {
                    if let Some((id, _)) = resolver.global_scope.get(name) {
                        Some(Ok(*id))
                    } else {
                        match resolver.resolve_expression(value) {
                            (Ok(value), deps) => {
                                let id = resolver.push_assignment(name, 0, value);

                                if resolver.globals.get(name.as_str()).unwrap().is_ok() {
                                    resolver.global_scope.push(name, (id, deps));
                                }

                                Some(Ok(id))
                            }
                            (Err(error), _) => Some(Err(error)),
                        }
                    }
                }
                ExpressionListEntry::FunctionDeclaration { .. } => None,
                ExpressionListEntry::Relation(..) => todo!(),
                ExpressionListEntry::Expression(expression) => {
                    Some(match resolver.resolve_expression(expression) {
                        (Ok(value), _) => {
                            let id = resolver.push_assignment("<anonymous>", 0, value);
                            Ok(id)
                        }
                        (Err(error), _) => Err(error),
                    })
                }
            }
        })
        .collect();

    let assignments = resolver.assignments.pop().unwrap();
    assert!(resolver.assignments.is_empty());
    (assignments, assignment_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::{
        BinaryOperator as ABo,
        Expression::{
            BinaryOperation as ABop,
            // Call as ACall,
            CallOrMultiply as ACallMul,
            // ChainedComparison as AComparison,
            For as AFor,
            Identifier as AId,
            List as AList,
            ListRange as AListRange,
            Number as ANum,
            // Piecewise as APiecewise,
            // SumProd as ASumProd,
            // UnaryOperation as AUop,
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
                        name: "<anonymous>".into(),
                        value: Expression::Number(5.0),
                    },
                    Assignment {
                        id: 1,
                        name: "<anonymous>".into(),
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
                        name: "c".into(),
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: 1,
                        name: "b".into(),
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
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: 1,
                        name: "a".into(),
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
                        name: "c".into(),
                        value: Expression::Number(1.0),
                    },
                    // b = c
                    Assignment {
                        id: 1,
                        name: "b".into(),
                        value: Expression::Identifier(0),
                    },
                    // with c = 2
                    Assignment {
                        id: 2,
                        name: "c".into(),
                        value: Expression::Number(2.0),
                    },
                    // b = c
                    Assignment {
                        id: 3,
                        name: "b".into(),
                        value: Expression::Identifier(2),
                    },
                    // a = b
                    Assignment {
                        id: 4,
                        name: "a".into(),
                        value: Expression::Identifier(3),
                    },
                    // with b = 3
                    Assignment {
                        id: 5,
                        name: "b".into(),
                        value: Expression::Number(3.0),
                    },
                    // with c = 2
                    Assignment {
                        id: 6,
                        name: "c".into(),
                        value: Expression::Number(2.0),
                    },
                    // a = b
                    Assignment {
                        id: 7,
                        name: "a".into(),
                        value: Expression::Identifier(5),
                    },
                    // a
                    Assignment {
                        id: 8,
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(7),
                    },
                    // with c = 4
                    Assignment {
                        id: 9,
                        name: "c".into(),
                        value: Expression::Number(4.0),
                    },
                    // b = c
                    Assignment {
                        id: 10,
                        name: "b".into(),
                        value: Expression::Identifier(9),
                    },
                    // b
                    Assignment {
                        id: 11,
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(10),
                    },
                    // with c = 5
                    Assignment {
                        id: 12,
                        name: "c".into(),
                        value: Expression::Number(5.0),
                    },
                    // a
                    Assignment {
                        id: 13,
                        name: "<anonymous>".into(),
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
                // a()
                ElExpr(ACallMul {
                    callee: "a".into(),
                    args: vec![],
                }),
                // b = 1
                ElAssign {
                    name: "b".into(),
                    value: ANum(1.0),
                },
                // b()
                ElExpr(ACallMul {
                    callee: "b".into(),
                    args: vec![],
                }),
                // c() = 2
                ElFunction {
                    name: "c".into(),
                    parameters: vec![],
                    body: ANum(2.0),
                },
                // c
                ElExpr(AId("c".into())),
            ]),
            (
                vec![Assignment {
                    id: 0,
                    name: "b".into(),
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
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: 1,
                        name: "<anonymous>".into(),
                        value: Expression::BinaryOperation {
                            operation: BinaryOperator::Mul,
                            left: bx(Expression::Identifier(0)),
                            right: bx(Expression::Number(2.0)),
                        },
                    },
                    Assignment {
                        id: 2,
                        name: "<anonymous>".into(),
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
                        name: "a3".into(),
                        value: Expression::Number(5.0),
                    },
                    // c = a3
                    Assignment {
                        id: 1,
                        name: "c".into(),
                        value: Expression::Identifier(0),
                    },
                    // with a1 = 6
                    Assignment {
                        id: 2,
                        name: "a1".into(),
                        value: Expression::Number(6.0),
                    },
                    // with a2 = 7
                    Assignment {
                        id: 3,
                        name: "a2".into(),
                        value: Expression::Number(7.0),
                    },
                    // a1 = 1
                    Assignment {
                        id: 4,
                        name: "a1".into(),
                        value: Expression::Number(1.0),
                    },
                    // a2 = 2
                    Assignment {
                        id: 5,
                        name: "a2".into(),
                        value: Expression::Number(2.0),
                    },
                    // a3 = 3
                    Assignment {
                        id: 6,
                        name: "a3".into(),
                        value: Expression::Number(3.0),
                    },
                    // a4 = 4
                    Assignment {
                        id: 7,
                        name: "a4".into(),
                        value: Expression::Number(4.0),
                    },
                    // b = a2
                    Assignment {
                        id: 8,
                        name: "b".into(),
                        value: Expression::Identifier(5),
                    },
                    // c = a3
                    Assignment {
                        id: 9,
                        name: "c".into(),
                        value: Expression::Identifier(6),
                    },
                    // d = a4
                    Assignment {
                        id: 10,
                        name: "d".into(),
                        value: Expression::Identifier(7),
                    },
                    // [a1, b, c, d]
                    Assignment {
                        id: 11,
                        name: "<anonymous>".into(),
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
    //             // f(a1, a2, a3, a4) = [a1, b, c, d]
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
    //             // b = a2
    //             ElAssign {
    //                 name: "b".into(),
    //                 value: AId("a2".into()),
    //             },
    //             // c = a3
    //             ElAssign {
    //                 name: "c".into(),
    //                 value: AId("a3".into()),
    //             },
    //             // a3 = 5
    //             ElAssign {
    //                 name: "a3".into(),
    //                 value: ANum(5.0),
    //             },
    //             // d = a4
    //             ElAssign {
    //                 name: "d".into(),
    //                 value: AId("a4".into()),
    //             },
    //             // f(1, 2, 3, 4) with a1 = 6, a2 = 7
    //             ElExpr(AWith {
    //                 body: bx(ACallMul {
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
    //                     id: 0,
    //                     value: Expression::Number(5.0),
    //                 },
    //                 // c = a3
    //                 Assignment {
    //                     id: 1,
    //                     value: Expression::Identifier(0),
    //                 },
    //                 // with a1 = 6
    //                 Assignment {
    //                     id: 2,
    //                     value: Expression::Number(6.0),
    //                 },
    //                 // with a2 = 7
    //                 Assignment {
    //                     id: 3,
    //                     value: Expression::Number(7.0),
    //                 },
    //                 // a1 = 1
    //                 Assignment {
    //                     id: 4,
    //                     value: Expression::Number(1.0),
    //                 },
    //                 // a2 = 2
    //                 Assignment {
    //                     id: 5,
    //                     value: Expression::Number(2.0),
    //                 },
    //                 // a3 = 3
    //                 Assignment {
    //                     id: 6,
    //                     value: Expression::Number(3.0),
    //                 },
    //                 // a4 = 4
    //                 Assignment {
    //                     id: 7,
    //                     value: Expression::Number(4.0),
    //                 },
    //                 // b = a2
    //                 Assignment {
    //                     id: 8,
    //                     value: Expression::Identifier(3),
    //                 },
    //                 // d = a4
    //                 Assignment {
    //                     id: 9,
    //                     value: Expression::Identifier(7),
    //                 },
    //                 // [a1, b, c, d]
    //                 Assignment {
    //                     id: 10,
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
                        name: "b".into(),
                        value: Expression::Number(2.0)
                    },
                    // a = 1
                    Assignment {
                        id: 1,
                        name: "a".into(),
                        value: Expression::Number(1.0)
                    },
                    // c = 3
                    Assignment {
                        id: 2,
                        name: "c".into(),
                        value: Expression::Number(3.0)
                    },
                    // a
                    Assignment {
                        id: 3,
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(1)
                    }
                ],
                vec![Some(Ok(1)), Some(Ok(3))]
            ),
        );
    }

    #[test]
    fn list_comp() {
        assert_eq!(
            resolve_names(&[
                // p for j=c, i=[1]
                ElExpr(AFor {
                    body: bx(AId("p".into())),
                    lists: vec![
                        ("j".into(), AId("c".into())),
                        ("i".into(), AList(vec![ANum(1.0)])),
                    ],
                }),
                // p = (q,i+k)
                ElAssign {
                    name: "p".into(),
                    value: ABop {
                        operation: BinaryOperator::Point,
                        left: bx(AId("q".into())),
                        right: bx(ABop {
                            operation: ABo::Add,
                            left: bx(AId("i".into())),
                            right: bx(AId("k".into()))
                        }),
                    },
                },
                // c = [2]
                ElAssign {
                    name: "c".into(),
                    value: AList(vec![ANum(2.0)]),
                },
                // q = jj
                ElAssign {
                    name: "q".into(),
                    value: ABop {
                        operation: BinaryOperator::Mul,
                        left: bx(AId("j".into())),
                        right: bx(AId("j".into())),
                    },
                },
                // k = 3
                ElAssign {
                    name: "k".into(),
                    value: ANum(3.0),
                },
            ]),
            (
                vec![
                    // c = [2]
                    Assignment {
                        id: 0,
                        name: "c".into(),
                        value: Expression::List(vec![Expression::Number(2.0)]),
                    },
                    // k = 3
                    Assignment {
                        id: 4,
                        name: "k".into(),
                        value: Expression::Number(3.0),
                    },
                    // p for j=c, i=[1]
                    Assignment {
                        id: 6,
                        name: "<anonymous>".into(),
                        value: Expression::For {
                            body: Body {
                                assignments: vec![
                                    // q = jj
                                    Assignment {
                                        id: 3,
                                        name: "q".into(),
                                        value: Expression::BinaryOperation {
                                            operation: BinaryOperator::Mul,
                                            left: bx(Expression::Identifier(1)),
                                            right: bx(Expression::Identifier(1)),
                                        },
                                    },
                                    // p = (q,i+k)
                                    Assignment {
                                        id: 5,
                                        name: "p".into(),
                                        value: Expression::BinaryOperation {
                                            operation: BinaryOperator::Point,
                                            left: bx(Expression::Identifier(3)),
                                            right: bx(Expression::BinaryOperation {
                                                operation: BinaryOperator::Add,
                                                left: bx(Expression::Identifier(2)),
                                                right: bx(Expression::Identifier(4)),
                                            }),
                                        },
                                    },
                                ],
                                value: bx(Expression::Identifier(5)),
                            },
                            lists: vec![
                                // j=c
                                Assignment {
                                    id: 1,
                                    name: "j".into(),
                                    value: Expression::Identifier(0)
                                },
                                // i=[1]
                                Assignment {
                                    id: 2,
                                    name: "i".into(),
                                    value: Expression::List(vec![Expression::Number(1.0)])
                                },
                            ],
                        },
                    },
                ],
                vec![
                    Some(Ok(6)),
                    Some(Err("'j' is not defined".into())),
                    Some(Ok(0)),
                    Some(Err("'j' is not defined".into())),
                    Some(Ok(4)),
                ],
            ),
        );
    }

    #[test]
    fn nested_list_comps() {
        assert_eq!(
            resolve_names(&[
                // E = C[1] + D[1] for j=[1...4]
                ElAssign {
                    name: "E".into(),
                    value: AFor {
                        body: bx(ABop {
                            operation: ABo::Add,
                            left: bx(ABop {
                                operation: ABo::Index,
                                left: bx(AId("C".into())),
                                right: bx(ANum(1.0)),
                            }),
                            right: bx(ABop {
                                operation: ABo::Index,
                                left: bx(AId("D".into())),
                                right: bx(ANum(1.0)),
                            }),
                        }),
                        lists: vec![(
                            "j".into(),
                            AListRange {
                                before_ellipsis: vec![ANum(1.0)],
                                after_ellipsis: vec![ANum(4.0)],
                            },
                        )],
                    },
                },
                // C = B for i=[1...5]
                ElAssign {
                    name: "C".into(),
                    value: AFor {
                        body: bx(AId("B".into())),
                        lists: vec![(
                            "i".into(),
                            AListRange {
                                before_ellipsis: vec![ANum(1.0)],
                                after_ellipsis: vec![ANum(5.0)],
                            },
                        )],
                    },
                },
                // D = B + A + F for i=[1...3]
                ElAssign {
                    name: "D".into(),
                    value: AFor {
                        body: bx(ABop {
                            operation: ABo::Add,
                            left: bx(ABop {
                                operation: ABo::Add,
                                left: bx(AId("B".into())),
                                right: bx(AId("A".into())),
                            }),
                            right: bx(AId("F".into())),
                        }),
                        lists: vec![(
                            "i".into(),
                            AListRange {
                                before_ellipsis: vec![ANum(1.0)],
                                after_ellipsis: vec![ANum(3.0)],
                            },
                        )],
                    },
                },
                // B = i^2
                ElAssign {
                    name: "B".into(),
                    value: ABop {
                        operation: ABo::Pow,
                        left: bx(AId("i".into())),
                        right: bx(ANum(2.0)),
                    },
                },
                // F = i + j
                ElAssign {
                    // TODO: change this back to "J" and see why it was panicking
                    name: "F".into(),
                    value: ABop {
                        operation: ABo::Add,
                        left: bx(AId("i".into())),
                        right: bx(AId("j".into())),
                    },
                },
                // A = 5
                ElAssign {
                    name: "A".into(),
                    value: ANum(5.0),
                },
            ]),
            (
                vec![
                    // C = B for i=[1...5]
                    Assignment {
                        id: 3,
                        name: "C".into(),
                        value: Expression::For {
                            body: Body {
                                assignments: vec![
                                    // B = i^2
                                    Assignment {
                                        id: 2,
                                        name: "B".into(),
                                        value: Expression::BinaryOperation {
                                            operation: ABo::Pow,
                                            left: bx(Expression::Identifier(1)),
                                            right: bx(Expression::Number(2.0)),
                                        },
                                    },
                                ],
                                // B
                                value: bx(Expression::Identifier(2)),
                            },
                            lists: vec![
                                // i=[1...5]
                                Assignment {
                                    id: 1,
                                    name: "i".into(),
                                    value: Expression::ListRange {
                                        before_ellipsis: vec![Expression::Number(1.0)],
                                        after_ellipsis: vec![Expression::Number(5.0)],
                                    },
                                },
                            ],
                        },
                    },
                    // A = 5
                    Assignment {
                        id: 6,
                        name: "A".into(),
                        value: Expression::Number(5.0),
                    },
                    // E = C[i] + D[i] for j=[1...4]
                    Assignment {
                        id: 9,
                        name: "E".into(),
                        value: Expression::For {
                            body: Body {
                                assignments: vec![
                                    // D = B + A + F for i=[1...3]
                                    Assignment {
                                        id: 8,
                                        name: "D".into(),
                                        value: Expression::For {
                                            body: Body {
                                                assignments: vec![
                                                    // B = i^2
                                                    Assignment {
                                                        id: 5,
                                                        name: "B".into(),
                                                        value: Expression::BinaryOperation {
                                                            operation: ABo::Pow,
                                                            left: bx(Expression::Identifier(4)),
                                                            right: bx(Expression::Number(2.0)),
                                                        },
                                                    },
                                                    // F = i + j
                                                    Assignment {
                                                        id: 7,
                                                        name: "F".into(),
                                                        value: Expression::BinaryOperation {
                                                            operation: ABo::Add,
                                                            left: bx(Expression::Identifier(4)),
                                                            right: bx(Expression::Identifier(0)),
                                                        },
                                                    },
                                                ],
                                                // B + A + F
                                                value: bx(Expression::BinaryOperation {
                                                    operation: ABo::Add,
                                                    left: bx(Expression::BinaryOperation {
                                                        operation: ABo::Add,
                                                        left: bx(Expression::Identifier(5)),
                                                        right: bx(Expression::Identifier(6)),
                                                    }),
                                                    right: bx(Expression::Identifier(7)),
                                                }),
                                            },
                                            lists: vec![
                                                // i=[1...3]
                                                Assignment {
                                                    id: 4,
                                                    name: "i".into(),
                                                    value: Expression::ListRange {
                                                        before_ellipsis: vec![Expression::Number(
                                                            1.0,
                                                        )],
                                                        after_ellipsis: vec![Expression::Number(
                                                            3.0,
                                                        )],
                                                    },
                                                },
                                            ],
                                        },
                                    },
                                ],
                                // C[i] + D[i]
                                value: bx(Expression::BinaryOperation {
                                    operation: ABo::Add,
                                    left: bx(Expression::BinaryOperation {
                                        operation: ABo::Index,
                                        left: bx(Expression::Identifier(3)),
                                        right: bx(Expression::Number(1.0)),
                                    }),
                                    right: bx(Expression::BinaryOperation {
                                        operation: ABo::Index,
                                        left: bx(Expression::Identifier(8)),
                                        right: bx(Expression::Number(1.0)),
                                    }),
                                }),
                            },
                            lists: vec![
                                // j=[1...4]
                                Assignment {
                                    id: 0,
                                    name: "j".into(),
                                    value: Expression::ListRange {
                                        before_ellipsis: vec![Expression::Number(1.0)],
                                        after_ellipsis: vec![Expression::Number(4.0)],
                                    },
                                },
                            ],
                        },
                    },
                ],
                vec![
                    Some(Ok(9)),
                    Some(Ok(3)),
                    Some(Err("'j' is not defined".into())),
                    Some(Err("'i' is not defined".into())),
                    Some(Err("'i' is not defined".into())),
                    Some(Ok(6)),
                ],
            ),
        );
    }

    #[test]
    fn proper_cleanup() {
        assert_eq!(
            resolve_names(&[
                ElExpr(AWith {
                    body: bx(ABop {
                        operation: ABo::Add,
                        left: bx(AId("b".into())),
                        right: bx(AId("c".into())),
                    }),
                    substitutions: vec![("a".into(), ANum(1.0))],
                }),
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into())
                }
            ]),
            (
                vec![
                    Assignment {
                        id: 0,
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: 1,
                        name: "b".into(),
                        value: Expression::Identifier(0),
                    },
                ],
                vec![
                    Some(Err("'c' is not defined".into())),
                    Some(Err("'a' is not defined".into())),
                ],
            ),
        );
    }

    // TODO: cache errors in scope to prevent duplicate assignments being made when trying to access error variable
}
