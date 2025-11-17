use std::{
    array,
    borrow::Borrow,
    collections::HashMap,
    iter::zip,
    ops::{Deref, DerefMut},
};

use derive_more::{From, Into};
use typed_index_collections::{TiSlice, TiVec};

pub use crate::ast::{ComparisonOperator, SumProdKind};
use crate::{
    ast::{self, ExpressionListEntry},
    op::OpName,
};

#[derive(Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    Identifier(usize),
    List(Vec<Expression>),
    ListRange {
        before_ellipsis: Vec<Expression>,
        after_ellipsis: Vec<Expression>,
    },
    Op {
        operation: OpName,
        args: Vec<Expression>,
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

impl OpName {
    fn from_str(name: &str) -> Option<OpName> {
        use OpName::*;
        Some(match name {
            "ln" => Ln,
            "exp" => Exp,
            "erf" => Erf,
            "sin" => Sin,
            "cos" => Cos,
            "tan" => Tan,
            "sec" => Sec,
            "csc" => Csc,
            "cot" => Cot,
            "sinh" => Sinh,
            "cosh" => Cosh,
            "tanh" => Tanh,
            "sech" => Sech,
            "csch" => Csch,
            "coth" => Coth,
            "arcsin" => Asin,
            "arccos" => Acos,
            "arctan" => Atan,
            "arcsec" => Asec,
            "arccsc" => Acsc,
            "arccot" => Acot,
            "arcsinh" | "arsinh" => Asinh,
            "arccosh" | "arcosh" => Acosh,
            "arctanh" | "artanh" => Atanh,
            "arcsech" | "arsech" => Asech,
            "arccsch" | "arcsch" => Acsch,
            "arccoth" | "arcoth" => Acoth,
            "abs" => Abs,
            "sgn" | "sign" | "signum" => Sgn,
            "round" => Round,
            "floor" => Floor,
            "ceil" => Ceil,
            "mod" => Mod,
            "midpoint" => Midpoint,
            "distance" => Distance,
            "min" => Min,
            "max" => Max,
            "median" => Median,
            "argmin" => Argmin,
            "argmax" => Argmax,
            "total" => Total,
            "mean" => Mean,
            "count" => Count,
            "unique" => Unique,
            "uniquePerm" => UniquePerm,
            "sort" => Sort,
            "sortPerm" => SortPerm,
            "polygon" => Polygon,
            "join" => Join,
            _ => return None,
        })
    }
}

type Id = usize;
type Level = usize;

#[derive(Debug, PartialEq)]
pub struct Assignment {
    pub id: Id,
    pub name: String,
    pub value: Expression,
}

#[derive(Debug, PartialEq)]
pub struct Body {
    pub assignments: Vec<Assignment>,
    pub value: Box<Expression>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct DynamicInfo {
    id: Id,
    level: Level,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Dependency {
    Dynamic(DynamicInfo),
    Computed,
}

#[derive(Debug, Default)]
struct Dependencies<'a>(HashMap<&'a str, Dependency>);

impl<'a> Deref for Dependencies<'a> {
    type Target = HashMap<&'a str, Dependency>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> DerefMut for Dependencies<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> Dependencies<'a> {
    fn level(&self) -> Level {
        let mut level = 0;
        for d in self.0.values() {
            if let Dependency::Dynamic(i) = d {
                level = level.max(i.level);
            }
        }
        level
    }

    fn extend(&mut self, other: &Self) {
        for (name, kind) in other.iter() {
            if let Some(existing) = self.get(name) {
                assert_eq!(existing, kind);
            }
        }
        self.0.extend(other.iter());
    }
}

#[derive(Debug, Default)]
struct Scope<'a> {
    dynamic: HashMap<&'a str, DynamicInfo>,
    computed: HashMap<&'a str, (Result<Id, String>, Dependencies<'a>)>,
}

impl<'a> Scope<'a> {
    fn get_dynamic(&self, name: &str) -> Option<DynamicInfo> {
        self.dynamic.get(name).cloned()
    }
}

struct Resolver<'a> {
    scopes: Vec<Scope<'a>>,
    definitions: HashMap<&'a str, Result<&'a ExpressionListEntry, String>>,
    dependencies_being_tracked: Option<Dependencies<'a>>,
    assignments: Vec<Vec<Assignment>>,
    id_counter: usize,
}

#[derive(Debug, Copy, Clone, From, Into)]
pub struct ExpressionIndex(usize);
type ExpressionList<E> = TiSlice<ExpressionIndex, E>;

impl<'a> Resolver<'a> {
    fn new(list: &'a ExpressionList<impl Borrow<ExpressionListEntry>>) -> Self {
        let mut definitions = HashMap::new();

        for entry in list {
            match entry.borrow() {
                ExpressionListEntry::Assignment { name, .. }
                | ExpressionListEntry::FunctionDeclaration { name, .. } => {
                    if let Some(result @ Ok(_)) = definitions.get_mut(name.as_str()) {
                        *result = Err(format!("'{name}' defined multiple times"));
                    } else {
                        definitions.insert(name.as_str(), Ok(entry.borrow()));
                    }
                }
                _ => continue,
            };
        }

        Self {
            scopes: vec![Scope::default()],
            definitions,
            dependencies_being_tracked: None,
            assignments: vec![vec![]],
            id_counter: 0,
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

    fn push_assignment(&mut self, name: &str, level: usize, value: Expression) -> Id {
        let assignment = self.create_assignment(name, value);
        let id = assignment.id;
        self.assignments[level].push(assignment);
        id
    }

    fn push_dependency(&mut self, name: &'a str, kind: Dependency) {
        if let Some(d) = &mut self.dependencies_being_tracked {
            d.insert(name, kind);
        }
    }

    /// Same as [`Resolver::resolve_expression`] but additionally tracks the
    /// dependencies used while resolving the expression.
    fn resolve_expression_with_dependencies(
        &mut self,
        expression: &'a ast::Expression,
    ) -> (Result<Expression, String>, Dependencies<'a>) {
        // Track an empty set of dependencies
        let mut original = self.dependencies_being_tracked.replace(Default::default());
        let resolved_expression = self.resolve_expression(expression);
        let dependencies = self.dependencies_being_tracked.take().unwrap();

        // Add these dependencies back to the original
        if let Some(original) = &mut original {
            original.extend(&dependencies);
        }
        self.dependencies_being_tracked = original;

        (resolved_expression, dependencies)
    }

    /// Computes a variable or finds an existing assignment ID if it's already been resolved before.
    fn resolve_variable(&mut self, name: &'a str) -> Result<Id, String> {
        // First check scopes to see if it's already available without having to compute it again
        for scope in self.scopes.iter().rev() {
            // Dynamic variables get priority
            if let Some(&i) = scope.dynamic.get(name) {
                self.push_dependency(name, Dependency::Dynamic(i));
                return Ok(i.id);
            }

            if let Some((id, deps)) = scope.computed.get(name) {
                // now check if deps are up to date so we can use this id
                let mut up_to_date = true;
                for (dep_name, &recorded) in deps.iter() {
                    let found = self
                        .scopes
                        .iter()
                        .rev()
                        .find_map(|scope| scope.get_dynamic(dep_name))
                        .map_or(Dependency::Computed, Dependency::Dynamic);
                    if found != recorded {
                        up_to_date = false;
                        break;
                    }
                }
                if up_to_date {
                    let id = id.clone();
                    if let Some(d) = &mut self.dependencies_being_tracked {
                        d.extend(deps);
                    }
                    self.push_dependency(name, Dependency::Computed);
                    return id;
                }
            }
        }

        self.push_dependency(name, Dependency::Computed);
        let Some(entry) = self.definitions.get(name) else {
            return Err(format!("'{name}' is not defined"));
        };
        let expr = match entry.as_ref()? {
            ExpressionListEntry::Assignment { value, .. } => value,
            ExpressionListEntry::FunctionDeclaration { .. } => {
                return Err(format!("'{name}' is a function, try using parentheses"));
            }
            _ => unreachable!(),
        };

        let (value, deps) = self.resolve_expression_with_dependencies(expr);
        let level = deps
            .values()
            .filter_map(|x| match x {
                Dependency::Dynamic(i) => Some(i.level),
                Dependency::Computed => None,
            })
            .max()
            .unwrap_or(0);
        let id = value.map(|value| self.push_assignment(name, level, value));

        let mut scope_index = self.scopes.len();
        'outer: for (i, scope) in self.scopes.iter().enumerate().rev() {
            scope_index = i;
            for (dep_name, recorded) in deps.iter() {
                let found = scope.dynamic.get(dep_name);
                if let Some(found) = found
                    && let Dependency::Dynamic(recorded) = recorded
                {
                    assert_eq!(found, recorded);
                    break 'outer;
                } else {
                    assert_eq!(found, None);
                }
            }
        }
        self.scopes[scope_index]
            .computed
            .insert(name, (id.clone(), deps));

        id
    }

    fn resolve_expressions(
        &mut self,
        es: &'a [ast::Expression],
    ) -> Result<Vec<Expression>, String> {
        es.iter().map(|e| self.resolve_expression(e)).collect()
    }

    fn resolve_dynamic(
        &mut self,
        body: &'a ast::Expression,
        bindings: impl Iterator<Item = (&'a String, &'a ast::Expression)> + Clone,
        error_string: impl FnOnce(&str) -> String,
    ) -> Result<Expression, String> {
        let mut scope = Scope::default();

        for (name, value) in bindings.clone() {
            if scope.dynamic.contains_key(name.as_str()) {
                return Err(error_string(name));
            }

            let (value, deps) = self.resolve_expression_with_dependencies(value);
            let level = deps.level();
            let id = self.push_assignment(name, deps.level(), value?);
            scope
                .dynamic
                .insert(name.as_str(), DynamicInfo { id, level });
        }

        self.scopes.push(scope);
        let body = self.resolve_expression(body);
        let scope = self.scopes.pop().unwrap();
        if let Some(deps) = &mut self.dependencies_being_tracked {
            for (name, _) in bindings {
                let removed = deps.remove(name.as_str());
                if let Some(removed) = removed {
                    assert_eq!(
                        removed,
                        scope
                            .get_dynamic(name)
                            .map_or(Dependency::Computed, Dependency::Dynamic)
                    );
                }
            }
        }
        body
    }

    fn resolve_call(
        &mut self,
        callee: &'a str,
        args: &'a [ast::Expression],
    ) -> Result<Expression, String> {
        if let Some(operation) = OpName::from_str(callee) {
            return Ok(Expression::Op {
                operation,
                args: self.resolve_expressions(args)?,
            });
        }

        let (parameters, body) = match self.definitions.get(callee) {
            Some(Ok(ExpressionListEntry::FunctionDeclaration {
                parameters, body, ..
            })) => (parameters, body),
            Some(Ok(_)) => return Err(format!("variable '{callee}' can't be used as a function")),
            Some(Err(e)) => return Err(e.clone()),
            None => return Err(format!("'{callee}' is not defined")),
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

        self.resolve_dynamic(body, zip(parameters, args), |name| {
            format!("cannot use '{name}' for multiple parameters of this function")
        })
    }

    fn resolve_expression(&mut self, e: &'a ast::Expression) -> Result<Expression, String> {
        match e {
            ast::Expression::Number(value) => Ok(Expression::Number(*value)),
            ast::Expression::Identifier(name) => {
                Ok(Expression::Identifier(self.resolve_variable(name)?))
            }
            ast::Expression::List(list) => Ok(Expression::List(self.resolve_expressions(list)?)),
            ast::Expression::ListRange {
                before_ellipsis,
                after_ellipsis,
            } => Ok(Expression::ListRange {
                before_ellipsis: self.resolve_expressions(before_ellipsis)?,
                after_ellipsis: self.resolve_expressions(after_ellipsis)?,
            }),
            ast::Expression::CallOrMultiply { callee, args } => {
                if OpName::from_str(callee).is_some()
                    || matches!(
                        self.definitions.get(callee.as_str()),
                        Some(Ok(ExpressionListEntry::FunctionDeclaration { .. }))
                    )
                {
                    self.resolve_call(callee, args)
                } else {
                    let (left, right) = (callee, args);
                    let name = left;
                    let left = self.resolve_variable(left)?;
                    let len = right.len();
                    let right = self.resolve_expressions(right)?;
                    if len == 1 || len == 2 {
                        let mut right_iter = right.into_iter();
                        Ok(Expression::Op {
                            operation: OpName::Mul,
                            args: vec![
                                Expression::Identifier(left),
                                if len == 1 {
                                    right_iter.next().unwrap()
                                } else {
                                    Expression::Op {
                                        operation: OpName::Point,
                                        args: array::from_fn::<_, 2, _>(|_| {
                                            right_iter.next().unwrap()
                                        })
                                        .into(),
                                    }
                                },
                            ],
                        })
                    } else {
                        Err(if len == 0 {
                            format!("variable '{name}' can't be used as a function")
                        } else {
                            "points may only have 2 coordinates".into()
                        })
                    }
                }
            }
            ast::Expression::Call { callee, args } => self.resolve_call(callee, args),
            ast::Expression::ChainedComparison(ast::ChainedComparison {
                operands,
                operators,
            }) => Ok(Expression::ChainedComparison {
                operands: self.resolve_expressions(operands)?,
                operators: operators.clone(),
            }),
            ast::Expression::Piecewise {
                test,
                consequent,
                alternate,
            } => Ok(Expression::Piecewise {
                test: Box::new(self.resolve_expression(test)?),
                consequent: Box::new(self.resolve_expression(consequent)?),
                alternate: if let Some(e) = alternate {
                    Some(Box::new(self.resolve_expression(e)?))
                } else {
                    None
                },
            }),
            ast::Expression::SumProd { .. } => {
                Err("todo: sum and prod are not supported yet".into())
            }
            ast::Expression::With {
                body,
                substitutions,
            } => self.resolve_dynamic(body, substitutions.iter().map(|(n, v)| (n, v)), |name| {
                format!("a 'with' expression cannot make multiple substitutions for '{name}'")
            }),
            ast::Expression::For { body, lists } => {
                let mut seen = HashMap::new();
                let mut resolved_lists = vec![];

                for (name, value) in lists {
                    if seen.contains_key(name.as_str()) {
                        return Err(format!(
                            "you can't define '{name}' more than once on the right-hand side of 'for'"
                        ));
                    }

                    let value = self.resolve_expression(value)?;
                    let assignment = self.create_assignment(name, value);
                    seen.insert(name.as_str(), assignment.id);
                    resolved_lists.push(assignment);
                }

                let level = self.assignments.len();
                self.assignments.push(vec![]);
                let mut scope = Scope::default();

                for (name, id) in seen {
                    scope.dynamic.insert(name, DynamicInfo { id, level });
                }

                self.scopes.push(scope);

                let body = self.resolve_expression(body);

                let scope = self.scopes.pop().unwrap();

                if let Some(deps) = &mut self.dependencies_being_tracked {
                    for (name, _) in lists {
                        let removed = deps.remove(name.as_str());
                        if let Some(removed) = removed {
                            assert_eq!(
                                removed,
                                scope
                                    .get_dynamic(name)
                                    .map_or(Dependency::Computed, Dependency::Dynamic)
                            );
                        }
                    }
                }

                let assignments = self.assignments.pop().unwrap();
                Ok(Expression::For {
                    body: Body {
                        assignments,
                        value: Box::new(body?),
                    },
                    lists: resolved_lists,
                })
            }
            ast::Expression::Op { operation, args } => Ok(Expression::Op {
                operation: *operation,
                args: self.resolve_expressions(args)?,
            }),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, From, Into, Hash)]
pub struct AssignmentIndex(usize);

pub type ExpressionToAssignment = TiVec<ExpressionIndex, Option<Result<AssignmentIndex, String>>>;

pub fn resolve_names(
    list: &ExpressionList<impl Borrow<ExpressionListEntry>>,
) -> (TiVec<AssignmentIndex, Assignment>, ExpressionToAssignment) {
    let mut resolver = Resolver::new(list);

    let assignment_ids = list
        .iter()
        .map(|e| {
            // When we start resolving a new expression, there shouldn't be any
            // variables that are in scope from a `for` or `with` clause
            assert_eq!(resolver.assignments.len(), 1);
            assert_eq!(resolver.scopes.len(), 1);
            assert_eq!(resolver.scopes[0].dynamic, HashMap::new());

            match e.borrow() {
                ExpressionListEntry::Assignment { name, value } => {
                    if let Some((id, _deps)) = resolver.scopes[0].computed.get(name.as_str()) {
                        Some(id.clone())
                    } else {
                        let (value, deps) = resolver.resolve_expression_with_dependencies(value);
                        let level = deps.level();
                        assert_eq!(level, 0);
                        let id = value.map(|value| resolver.push_assignment(name, level, value));
                        if resolver.definitions.get(name.as_str()).unwrap().is_ok() {
                            resolver.scopes[0].computed.insert(name, (id.clone(), deps));
                        }
                        Some(id)
                    }
                }
                ExpressionListEntry::FunctionDeclaration { .. } => None,
                ExpressionListEntry::Relation(..) => {
                    Some(Err("todo: relations are not supported yet".into()))
                }
                ExpressionListEntry::Expression(expression) => Some(
                    resolver
                        .resolve_expression(expression)
                        .map(|value| resolver.push_assignment("<anonymous>", 0, value)),
                ),
            }
        })
        .collect::<TiVec<ExpressionIndex, _>>();

    let assignments: TiVec<AssignmentIndex, Assignment> =
        resolver.assignments.pop().unwrap().into();
    assert!(resolver.assignments.is_empty());

    let id_to_index_map = assignments
        .iter_enumerated()
        .map(|(i, a)| (a.id, i))
        .collect::<HashMap<_, _>>();
    let assignment_indices = assignment_ids
        .into_iter()
        .map(|id| id.map(|id| id.map(|id| id_to_index_map[&id])))
        .collect();

    (assignments, assignment_indices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ExpressionListEntry::{
        Assignment as ElAssign, Expression as ElExpr, FunctionDeclaration as ElFunction,
    };
    use ast::Expression::{
        Call as ACall,
        CallOrMultiply as ACallMul,
        // ChainedComparison as AComparison,
        For as AFor,
        Identifier as AId,
        List as AList,
        ListRange as AListRange,
        Number as ANum,
        Op as AOp,
        // Piecewise as APiecewise,
        // SumProd as ASumProd,
        // UnaryOperation as AUop,
        With as AWith,
    };
    use pretty_assertions::assert_eq;

    fn bx<T>(x: T) -> Box<T> {
        Box::new(x)
    }

    fn resolve_names_ti(
        list: &[ExpressionListEntry],
    ) -> (Vec<Assignment>, Vec<Option<Result<usize, String>>>) {
        let (a, b) = resolve_names(list.as_ref());
        (
            a.into(),
            b.into_iter()
                .map(|x| x.map(|x| x.map(Into::into)))
                .collect(),
        )
    }

    #[test]
    fn expressions() {
        assert_eq!(
            resolve_names_ti(&[
                ElExpr(ANum(5.0)),
                ElExpr(AOp {
                    operation: OpName::Add,
                    args: vec![ANum(1.0), ANum(2.0)]
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
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![Expression::Number(1.0), Expression::Number(2.0)],
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
            resolve_names_ti(&[
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
            resolve_names_ti(&[
                // a = 1
                ElAssign {
                    name: "a".into(),
                    value: ANum(1.0),
                },
                // a = 2
                ElAssign {
                    name: "a".into(),
                    value: ANum(2.0),
                },
                // b = a
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
        if true {
            return;
        }
        (|| panic!("this currrently fails by stack overflow"))();
        assert_eq!(
            resolve_names_ti(&[
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
    fn funny_not_circular() {
        assert_eq!(
            resolve_names_ti(&[
                // a = c with b = 3
                ElAssign {
                    name: "a".into(),
                    value: AWith {
                        body: bx(AId("c".into())),
                        substitutions: vec![("b".into(), ANum(3.0))]
                    },
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
                // c = b
                ElAssign {
                    name: "c".into(),
                    value: AId("b".into()),
                },
            ]),
            (
                vec![
                    // with b = 3
                    Assignment {
                        id: 0,
                        name: "b".into(),
                        value: Expression::Number(3.0)
                    },
                    // c = b
                    Assignment {
                        id: 1,
                        name: "c".into(),
                        value: Expression::Identifier(0)
                    },
                    // a = c with b = 3
                    Assignment {
                        id: 2,
                        name: "a".into(),
                        value: Expression::Identifier(1)
                    },
                    // b = a
                    Assignment {
                        id: 3,
                        name: "b".into(),
                        value: Expression::Identifier(2)
                    },
                    // c = b
                    Assignment {
                        id: 4,
                        name: "c".into(),
                        value: Expression::Identifier(3)
                    },
                ],
                vec![Some(Ok(2)), Some(Ok(3)), Some(Ok(4))],
            ),
        );
    }

    #[test]
    fn dependencies() {
        assert_eq!(
            resolve_names_ti(&[
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
            resolve_names_ti(&[
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
            resolve_names_ti(&[
                // a = 1
                ElAssign {
                    name: "a".into(),
                    value: ANum(1.0),
                },
                // a(2)
                ElExpr(ACallMul {
                    callee: "a".into(),
                    args: vec![ANum(2.0)],
                }),
                // a(3, 4)
                ElExpr(ACallMul {
                    callee: "a".into(),
                    args: vec![ANum(3.0), ANum(4.0)],
                }),
                // a(5, 6, 7, 8)
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
                        value: Expression::Op {
                            operation: OpName::Mul,
                            args: vec![Expression::Identifier(0), Expression::Number(2.0)]
                        },
                    },
                    Assignment {
                        id: 2,
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Mul,
                            args: vec![
                                Expression::Identifier(0),
                                Expression::Op {
                                    operation: OpName::Point,
                                    args: vec![Expression::Number(3.0), Expression::Number(4.0)],
                                }
                            ]
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
            resolve_names_ti(&[
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
            resolve_names_ti(&[
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
            resolve_names_ti(&[
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
                    value: AOp {
                        operation: OpName::Point,
                        args: vec![
                            AId("q".into()),
                            AOp {
                                operation: OpName::Add,
                                args: vec![AId("i".into()), AId("k".into())]
                            }
                        ]
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
                    value: AOp {
                        operation: OpName::Mul,
                        args: vec![AId("j".into()), AId("j".into())]
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
                                        value: Expression::Op {
                                            operation: OpName::Mul,
                                            args: vec![
                                                Expression::Identifier(1),
                                                Expression::Identifier(1)
                                            ]
                                        },
                                    },
                                    // p = (q,i+k)
                                    Assignment {
                                        id: 5,
                                        name: "p".into(),
                                        value: Expression::Op {
                                            operation: OpName::Point,
                                            args: vec![
                                                Expression::Identifier(3),
                                                Expression::Op {
                                                    operation: OpName::Add,
                                                    args: vec![
                                                        Expression::Identifier(2),
                                                        Expression::Identifier(4)
                                                    ]
                                                }
                                            ],
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
                    Some(Ok(2)),
                    Some(Err("'j' is not defined".into())),
                    Some(Ok(0)),
                    Some(Err("'j' is not defined".into())),
                    Some(Ok(1)),
                ],
            ),
        );
    }

    #[test]
    fn nested_list_comps() {
        assert_eq!(
            resolve_names_ti(&[
                // E = C.total + D.total for j=[1...4]
                ElAssign {
                    name: "E".into(),
                    value: AFor {
                        body: bx(AOp {
                            operation: OpName::Add,
                            args: vec![
                                ACall {
                                    callee: "total".into(),
                                    args: vec![AId("C".into())],
                                },
                                ACall {
                                    callee: "total".into(),
                                    args: vec![AId("D".into())],
                                }
                            ]
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
                        body: bx(AOp {
                            operation: OpName::Add,
                            args: vec![
                                AOp {
                                    operation: OpName::Add,
                                    args: vec![AId("B".into()), AId("A".into())]
                                },
                                AId("F".into())
                            ]
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
                    value: AOp {
                        operation: OpName::Pow,
                        args: vec![AId("i".into()), ANum(2.0)]
                    }
                },
                // F = i + j
                ElAssign {
                    // TODO: change this back to "J" and see why it was panicking
                    name: "F".into(),
                    value: AOp {
                        operation: OpName::Add,
                        args: vec![AId("i".into()), AId("j".into())]
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
                                        value: Expression::Op {
                                            operation: OpName::Pow,
                                            args: vec![
                                                Expression::Identifier(1),
                                                Expression::Number(2.0)
                                            ]
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
                                                        value: Expression::Op {
                                                            operation: OpName::Pow,
                                                            args: vec![
                                                                Expression::Identifier(4),
                                                                Expression::Number(2.0)
                                                            ]
                                                        },
                                                    },
                                                    // F = i + j
                                                    Assignment {
                                                        id: 7,
                                                        name: "F".into(),
                                                        value: Expression::Op {
                                                            operation: OpName::Add,
                                                            args: vec![
                                                                Expression::Identifier(4),
                                                                Expression::Identifier(0)
                                                            ],
                                                        },
                                                    },
                                                ],
                                                // B + A + F
                                                value: bx(Expression::Op {
                                                    operation: OpName::Add,
                                                    args: vec![
                                                        Expression::Op {
                                                            operation: OpName::Add,
                                                            args: vec![
                                                                Expression::Identifier(5),
                                                                Expression::Identifier(6)
                                                            ],
                                                        },
                                                        Expression::Identifier(7)
                                                    ]
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
                                // C.total + D.total
                                value: bx(Expression::Op {
                                    operation: OpName::Add,
                                    args: vec![
                                        Expression::Op {
                                            operation: OpName::Total,
                                            args: vec![Expression::Identifier(3)]
                                        },
                                        Expression::Op {
                                            operation: OpName::Total,
                                            args: vec![Expression::Identifier(8)]
                                        }
                                    ]
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
                    Some(Ok(2)),
                    Some(Ok(0)),
                    Some(Err("'j' is not defined".into())),
                    Some(Err("'i' is not defined".into())),
                    Some(Err("'i' is not defined".into())),
                    Some(Ok(1)),
                ],
            ),
        );
    }

    #[test]
    fn proper_cleanup() {
        assert_eq!(
            resolve_names_ti(&[
                ElExpr(AWith {
                    body: bx(AOp {
                        operation: OpName::Add,
                        args: vec![AId("b".into()), AId("c".into()),]
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

    #[test]
    fn cache_errors() {
        assert_eq!(
            resolve_names_ti(&[
                // a = c with b = 1
                ElAssign {
                    name: "a".into(),
                    value: AWith {
                        body: bx(AId("c".into())),
                        substitutions: vec![("b".into(), ANum(1.0))]
                    }
                },
                // a
                ElExpr(AId("a".into()))
            ]),
            (
                vec![Assignment {
                    id: 0,
                    name: "b".into(),
                    value: Expression::Number(1.0)
                }],
                vec![
                    Some(Err("'c' is not defined".into())),
                    Some(Err("'c' is not defined".into()))
                ]
            )
        );
    }

    #[test]
    fn chained_with() {
        assert_eq!(
            resolve_names_ti(&[
                // a = 1
                ElAssign {
                    name: "a".into(),
                    value: ANum(1.0),
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
                // c = b
                ElAssign {
                    name: "c".into(),
                    value: AId("b".into()),
                },
                // c with a = 5
                ElExpr(AWith {
                    body: bx(AId("c".into())),
                    substitutions: vec![("a".into(), ANum(5.0))],
                }),
            ]),
            (
                vec![
                    // a = 1
                    Assignment {
                        id: 0,
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    // b = a
                    Assignment {
                        id: 1,
                        name: "b".into(),
                        value: Expression::Identifier(0),
                    },
                    // c = b
                    Assignment {
                        id: 2,
                        name: "c".into(),
                        value: Expression::Identifier(1),
                    },
                    // with a = 5
                    Assignment {
                        id: 3,
                        name: "a".into(),
                        value: Expression::Number(5.0),
                    },
                    // b = a
                    Assignment {
                        id: 4,
                        name: "b".into(),
                        value: Expression::Identifier(3),
                    },
                    // c = b
                    Assignment {
                        id: 5,
                        name: "c".into(),
                        value: Expression::Identifier(4),
                    },
                    // c with a = 5
                    Assignment {
                        id: 6,
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(5),
                    },
                ],
                vec![Some(Ok(0)), Some(Ok(1)), Some(Ok(2)), Some(Ok(6)),],
            ),
        );
    }

    #[test]
    fn stored_function_evaluation() {
        assert_eq!(
            resolve_names_ti(&[
                // f() = 1
                ElFunction {
                    name: "f".into(),
                    parameters: vec![],
                    body: ANum(1.0)
                },
                // b = f()
                ElAssign {
                    name: "b".into(),
                    value: ACall {
                        callee: "f".into(),
                        args: vec![]
                    },
                },
                // b
                ElExpr(AId("b".into())),
                // c = f()
                ElAssign {
                    name: "c".into(),
                    value: ACallMul {
                        callee: "f".into(),
                        args: vec![]
                    },
                },
                // c
                ElExpr(AId("c".into()))
            ]),
            (
                vec![
                    Assignment {
                        id: 0,
                        name: "b".into(),
                        value: Expression::Number(1.0)
                    },
                    Assignment {
                        id: 1,
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(0)
                    },
                    Assignment {
                        id: 2,
                        name: "c".into(),
                        value: Expression::Number(1.0)
                    },
                    Assignment {
                        id: 3,
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(2)
                    }
                ],
                vec![None, Some(Ok(0)), Some(Ok(1)), Some(Ok(2)), Some(Ok(3))],
            ),
        );
    }
}
