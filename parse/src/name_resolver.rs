use std::{
    array,
    borrow::Borrow,
    collections::{HashMap, HashSet},
    fmt::Display,
    iter::zip,
    ops::{Deref, DerefMut},
    sync::OnceLock,
};

use derive_more::{From, Into};
use typed_index_collections::{TiSlice, TiVec, ti_vec};

pub use crate::ast::{ComparisonOperator, SumProdKind};
use crate::{
    ast::{self, Statement},
    op::OpName,
};

#[derive(Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    Identifier(Id),
    Slider {
        value: Box<Expression>,
        slider: Slider<Box<Expression>>,
    },
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
        variable: Id,
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
            "count" | "length" => Count,
            "repeat" => Repeat,
            "unique" => Unique,
            "uniquePerm" => UniquePerm,
            "sort" => Sort,
            "sortPerm" => SortPerm,
            "polygon" => Polygon,
            "vertices" => Vertices,
            "join" => Join,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Id(pub usize);

#[derive(Debug, Copy, Clone, From, Into, PartialEq)]
struct Level(usize);

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
struct SubstitutionInfo {
    id: Id,
    level: Level,
    kind: ScopeKind,
    scope_index: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Dependency {
    Substitution(SubstitutionInfo),
    Computed,
}

impl Dependency {
    fn is_lexical(&self) -> bool {
        matches!(
            self,
            Dependency::Substitution(SubstitutionInfo {
                kind: ScopeKind::Lexical { .. },
                ..
            })
        )
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
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
            if let Dependency::Substitution(i) = d {
                level = level.max(i.level.into());
            }
        }
        level.into()
    }

    fn scope_index(&self) -> usize {
        let mut scope_index = 0;
        for d in self.0.values() {
            if let Dependency::Substitution(i) = d {
                scope_index = scope_index.max(i.scope_index);
            }
        }
        scope_index
    }

    fn extend(&mut self, other: &Self) {
        for (name, kind) in other.iter() {
            if let Some(existing) = self.get(name) {
                if !existing.is_lexical() && kind.is_lexical() {
                    continue;
                }
                if !existing.is_lexical() || kind.is_lexical() {
                    assert_eq!(existing, kind);
                }
            }
            self.insert(name, *kind);
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum ScopeKind {
    Lexical { line_count: usize }, // function parameters, sum/prod index
    Dynamic,                       // with/for variables
}

#[derive(Debug)]
struct Scope<'a> {
    kind: ScopeKind,
    substitutions: HashMap<&'a str, SubstitutionInfo>,
    computed: HashMap<
        &'a str,
        (
            Result<Id, NameError>,
            Option<Slider<(Result<Id, NameError>, Dependencies<'a>)>>,
            Dependencies<'a>,
        ),
    >,
}

#[derive(Debug, Default)]
struct CycleDetector<'a> {
    stack: Vec<&'a str>,
    // We use a counter per name instead of something simpler like `seen: HashSet<&str>`
    // so that we can still allow examples like `c = b; b = a; a = c with b = 3` to work
    // (see name_resolver::tests::funny_not_circular_reversed)
    // TODO is there a cleaner way to allow that test to pass?
    counts: HashMap<&'a str, usize>,
}

impl<'a> CycleDetector<'a> {
    fn push(&mut self, name: &'a str) -> Result<(), NameError> {
        let count = self.counts.entry(name).or_insert(0);
        if *count == 2 {
            let start = self.stack.iter().rposition(|&n| n == name).unwrap();
            let names = self.stack[start..].iter().cloned();
            return Err(NameError::cyclic_definition(names));
        }
        *count += 1;
        self.stack.push(name);
        Ok(())
    }

    fn pop(&mut self) {
        let name = self.stack.pop().unwrap();
        *self.counts.get_mut(name).unwrap() -= 1;
    }
}

struct Resolver<'a> {
    use_v1_9_scoping_rules: bool,
    scopes: Vec<Scope<'a>>,
    line_count: usize,
    definitions:
        HashMap<&'a str, Result<(&'a Statement, Option<Slider<&'a ast::Expression>>), NameError>>,
    dependencies_being_tracked: Option<Dependencies<'a>>,
    assignments: TiVec<Level, Vec<Assignment>>,
    freevars: HashMap<&'a str, Id>,
    id_counter: usize,
    cycle_detector: CycleDetector<'a>,
}

#[derive(Debug, Copy, Clone, From, Into, PartialEq, Eq, Hash)]
pub struct ExpressionIndex(usize);

impl<'a> Resolver<'a> {
    fn new(
        list: impl Iterator<Item = (&'a Statement, Option<Slider<&'a ast::Expression>>)>,
        undefinable_names: &HashSet<&str>,
        use_v1_9_scoping_rules: bool,
    ) -> Self {
        let mut definitions = HashMap::new();

        for (statement, slider) in list {
            match statement {
                Statement::Assignment { name, .. }
                | Statement::FunctionDeclaration { name, .. }
                    if !undefinable_names.contains(name.as_str()) =>
                {
                    if let Some(result) = definitions.get_mut(name.as_str()) {
                        *result = Err(NameError::MultipleDefinitions(name.into()));
                    } else {
                        definitions.insert(name.as_str(), Ok((statement, slider)));
                    }
                }
                _ => continue,
            };
        }

        Self {
            use_v1_9_scoping_rules,
            scopes: vec![Scope {
                kind: ScopeKind::Dynamic,
                substitutions: HashMap::new(),
                computed: HashMap::new(),
            }],
            line_count: 0,
            definitions,
            dependencies_being_tracked: None,
            assignments: ti_vec![vec![]],
            freevars: HashMap::new(),
            id_counter: 0,
            cycle_detector: CycleDetector::default(),
        }
    }

    fn next_id(&mut self) -> Id {
        let id = Id(self.id_counter);
        self.id_counter += 1;
        id
    }

    fn create_new_freevar(&mut self, name: &'a str) -> Id {
        let id = self.next_id();
        let existing = self.freevars.insert(name, id);
        assert_eq!(existing, None);
        id
    }

    fn create_assignment(&mut self, name: &str, value: Expression) -> Assignment {
        Assignment {
            id: self.next_id(),
            name: name.to_string(),
            value,
        }
    }

    fn push_assignment(&mut self, name: &str, level: Level, value: Expression) -> Id {
        let assignment = self.create_assignment(name, value);
        let id = assignment.id;
        self.assignments[level].push(assignment);
        id
    }

    fn push_dependency(&mut self, name: &'a str, kind: Dependency) {
        if let Some(d) = &mut self.dependencies_being_tracked {
            if let Some(existing) = d.get(name) {
                if !existing.is_lexical() && kind.is_lexical() {
                    return;
                }
                if !existing.is_lexical() || kind.is_lexical() {
                    assert_eq!(*existing, kind);
                }
            }
            d.insert(name, kind);
        }
    }

    fn resolve_with_dependencies<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> T,
        substitutions: Option<(ScopeKind, HashMap<&'a str, SubstitutionInfo>)>,
    ) -> (T, Dependencies<'a>) {
        let using_scope = substitutions.is_some();
        if let Some((kind, substitutions)) = substitutions {
            self.scopes.push(Scope {
                kind,
                substitutions,
                computed: HashMap::new(),
            });
        }

        // Track an empty set of dependencies
        let mut original = self.dependencies_being_tracked.replace(Default::default());
        let value = f(self);
        let mut dependencies = self.dependencies_being_tracked.take().unwrap();

        if using_scope {
            // Untrack dependencies on variables in the scope
            for (name, info) in self.scopes.pop().unwrap().substitutions {
                // The hoopla below is to deal with cases like the `function_transitive_dependency` test,
                // where `f`'s body depends on the computed global value of `a` (transitively via `c`)
                // while itself defining a more recent lexical substitution for `a`
                if let Some(existing) = dependencies.get(name)
                    && (info.kind == ScopeKind::Dynamic || existing.is_lexical())
                {
                    assert_eq!(*existing, Dependency::Substitution(info));
                    dependencies.remove(name);
                }
            }
        }

        // Add these dependencies back to the original
        if let Some(original) = &mut original {
            original.extend(&dependencies);
        }
        self.dependencies_being_tracked = original;

        (value, dependencies)
    }

    /// Same as [`Resolver::resolve_expression`] but additionally tracks the
    /// dependencies used and optionally uses a substitution scope while resolving
    /// the expression.
    fn resolve_expression_with_dependencies(
        &mut self,
        expression: &'a ast::Expression,
        substitutions: Option<(ScopeKind, HashMap<&'a str, SubstitutionInfo>)>,
    ) -> (Result<Expression, NameError>, Dependencies<'a>) {
        self.resolve_with_dependencies(|this| this.resolve_expression(expression), substitutions)
    }

    /// Finds the most recent substitution for a variable if it exists, with the
    /// choice to include lexically scoped substitutions in the search or not.
    fn find_substitution(&self, name: &'a str, include_lexical: bool) -> Option<SubstitutionInfo> {
        // Search dynamic scopes, also including the current line's lexical scope
        for scope in self.scopes.iter().rev() {
            let line_count = self.line_count;
            if (scope.kind == ScopeKind::Dynamic
                || include_lexical && scope.kind == ScopeKind::Lexical { line_count })
                && let Some(&i) = scope.substitutions.get(name)
            {
                return Some(i);
            }
        }

        if include_lexical {
            if self.definitions.contains_key(name) {
                // Global definitions are preferred over lexical substitutions if
                // the lexical substitution isn't in the current line
                return None;
            }

            for scope in self.scopes.iter().rev() {
                if let ScopeKind::Lexical { .. } = scope.kind
                    && let Some(&i) = scope.substitutions.get(name)
                {
                    return Some(i);
                }
            }
        }

        None
    }

    /// Resolves a variable with an optional slider definition.
    fn resolve_value_slider(
        &mut self,
        name: &'a str,
        value: &'a ast::Expression,
        slider: Option<Slider<&'a ast::Expression>>,
    ) -> (
        Result<Id, NameError>,
        Option<Slider<(Result<Id, NameError>, Dependencies<'a>)>>,
        Dependencies<'a>,
    ) {
        let ((value, slider), deps) = self.resolve_with_dependencies(
            |this| {
                let value = this.resolve_expression(value);
                let slider =
                    slider.map(|s| s.map(|e| this.resolve_expression_with_dependencies(e, None)));
                (value, slider)
            },
            None,
        );
        let level = deps.level();

        let Some(slider) = slider else {
            let id = value.map(|value| self.push_assignment(name, level, value));
            return (id, None, deps);
        };

        let mut slider_error = None;
        let mut f = |name, x: Option<(Result<_, NameError>, _)>| {
            x.map(|x| {
                (
                    x.0.map(|x| self.push_assignment(name, level, x))
                        .inspect_err(|e| slider_error = Some(e.clone())),
                    x.1,
                )
            })
        };
        let slider_id = Slider {
            min: f("<slider min>", slider.min),
            max: f("<slider max>", slider.max),
            step: f("<slider step>", slider.step),
        };

        let value_id = match slider_error {
            Some(e) => Err(e),
            None => {
                let value =
                    value.expect("slider latex should have just been number literal so no error");
                let value = Expression::Slider {
                    value: Box::new(value),
                    slider: slider_id
                        .clone()
                        .map(|id| Box::new(Expression::Identifier(id.0.expect("slider_error")))),
                };
                Ok(self.push_assignment(name, level, value))
            }
        };

        (value_id, Some(slider_id), deps)
    }

    /// Computes a variable or finds an existing assignment ID if it's already been resolved before.
    fn resolve_variable(&mut self, name: &'a str) -> Result<Id, NameError> {
        if let Some(i) = self.find_substitution(name, true) {
            self.push_dependency(name, Dependency::Substitution(i));
            return Ok(i.id);
        }

        // It wasn't available as a substitution so that means we'll depend on it as a computed variable
        self.push_dependency(name, Dependency::Computed);

        // Check scopes to see if it's already available without having to compute it again
        for scope in self.scopes.iter().rev() {
            if let Some((id, _, deps)) = scope.computed.get(name) {
                // now check if deps are up to date so we can use this id
                let mut up_to_date = true;
                for (dep_name, &recorded) in deps.iter() {
                    let depends_on_it_being_undefined = recorded == Dependency::Computed
                        && !self.definitions.contains_key(dep_name);
                    let found = self
                        .find_substitution(
                            dep_name,
                            depends_on_it_being_undefined || recorded.is_lexical(),
                        )
                        .map_or(Dependency::Computed, Dependency::Substitution);
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
                    return id;
                }
            }
        }

        // It hasn't been computed before so we'll have to compute it again
        let (id, slider, deps) = if let Some(statement) = self.definitions.get(name) {
            let (expr, slider) = match statement.as_ref().map_err(Clone::clone)? {
                (Statement::Assignment { value, .. }, slider) => (value, slider.clone()),
                (Statement::FunctionDeclaration { .. }, None) => {
                    return Err(NameError::FunctionAsVariable(name.into()));
                }
                _ => unreachable!(),
            };

            self.cycle_detector.push(name)?;
            self.line_count += 1;
            let result = self.resolve_value_slider(name, expr, slider);
            self.line_count -= 1;
            self.cycle_detector.pop();

            result
        } else {
            (
                Ok(self.create_new_freevar(name)),
                None,
                Dependencies::default(),
            )
        };

        let _existing = self.scopes[deps.scope_index()]
            .computed
            .insert(name, (id.clone(), slider, deps));
        // TODO this assert doesn't work because of our hacky cycle detector requiring two counts
        // assert_eq!(
        //     existing, None,
        //     "if it already existed then why did we just bother computing it again?"
        // );

        id
    }

    fn resolve_expressions(
        &mut self,
        es: &'a [ast::Expression],
    ) -> Result<Vec<Expression>, NameError> {
        es.iter().map(|e| self.resolve_expression(e)).collect()
    }

    fn resolve_substitutions(
        &mut self,
        body: &'a ast::Expression,
        is_lexical: bool,
        bindings: impl Iterator<Item = (&'a String, &'a ast::Expression)>,
        error: impl FnOnce(String) -> NameError,
    ) -> Result<Expression, NameError> {
        let mut substitutions = HashMap::new();
        let kind = if is_lexical {
            ScopeKind::Lexical {
                line_count: self.line_count + 1,
            }
        } else {
            ScopeKind::Dynamic
        };

        for (name, value) in bindings {
            if substitutions.contains_key(name.as_str()) {
                return Err(error(name.into()));
            }

            let (value, deps) = self.resolve_expression_with_dependencies(value, None);
            let level = deps.level();
            let id = self.push_assignment(name, level, value?);
            substitutions.insert(
                name.as_str(),
                SubstitutionInfo {
                    id,
                    level,
                    kind,
                    scope_index: self.scopes.len(),
                },
            );
        }

        if is_lexical {
            self.line_count += 1;
        }

        let (body, _) =
            self.resolve_expression_with_dependencies(body, Some((kind, substitutions)));

        if is_lexical {
            self.line_count -= 1;
        }

        body
    }

    fn resolve_call(
        &mut self,
        callee: &'a str,
        args: &'a [ast::Expression],
    ) -> Result<Expression, NameError> {
        if let Some(operation) = OpName::from_str(callee) {
            return Ok(Expression::Op {
                operation,
                args: self.resolve_expressions(args)?,
            });
        }

        let (parameters, body) = match self.definitions.get(callee) {
            Some(Ok((
                Statement::FunctionDeclaration {
                    parameters, body, ..
                },
                _,
            ))) => (parameters, body),
            Some(Ok(_)) => return Err(NameError::VariableAsFunction(callee.into())),
            Some(Err(e)) => return Err(e.clone()),
            None => return Err(NameError::undefined([callee])),
        };

        if parameters.len() != args.len() {
            return Err(NameError::ArityMismatch {
                callee: callee.into(),
                expected: parameters.len(),
                found: args.len(),
            });
        }

        self.cycle_detector.push(callee)?;
        let value = self.resolve_substitutions(
            body,
            !self.use_v1_9_scoping_rules,
            zip(parameters, args),
            NameError::DuplicateFunctionParameter,
        );
        self.cycle_detector.pop();
        value
    }

    fn resolve_expression(&mut self, e: &'a ast::Expression) -> Result<Expression, NameError> {
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
                        Some(Ok((Statement::FunctionDeclaration { .. }, _)))
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
                            NameError::VariableAsFunction(name.into())
                        } else {
                            NameError::BadPointDimension
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
            ast::Expression::SumProd { .. } => Err(NameError::TodoSumProd),
            ast::Expression::With {
                body,
                substitutions,
            } => self.resolve_substitutions(
                body,
                false,
                substitutions.iter().map(|(n, v)| (n, v)),
                NameError::DuplicateWithSubstitution,
            ),
            ast::Expression::For { body, lists } => {
                let level = self.assignments.next_key();
                let mut substitutions = HashMap::new();
                let mut resolved_lists = vec![];

                for (name, value) in lists {
                    if substitutions.contains_key(name.as_str()) {
                        return Err(NameError::DuplicateListComprehensionInput(name.into()));
                    }

                    let value = self.resolve_expression(value)?;
                    let assignment = self.create_assignment(name, value);
                    substitutions.insert(
                        name.as_str(),
                        SubstitutionInfo {
                            id: assignment.id,
                            level,
                            kind: ScopeKind::Dynamic,
                            scope_index: self.scopes.len(),
                        },
                    );
                    resolved_lists.push(assignment);
                }

                assert_eq!(self.assignments.next_key(), level);
                self.assignments.push(vec![]);
                let (body, _) = self.resolve_expression_with_dependencies(
                    body,
                    Some((ScopeKind::Dynamic, substitutions)),
                );
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

bitflags::bitflags! {
    #[derive(Debug, PartialEq)]
    pub struct PlotKinds: u8 {
        /// `y = f(x)`
        const NORMAL = 1 << 0;
        /// `x = f(y)`
        const INVERSE = 1 << 1;
        /// `(x(t), y(t))`
        const PARAMETRIC = 1 << 2;
        /// `f(x, y) = 0`
        const IMPLICIT = 1 << 3;
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NameError {
    ArityMismatch {
        callee: String,
        expected: usize,
        found: usize,
    },
    BadPointDimension,
    CyclicDefinition(Vec<String>),
    DuplicateFunctionParameter(String),
    DuplicateListComprehensionInput(String),
    DuplicateWithSubstitution(String),
    ExpressionWithFreeVariablY,
    FunctionAsVariable(String),
    MultipleDefinitions(String),
    TodoChainedRelation,
    TodoInequality,
    TodoSumProd,
    Undefined(Vec<String>),
    VariableAsFunction(String),
}

fn sorted<S: Into<String>>(n: impl IntoIterator<Item = S>) -> Vec<String> {
    let mut n = n.into_iter().map(Into::into).collect::<Vec<_>>();
    n.sort();
    n
}

impl NameError {
    pub fn cyclic_definition<S: Into<String>>(n: impl IntoIterator<Item = S>) -> NameError {
        NameError::CyclicDefinition(sorted(n))
    }

    pub fn undefined<S: Into<String>>(n: impl IntoIterator<Item = S>) -> NameError {
        NameError::Undefined(sorted(n))
    }
}

impl Display for NameError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NameError::ArityMismatch {
                callee,
                expected,
                found,
            } => {
                write!(
                    f,
                    "function '{callee}' requires {}{}",
                    if found > expected { "only " } else { "" },
                    if *expected == 1 {
                        "1 argument".into()
                    } else {
                        format!("{} arguments", expected)
                    }
                )
            }
            NameError::BadPointDimension => write!(f, "points may only have 2 or 3 coordinates"),
            NameError::CyclicDefinition(names) => match &names[..] {
                [] => write!(f, "[internal] cyclic definition with no names"),
                [a] => write!(f, "'{a}' can't be defined in terms of itself"),
                [first @ .., last] => {
                    write!(
                        f,
                        "{} and '{last}' can't be defined in terms of each other",
                        first
                            .iter()
                            .map(|n| format!("'{n}'"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                }
            },
            NameError::DuplicateFunctionParameter(name) => write!(
                f,
                "cannot use '{name}' for multiple parameters of this function"
            ),
            NameError::DuplicateListComprehensionInput(name) => write!(
                f,
                "you can't define '{name}' more than once on the right-hand side of 'for'"
            ),
            NameError::DuplicateWithSubstitution(name) => write!(
                f,
                "a 'with' expression cannot make multiple substitutions for '{name}'"
            ),
            NameError::ExpressionWithFreeVariablY => {
                write!(f, "try adding 'x=' to the beginning of this equation")
            }
            NameError::FunctionAsVariable(name) => {
                write!(f, "'{name}' is a function, try using parentheses")
            }
            NameError::MultipleDefinitions(name) => write!(f, "'{name}' defined multiple times"),
            NameError::TodoChainedRelation => {
                write!(f, "todo: chained relations are not implemented yet")
            }
            NameError::TodoInequality => {
                write!(f, "todo: inequalities are not implemented yet")
            }
            NameError::TodoSumProd => write!(f, "todo: sum and prod are not implemented yet"),
            NameError::Undefined(names) => {
                match &names
                    .iter()
                    .filter(|&n| n != "x" && n != "y")
                    .collect::<Vec<_>>()[..]
                {
                    [] => write!(f, "[internal] nothing is undefined"),
                    [a] => write!(f, "'{a}' is not defined"),
                    [first @ .., last] => write!(
                        f,
                        "{} and '{last}' are not defined",
                        first
                            .iter()
                            .map(|n| format!("'{n}'"))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ),
                }
            }
            NameError::VariableAsFunction(name) => {
                write!(f, "variable '{name}' can't be used as a function")
            }
        }
    }
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct Domain<T> {
    pub min: T,
    pub max: T,
}

impl Domain<&'static ast::Expression> {
    pub const ZERO_TO_ONE: Self = Domain {
        min: &ast::Expression::Number(0.0),
        max: &ast::Expression::Number(1.0),
    };
}

#[derive(Debug, Clone, PartialEq)]
pub struct Slider<T> {
    pub min: Option<T>,
    pub max: Option<T>,
    pub step: Option<T>,
}

impl<T> Slider<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Slider<U> {
        Slider {
            min: self.min.map(&mut f),
            max: self.max.map(&mut f),
            step: self.step.map(&mut f),
        }
    }

    pub fn fields(&self) -> impl Iterator<Item = &T> {
        [&self.min, &self.max, &self.step].into_iter().flatten()
    }

    pub fn fields_mut(&mut self) -> impl Iterator<Item = &mut T> {
        [&mut self.min, &mut self.max, &mut self.step]
            .into_iter()
            .flatten()
    }
}

#[derive(Debug, PartialEq)]
pub struct ExpressionListEntry<'a> {
    pub expression: &'a Statement,
    pub parametric_domain: Domain<&'a ast::Expression>,
    // TODO design better types so that `slider` can only be
    // provided when `expression` is `Statement::Assignment`
    pub slider: Option<Slider<&'a ast::Expression>>,
}

#[derive(Debug, PartialEq)]
pub enum ExpressionResult {
    None,
    Err(NameError),
    Value(Id),
    // TODO `Slider` probably shouldn't be mutually exclusive to `Plot` because
    // of cases like x=1 needing to both showing a slider and plot a line
    Slider {
        /// `value` is `None` when there's an error in the slider fields
        value: Option<Id>,
        slider: Slider<Result<Id, NameError>>,
    },
    Plot {
        allowed_kinds: PlotKinds,
        value: Id,
        parameters: Vec<Id>,
        domain: Option<Domain<Result<Id, NameError>>>,
    },
}

trait ToVec {
    type T;
    fn to_vec(self) -> Vec<Self::T>;
}

impl<T> ToVec for Option<T> {
    type T = T;

    fn to_vec(self) -> Vec<Self::T> {
        match self {
            Some(x) => vec![x],
            None => vec![],
        }
    }
}

#[derive(Debug)]
pub struct Output {
    pub assignments: Vec<Assignment>,
    pub results: TiVec<ExpressionIndex, ExpressionResult>,
    pub freevars: HashMap<String, Id>,
    pub builtin_constants: HashMap<String, Id>,
}

fn resolve_relation(
    resolver: &mut Resolver,
    operands: (Result<Vec<Expression>, NameError>, Dependencies),
    operators: &[ComparisonOperator],
) -> ExpressionResult {
    let [operator] = &operators else {
        return ExpressionResult::Err(NameError::TodoChainedRelation);
    };
    if *operator != ast::ComparisonOperator::Equal {
        return ExpressionResult::Err(NameError::TodoInequality);
    }
    let value = operands.0.map(|args| Expression::Op {
        operation: OpName::Sub,
        args,
    });
    let deps = operands.1;
    let level = deps.level();
    assert_eq!(level, Level(0));
    let id = value.map(|value| resolver.push_assignment("<anonymous>", level, value));
    let mut freevars = deps
        .keys()
        .cloned()
        .filter(|name| resolver.freevars.contains_key(name))
        .collect::<Vec<_>>();
    freevars.sort();
    match id {
        Ok(id) => {
            if matches!(freevars[..], ["x"] | ["y"] | ["x", "y"]) {
                ExpressionResult::Plot {
                    allowed_kinds: PlotKinds::IMPLICIT,
                    value: id,
                    parameters: vec![
                        resolver.resolve_variable("x").unwrap(),
                        resolver.resolve_variable("y").unwrap(),
                    ],
                    domain: None,
                }
            } else if freevars.is_empty() {
                ExpressionResult::None
            } else {
                ExpressionResult::Err(NameError::undefined(freevars))
            }
        }
        Err(e) => ExpressionResult::Err(e),
    }
}

pub fn resolve_names<'a>(
    list: &TiSlice<ExpressionIndex, impl Borrow<ExpressionListEntry<'a>>>,
    builtin_constants: &[&str],
    use_v1_9_scoping_rules: bool,
) -> Output {
    let mut undefinable_names = HashSet::new();
    undefinable_names.extend(builtin_constants.iter().cloned().chain(["x", "y"]));
    let mut resolver = Resolver::new(
        list.iter().map(|e| {
            let e = e.borrow();
            (e.expression, e.slider.clone())
        }),
        &undefinable_names,
        use_v1_9_scoping_rules,
    );

    let builtin_constants = builtin_constants
        .iter()
        .map(|&name| {
            let id = resolver.next_id();
            let deps = Dependencies::default();
            let existing = resolver.scopes[deps.scope_index()]
                .computed
                .insert(name, (Ok(id), None, deps));
            assert_eq!(existing, None);
            (name.to_string(), id)
        })
        .collect::<HashMap<_, _>>();

    let results = list
        .iter()
        .map(Borrow::borrow)
        .map(|e| {
            // When we start resolving a new expression, there shouldn't be any
            // variables that are in scope from a `for` or `with` clause
            assert_eq!(resolver.line_count, 0);
            assert_eq!(resolver.assignments.len(), 1);
            assert_eq!(resolver.scopes.len(), 1);
            assert_eq!(resolver.scopes[0].substitutions, HashMap::new());
            assert_eq!(resolver.cycle_detector.stack, Vec::<&str>::new());
            assert!(resolver.cycle_detector.counts.values().all(|&c| c == 0));

            let mut result = match e.expression {
                Statement::Assignment { name, value } => {
                    let (id, slider_id, deps) = if !undefinable_names.contains(name.as_str())
                        && let Some((id, slider_id, deps)) =
                            resolver.scopes[0].computed.get(name.as_str())
                    {
                        (id.clone(), slider_id.clone(), deps.clone())
                    } else {
                        resolver
                            .cycle_detector
                            .push(name)
                            .expect("can't have a cycle before you even begin");
                        let (id, slider_id, deps) =
                            resolver.resolve_value_slider(name, value, e.slider.clone());
                        resolver.cycle_detector.pop();
                        assert_eq!(deps.level(), Level(0));

                        if let Some(Ok(_)) = resolver.definitions.get(name.as_str()) {
                            resolver.scopes[0]
                                .computed
                                .insert(name, (id.clone(), slider_id.clone(), deps.clone()));
                        }

                        (id, slider_id, deps)
                    };
                    let mut freevars = deps
                        .keys()
                        .cloned()
                        .filter(|name| resolver.freevars.contains_key(name))
                        .collect::<Vec<_>>();
                    freevars.sort();
                    match (id, slider_id) {
                        (Ok(id), None) => {
                            if name == "x" && freevars.len() <= 1 && freevars != ["x"]
                                || name != "y" && freevars == ["y"]
                            {
                                ExpressionResult::Plot {
                                    allowed_kinds: if freevars == ["y"] {
                                        PlotKinds::INVERSE
                                    } else {
                                        PlotKinds::INVERSE | PlotKinds::PARAMETRIC
                                    },
                                    value: id,
                                    parameters: freevars
                                        .first()
                                        .map(|v| resolver.freevars[v])
                                        .to_vec(),
                                    domain: None,
                                }
                            } else if freevars.len() == 1
                                && freevars != [name]
                                && (name == "y" || !undefinable_names.contains(name.as_str()))
                                || name == "y" && freevars.is_empty()
                            {
                                ExpressionResult::Plot {
                                    allowed_kinds: if freevars == ["x"] {
                                        PlotKinds::NORMAL
                                    } else {
                                        PlotKinds::NORMAL | PlotKinds::PARAMETRIC
                                    },
                                    value: id,
                                    parameters: freevars
                                        .first()
                                        .map(|v| resolver.freevars[v])
                                        .to_vec(),
                                    domain: None,
                                }
                            } else if undefinable_names.contains(name.as_str())
                                && matches!(freevars[..], [] | ["x"] | ["y"] | ["x", "y"])
                            {
                                let lhs = Expression::Identifier(
                                    resolver.resolve_variable(name).unwrap(),
                                );
                                let rhs = Expression::Identifier(id);
                                let f = Expression::Op {
                                    operation: OpName::Sub,
                                    args: vec![lhs, rhs],
                                };
                                let value =
                                    resolver.push_assignment("<implicit plot>", Level(0), f);
                                ExpressionResult::Plot {
                                    allowed_kinds: PlotKinds::IMPLICIT,
                                    value,
                                    parameters: vec![
                                        resolver.resolve_variable("x").unwrap(),
                                        resolver.resolve_variable("y").unwrap(),
                                    ],
                                    domain: None,
                                }
                            } else if freevars.is_empty() {
                                ExpressionResult::Value(id)
                            } else {
                                ExpressionResult::Err(NameError::undefined(freevars))
                            }
                        }
                        (Err(e), None) => ExpressionResult::Err(e),
                        (id, Some(slider)) => {
                            // Check if slider depends on any freevars. If so then error,
                            // because sliders shouldn't depend on undefined variables
                            let mut had_error = false;
                            let slider = slider.map(|(id, deps)| {
                                id.and_then(|id| {
                                    let mut freevars = deps
                                        .keys()
                                        .cloned()
                                        .filter(|name| resolver.freevars.contains_key(name))
                                        .peekable();
                                    if freevars.peek().is_none() {
                                        Ok(id)
                                    } else {
                                        had_error = true;
                                        Err(NameError::undefined(freevars))
                                    }
                                })
                            });
                            ExpressionResult::Slider {
                                value: if had_error { None } else { id.ok() },
                                slider,
                            }
                        }
                    }
                }
                Statement::FunctionDeclaration {
                    name,
                    parameters,
                    body,
                } => {
                    if let Some(name) = OpName::from_str(name) {
                        let operands = resolver.resolve_with_dependencies(
                            |resolver| {
                                let lhs = Expression::Op {
                                    operation: name,
                                    args: parameters
                                        .iter()
                                        .map(|p| {
                                            Ok(Expression::Identifier(
                                                resolver.resolve_variable(p.as_str())?,
                                            ))
                                        })
                                        .collect::<Result<_, _>>()?,
                                };
                                let rhs = resolver.resolve_expression(body)?;
                                Ok(vec![lhs, rhs])
                            },
                            None,
                        );
                        resolve_relation(&mut resolver, operands, &[ComparisonOperator::Equal])
                    } else if let [parameter] = parameters.as_slice() {
                        let arg = "<anonymous function argument>";
                        let (value, deps) = resolver.resolve_with_dependencies(
                            |resolver| {
                                static ARG: OnceLock<ast::Expression> = OnceLock::new();
                                resolver.cycle_detector.push(name).unwrap();
                                let value = resolver.resolve_substitutions(
                                    body,
                                    !resolver.use_v1_9_scoping_rules,
                                    zip(
                                        parameters,
                                        [ARG.get_or_init(|| {
                                            ast::Expression::Identifier(arg.into())
                                        })],
                                    ),
                                    NameError::DuplicateFunctionParameter,
                                );
                                resolver.cycle_detector.pop();
                                value
                            },
                            None,
                        );
                        let level = deps.level();
                        assert_eq!(level, Level(0));
                        let id = value.map(|value| {
                            resolver.push_assignment("<anonymous function plot>", level, value)
                        });
                        let mut freevars = deps
                            .keys()
                            .cloned()
                            .filter(|name| resolver.freevars.contains_key(name))
                            .collect::<Vec<_>>();
                        freevars.sort();

                        match id {
                            Ok(id) => match freevars[..] {
                                [] | [_] => ExpressionResult::Plot {
                                    allowed_kinds: if parameter == "y" || name == "x" {
                                        PlotKinds::INVERSE
                                    } else {
                                        PlotKinds::NORMAL
                                    },
                                    value: id,
                                    parameters: freevars
                                        .first()
                                        .map(|v| resolver.freevars[v])
                                        .to_vec(),
                                    domain: None,
                                },
                                _ => ExpressionResult::Err(NameError::undefined(freevars)),
                            },
                            Err(e) => ExpressionResult::Err(e),
                        }
                    } else {
                        ExpressionResult::None
                    }
                }
                Statement::Relation(ast::ChainedComparison {
                    operands,
                    operators,
                }) => {
                    let operands = resolver.resolve_with_dependencies(
                        |resolver| {
                            operands
                                .iter()
                                .map(|e| resolver.resolve_expression(e))
                                .collect::<Result<_, _>>()
                        },
                        None,
                    );
                    resolve_relation(&mut resolver, operands, operators)
                }
                Statement::Expression(value) => {
                    let (value, deps) = resolver.resolve_expression_with_dependencies(value, None);
                    let level = deps.level();
                    assert_eq!(level, Level(0));
                    let id =
                        value.map(|value| resolver.push_assignment("<anonymous>", level, value));
                    let mut freevars = deps
                        .keys()
                        .cloned()
                        .filter(|name| resolver.freevars.contains_key(name))
                        .collect::<Vec<_>>();
                    freevars.sort();
                    match id {
                        Ok(id) => match freevars[..] {
                            [] => ExpressionResult::Value(id),
                            ["x"] => ExpressionResult::Plot {
                                allowed_kinds: PlotKinds::NORMAL,
                                value: id,
                                parameters: vec![resolver.freevars["x"]],
                                domain: None,
                            },
                            [v] if v != "y" => ExpressionResult::Plot {
                                allowed_kinds: PlotKinds::PARAMETRIC,
                                value: id,
                                parameters: vec![resolver.freevars[v]],
                                domain: None,
                            },
                            ["y"] => ExpressionResult::Err(NameError::ExpressionWithFreeVariablY),
                            _ => ExpressionResult::Err(NameError::undefined(freevars)),
                        },
                        Err(e) => ExpressionResult::Err(e),
                    }
                }
            };

            match &mut result {
                ExpressionResult::Plot {
                    allowed_kinds,
                    domain,
                    ..
                } if allowed_kinds.contains(PlotKinds::PARAMETRIC) => {
                    let mut f = |e, name| {
                        let (value, deps) = resolver.resolve_expression_with_dependencies(e, None);
                        value.and_then(|value| {
                            let freevars = deps
                                .keys()
                                .filter(|&name| resolver.freevars.contains_key(name))
                                .map(ToString::to_string)
                                .collect::<Vec<_>>();
                            if freevars.is_empty() {
                                let level = deps.level();
                                assert_eq!(level, Level(0));
                                let id = resolver.push_assignment(name, level, value);
                                Ok(id)
                            } else {
                                Err(NameError::undefined(freevars))
                            }
                        })
                    };
                    *domain = Some(Domain {
                        min: f(e.parametric_domain.min, "<parametric min>"),
                        max: f(e.parametric_domain.max, "<parametric max>"),
                    });
                }
                _ => {}
            }

            result
        })
        .collect();

    let assignments = resolver.assignments.pop().unwrap();
    assert!(resolver.assignments.is_empty());
    let freevars = resolver
        .freevars
        .iter()
        .map(|(&k, &v)| (k.into(), v))
        .collect();
    Output {
        assignments,
        results,
        freevars,
        builtin_constants,
    }
}

#[cfg(test)]
mod tests {
    use std::hash::Hash;

    use super::*;
    use Statement::{
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

    fn canonicalize_assignment_ids(a: &mut [Assignment], f: &mut impl FnMut(&mut Id)) {
        for a in a {
            a.canonicalize_ids(f);
        }
    }

    impl Body {
        fn canonicalize_ids(&mut self, f: &mut impl FnMut(&mut Id)) {
            canonicalize_assignment_ids(&mut self.assignments, f);
            self.value.canonicalize_ids(f);
        }
    }

    impl Expression {
        fn canonicalize_ids(&mut self, f: &mut impl FnMut(&mut Id)) {
            fn canonicalize_list(es: &mut [Expression], f: &mut impl FnMut(&mut Id)) {
                for e in es {
                    e.canonicalize_ids(f);
                }
            }

            match self {
                Expression::Number(_) => {}
                Expression::Identifier(id) => f(id),
                Expression::Slider { value, slider } => {
                    value.canonicalize_ids(f);
                    for field in slider.fields_mut() {
                        field.canonicalize_ids(f);
                    }
                }
                Expression::List(list) => canonicalize_list(list, f),
                Expression::ListRange {
                    before_ellipsis,
                    after_ellipsis,
                } => {
                    canonicalize_list(before_ellipsis, f);
                    canonicalize_list(after_ellipsis, f);
                }
                Expression::Op { args, .. } => canonicalize_list(args, f),
                Expression::ChainedComparison { operands, .. } => canonicalize_list(operands, f),
                Expression::Piecewise {
                    test,
                    consequent,
                    alternate,
                } => {
                    test.canonicalize_ids(f);
                    consequent.canonicalize_ids(f);
                    if let Some(a) = alternate {
                        a.canonicalize_ids(f);
                    }
                }
                Expression::SumProd {
                    variable,
                    lower_bound,
                    upper_bound,
                    body,
                    ..
                } => {
                    f(variable);
                    lower_bound.canonicalize_ids(f);
                    upper_bound.canonicalize_ids(f);
                    body.canonicalize_ids(f);
                }
                Expression::For { body, lists } => {
                    body.canonicalize_ids(f);
                    canonicalize_assignment_ids(lists, f);
                }
            }
        }
    }

    impl Assignment {
        fn canonicalize_ids(&mut self, f: &mut impl FnMut(&mut Id)) {
            f(&mut self.id);
            self.value.canonicalize_ids(f);
        }
    }

    impl ExpressionResult {
        fn canonicalize_ids(&mut self, f: &mut impl FnMut(&mut Id)) {
            match self {
                ExpressionResult::None => {}
                ExpressionResult::Err(_) => {}
                ExpressionResult::Value(id) => f(id),
                ExpressionResult::Slider { value, slider } => {
                    if let Some(id) = value {
                        f(id);
                    }
                    for field in slider.fields_mut() {
                        if let Ok(id) = field {
                            f(id);
                        }
                    }
                }
                ExpressionResult::Plot {
                    allowed_kinds: _,
                    value,
                    parameters,
                    domain,
                } => {
                    f(value);
                    for p in parameters {
                        f(p);
                    }
                    if let Some(Domain { min, max }) = domain {
                        if let Ok(id) = min {
                            f(id);
                        }
                        if let Ok(id) = max {
                            f(id);
                        }
                    }
                }
            }
        }
    }

    type ResolveResult = (Vec<Assignment>, Vec<ExpressionResult>, HashMap<String, Id>);
    type ResolveResultWithBuiltins = (
        Vec<Assignment>,
        Vec<ExpressionResult>,
        HashMap<String, Id>,
        HashMap<String, Id>,
    );

    /// Rename `Id`s to be in a canonical order so that tests don't fail just
    /// because `Id`s are different even if they represent the same result
    fn canonicalize_ids_with_builtins(
        (mut assignments, mut results, mut freevars, mut builtin_constants): ResolveResultWithBuiltins,
    ) -> ResolveResultWithBuiltins {
        let mut old_to_new = HashMap::new();
        let f = &mut |id: &mut Id| {
            let next = Id(old_to_new.len());
            *id = *old_to_new.entry(*id).or_insert(next);
        };

        canonicalize_assignment_ids(&mut assignments, f);
        for r in &mut results {
            r.canonicalize_ids(f);
        }

        let mut g = |h: &mut HashMap<String, Id>| {
            let mut kvs = h.iter_mut().collect::<Vec<_>>();
            kvs.sort_by_key(|(k, _)| *k);
            kvs.iter_mut().for_each(|(_, v)| f(v));
        };
        g(&mut freevars);
        g(&mut builtin_constants);

        (assignments, results, freevars, builtin_constants)
    }

    /// Rename `Id`s to be in a canonical order so that tests don't fail just
    /// because `Id`s are different even if they represent the same result
    fn canonicalize_ids(result: ResolveResult) -> ResolveResult {
        let (a, b, c, _) =
            canonicalize_ids_with_builtins((result.0, result.1, result.2, HashMap::new()));
        (a, b, c)
    }

    #[derive(Default)]
    struct IdGenerator<T>(HashMap<T, Id>);

    impl<T: Eq + Hash> std::ops::Index<T> for IdGenerator<T> {
        type Output = Id;

        fn index(&self, index: T) -> &Self::Output {
            &self.0[&index]
        }
    }

    impl<T: Eq + Hash + std::fmt::Debug> IdGenerator<T> {
        fn new_id(&mut self, name: T) -> Id {
            let id = Id(self.0.len());
            assert!(!self.0.contains_key(&name), "id {name:?} already exists");
            self.0.insert(name, id);
            id
        }
    }

    fn bx<T>(x: T) -> Box<T> {
        Box::new(x)
    }

    fn resolve_names_ti(list: &[Statement]) -> ResolveResult {
        let list = list
            .iter()
            .map(|e| ExpressionListEntry {
                expression: e,
                parametric_domain: Domain {
                    min: &ANum(0.0),
                    max: &ANum(1.0),
                },
                slider: None,
            })
            .collect::<TiVec<_, _>>();
        let o = resolve_names(list.as_ref(), &[], false);
        (o.assignments, o.results.into(), o.freevars)
    }

    fn resolve_names_ti_with_builtins(
        list: &[Statement],
        builtin_constants: &[&str],
    ) -> ResolveResultWithBuiltins {
        let list = list
            .iter()
            .map(|e| ExpressionListEntry {
                expression: e,
                parametric_domain: Domain {
                    min: &ANum(0.0),
                    max: &ANum(1.0),
                },
                slider: None,
            })
            .collect::<TiVec<_, _>>();
        let o = resolve_names(list.as_ref(), builtin_constants, false);
        (
            o.assignments,
            o.results.into(),
            o.freevars,
            o.builtin_constants,
        )
    }

    fn resolve_names_ti_v1_9(list: &[Statement]) -> ResolveResult {
        let list = list
            .iter()
            .map(|e| ExpressionListEntry {
                expression: e,
                parametric_domain: Domain {
                    min: &ANum(0.0),
                    max: &ANum(1.0),
                },
                slider: None,
            })
            .collect::<TiVec<_, _>>();
        let o = resolve_names(list.as_ref(), &[], true);
        (o.assignments, o.results.into(), o.freevars)
    }

    fn assert_eq(a: ResolveResult, b: ResolveResult) {
        assert_eq!(canonicalize_ids(a), canonicalize_ids(b));
    }

    fn assert_eq_with_builtins(a: ResolveResultWithBuiltins, b: ResolveResultWithBuiltins) {
        assert_eq!(
            canonicalize_ids_with_builtins(a),
            canonicalize_ids_with_builtins(b)
        );
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
                        id: Id(0),
                        name: "<anonymous>".into(),
                        value: Expression::Number(5.0),
                    },
                    Assignment {
                        id: Id(1),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![Expression::Number(1.0), Expression::Number(2.0)],
                        },
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(1))
                ],
                HashMap::from([]),
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
                        id: Id(0),
                        name: "c".into(),
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(1))
                ],
                HashMap::from([]),
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
                // c = 1
                ElAssign {
                    name: "c".into(),
                    value: ANum(1.0),
                },
                // c = 2
                ElAssign {
                    name: "c".into(),
                    value: ANum(2.0),
                },
                // c = 3
                ElAssign {
                    name: "c".into(),
                    value: ANum(3.0),
                },
                // d = c
                ElAssign {
                    name: "d".into(),
                    value: AId("c".into()),
                },
            ]),
            (
                vec![
                    Assignment {
                        id: Id(0),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: Id(1),
                        name: "a".into(),
                        value: Expression::Number(2.0),
                    },
                    Assignment {
                        id: Id(2),
                        name: "c".into(),
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: Id(3),
                        name: "c".into(),
                        value: Expression::Number(2.0),
                    },
                    Assignment {
                        id: Id(4),
                        name: "c".into(),
                        value: Expression::Number(3.0),
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(1)),
                    ExpressionResult::Err(NameError::MultipleDefinitions("a".into())),
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(4)),
                    ExpressionResult::Err(NameError::MultipleDefinitions("c".into())),
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn circular_error() {
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
                    ExpressionResult::Err(NameError::CyclicDefinition(vec![
                        "a".into(),
                        "b".into()
                    ])),
                    ExpressionResult::Err(NameError::CyclicDefinition(vec![
                        "a".into(),
                        "b".into()
                    ])),
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn circular_substitution_error() {
        assert_eq!(
            resolve_names_ti(&[
                // a = 1
                ElAssign {
                    name: "a".into(),
                    value: ANum(1.0)
                },
                // b = b + a with a = a + 1
                ElAssign {
                    name: "b".into(),
                    value: AWith {
                        body: bx(AOp {
                            operation: OpName::Add,
                            args: vec![AId("a".into()), AId("b".into())]
                        }),
                        substitutions: vec![(
                            "a".into(),
                            AOp {
                                operation: OpName::Add,
                                args: vec![AId("a".into()), ANum(1.0)]
                            }
                        )]
                    }
                },
            ]),
            (
                vec![
                    // a = 1
                    Assignment {
                        id: Id(0),
                        name: "a".into(),
                        value: Expression::Number(1.0)
                    },
                    // with a = a + 1
                    Assignment {
                        id: Id(1),
                        name: "a".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![Expression::Identifier(Id(0)), Expression::Number(1.0)]
                        }
                    },
                    // with a = a + 1
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![Expression::Identifier(Id(1)), Expression::Number(1.0)]
                        }
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Err(NameError::CyclicDefinition(vec!["b".into()])),
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn circular_function_error() {
        assert_eq!(
            resolve_names_ti(&[
                // f(x) = g(x)
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["x".into()],
                    body: ACallMul {
                        callee: "g".into(),
                        args: vec![AId("x".into())]
                    },
                },
                // g(x) = a
                ElFunction {
                    name: "g".into(),
                    parameters: vec!["x".into()],
                    body: AId("a".into()),
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
                // a = f(3)
                ElAssign {
                    name: "a".into(),
                    value: ACallMul {
                        callee: "f".into(),
                        args: vec![ANum(3.0)]
                    },
                },
                // h() = i()
                ElFunction {
                    name: "h".into(),
                    parameters: vec![],
                    body: ACallMul {
                        callee: "i".into(),
                        args: vec![]
                    },
                },
                // i() = h()
                ElFunction {
                    name: "i".into(),
                    parameters: vec![],
                    body: ACallMul {
                        callee: "h".into(),
                        args: vec![]
                    },
                },
                // h()
                ElExpr(ACall {
                    callee: "h".into(),
                    args: vec![]
                })
            ]),
            (
                vec![
                    // freevar <anonymous function argument>: 0
                    // x = <anonymous function argument>
                    Assignment {
                        id: Id(1),
                        name: "x".into(),
                        value: Expression::Identifier(Id(0))
                    },
                    // x = x
                    Assignment {
                        id: Id(2),
                        name: "x".into(),
                        value: Expression::Identifier(Id(1))
                    },
                    // x = 3
                    Assignment {
                        id: Id(3),
                        name: "x".into(),
                        value: Expression::Number(3.0)
                    },
                    // x = x
                    Assignment {
                        id: Id(4),
                        name: "x".into(),
                        value: Expression::Identifier(Id(3))
                    },
                    // x = <anonymous function argument>
                    Assignment {
                        id: Id(5),
                        name: "x".into(),
                        value: Expression::Identifier(Id(0))
                    },
                ],
                vec![
                    ExpressionResult::Err(NameError::CyclicDefinition(vec![
                        "a".into(),
                        "f".into(),
                        "g".into()
                    ])),
                    ExpressionResult::Err(NameError::CyclicDefinition(vec![
                        "a".into(),
                        "f".into(),
                        "g".into()
                    ])),
                    ExpressionResult::Err(NameError::CyclicDefinition(vec![
                        "a".into(),
                        "f".into(),
                        "g".into()
                    ])),
                    ExpressionResult::Err(NameError::CyclicDefinition(vec![
                        "a".into(),
                        "f".into(),
                        "g".into()
                    ])),
                    ExpressionResult::None,
                    ExpressionResult::None,
                    ExpressionResult::Err(NameError::CyclicDefinition(vec![
                        "h".into(),
                        "i".into()
                    ])),
                ],
                HashMap::from([("<anonymous function argument>".into(), Id(0))]),
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
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(3.0)
                    },
                    // c = b
                    Assignment {
                        id: Id(1),
                        name: "c".into(),
                        value: Expression::Identifier(Id(0))
                    },
                    // a = c with b = 3
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Identifier(Id(1))
                    },
                    // b = a
                    Assignment {
                        id: Id(3),
                        name: "b".into(),
                        value: Expression::Identifier(Id(2))
                    },
                    // c = b
                    Assignment {
                        id: Id(4),
                        name: "c".into(),
                        value: Expression::Identifier(Id(3))
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(4))
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn funny_not_circular_reversed() {
        assert_eq!(
            resolve_names_ti(&[
                // c = b
                ElAssign {
                    name: "c".into(),
                    value: AId("b".into()),
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
                // a = c with b = 3
                ElAssign {
                    name: "a".into(),
                    value: AWith {
                        body: bx(AId("c".into())),
                        substitutions: vec![("b".into(), ANum(3.0))]
                    },
                },
            ]),
            (
                vec![
                    // with b = 3
                    Assignment {
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(3.0)
                    },
                    // c = b
                    Assignment {
                        id: Id(1),
                        name: "c".into(),
                        value: Expression::Identifier(Id(0))
                    },
                    // a = c with b = 3
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Identifier(Id(1))
                    },
                    // b = a
                    Assignment {
                        id: Id(3),
                        name: "b".into(),
                        value: Expression::Identifier(Id(2))
                    },
                    // c = b
                    Assignment {
                        id: Id(4),
                        name: "c".into(),
                        value: Expression::Identifier(Id(3))
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(4)),
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(2))
                ],
                HashMap::from([]),
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
                        id: Id(0),
                        name: "c".into(),
                        value: Expression::Number(1.0),
                    },
                    // b = c
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // with c = 2
                    Assignment {
                        id: Id(2),
                        name: "c".into(),
                        value: Expression::Number(2.0),
                    },
                    // b = c
                    Assignment {
                        id: Id(3),
                        name: "b".into(),
                        value: Expression::Identifier(Id(2)),
                    },
                    // a = b
                    Assignment {
                        id: Id(4),
                        name: "a".into(),
                        value: Expression::Identifier(Id(3)),
                    },
                    // with b = 3
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Number(3.0),
                    },
                    // with c = 2
                    Assignment {
                        id: Id(6),
                        name: "c".into(),
                        value: Expression::Number(2.0),
                    },
                    // a = b
                    Assignment {
                        id: Id(7),
                        name: "a".into(),
                        value: Expression::Identifier(Id(5)),
                    },
                    // a
                    Assignment {
                        id: Id(8),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(7)),
                    },
                    // with c = 4
                    Assignment {
                        id: Id(9),
                        name: "c".into(),
                        value: Expression::Number(4.0),
                    },
                    // b = c
                    Assignment {
                        id: Id(10),
                        name: "b".into(),
                        value: Expression::Identifier(Id(9)),
                    },
                    // b
                    Assignment {
                        id: Id(11),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(10)),
                    },
                    // with c = 5
                    Assignment {
                        id: Id(12),
                        name: "c".into(),
                        value: Expression::Number(5.0),
                    },
                    // a
                    Assignment {
                        id: Id(13),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(1)),
                    ExpressionResult::Value(Id(4)),
                    ExpressionResult::Value(Id(8)),
                    ExpressionResult::Value(Id(11)),
                    ExpressionResult::Value(Id(13)),
                ],
                HashMap::from([]),
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
                vec![
                    // freevar a: 0,
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Number(1.0),
                    }
                ],
                vec![
                    ExpressionResult::Err(NameError::VariableAsFunction("a".into())),
                    ExpressionResult::Value(Id(1)),
                    ExpressionResult::Err(NameError::VariableAsFunction("b".into())),
                    ExpressionResult::None,
                    ExpressionResult::Err(NameError::FunctionAsVariable("c".into())),
                ],
                HashMap::from([("a".into(), Id(0))]),
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
                        id: Id(0),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    Assignment {
                        id: Id(1),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Mul,
                            args: vec![Expression::Identifier(Id(0)), Expression::Number(2.0)]
                        },
                    },
                    Assignment {
                        id: Id(2),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Mul,
                            args: vec![
                                Expression::Identifier(Id(0)),
                                Expression::Op {
                                    operation: OpName::Point,
                                    args: vec![Expression::Number(3.0), Expression::Number(4.0)],
                                }
                            ]
                        },
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(1)),
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Err(NameError::BadPointDimension),
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn function_v1_9() {
        assert_eq(
            resolve_names_ti_v1_9(&[
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
                    // freevar a2: 0
                    // b = a2
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    Assignment {
                        id: Id(16),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(17),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // a3 = 5
                    Assignment {
                        id: Id(2),
                        name: "a3".into(),
                        value: Expression::Number(5.0),
                    },
                    // c = a3
                    Assignment {
                        id: Id(3),
                        name: "c".into(),
                        value: Expression::Identifier(Id(2)),
                    },
                    // freevar a4: 4
                    // d = a4
                    Assignment {
                        id: Id(5),
                        name: "d".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    Assignment {
                        id: Id(18),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(19),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // with a1 = 6
                    Assignment {
                        id: Id(6),
                        name: "a1".into(),
                        value: Expression::Number(6.0),
                    },
                    // with a2 = 7
                    Assignment {
                        id: Id(7),
                        name: "a2".into(),
                        value: Expression::Number(7.0),
                    },
                    // a1 = 1
                    Assignment {
                        id: Id(8),
                        name: "a1".into(),
                        value: Expression::Number(1.0),
                    },
                    // a2 = 2
                    Assignment {
                        id: Id(9),
                        name: "a2".into(),
                        value: Expression::Number(2.0),
                    },
                    // a3 = 3
                    Assignment {
                        id: Id(10),
                        name: "a3".into(),
                        value: Expression::Number(3.0),
                    },
                    // a4 = 4
                    Assignment {
                        id: Id(11),
                        name: "a4".into(),
                        value: Expression::Number(4.0),
                    },
                    // b = a2
                    Assignment {
                        id: Id(12),
                        name: "b".into(),
                        value: Expression::Identifier(Id(9)),
                    },
                    // c = a3
                    Assignment {
                        id: Id(13),
                        name: "c".into(),
                        value: Expression::Identifier(Id(10)),
                    },
                    // d = a4
                    Assignment {
                        id: Id(14),
                        name: "d".into(),
                        value: Expression::Identifier(Id(11)),
                    },
                    // [a1, b, c, d]
                    Assignment {
                        id: Id(15),
                        name: "<anonymous>".into(),
                        value: Expression::List(vec![
                            Expression::Identifier(Id(8)),
                            Expression::Identifier(Id(12)),
                            Expression::Identifier(Id(13)),
                            Expression::Identifier(Id(14)),
                        ]),
                    },
                ],
                vec![
                    ExpressionResult::None,
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(1),
                        parameters: vec![Id(0)],
                        domain: Some(Domain {
                            min: Ok(Id(16)),
                            max: Ok(Id(17)),
                        }),
                    },
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)],
                        domain: Some(Domain {
                            min: Ok(Id(18)),
                            max: Ok(Id(19)),
                        }),
                    },
                    ExpressionResult::Value(Id(15)),
                ],
                HashMap::from([("a2".into(), Id(0)), ("a4".into(), Id(4))]),
            ),
        );
    }

    #[test]
    fn function_v1_10() {
        // https://www.desmos.com/calculator/1jougp3ykk
        assert_eq(
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
                    // freevar a2: 0,
                    // b = a2
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    Assignment {
                        id: Id(15),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(16),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // a3 = 5
                    Assignment {
                        id: Id(2),
                        name: "a3".into(),
                        value: Expression::Number(5.0),
                    },
                    // c = a3
                    Assignment {
                        id: Id(3),
                        name: "c".into(),
                        value: Expression::Identifier(Id(2)),
                    },
                    // freevar a4: 4
                    // d = a4
                    Assignment {
                        id: Id(5),
                        name: "d".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    Assignment {
                        id: Id(17),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(18),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // with a1 = 6
                    Assignment {
                        id: Id(6),
                        name: "a1".into(),
                        value: Expression::Number(6.0),
                    },
                    // with a2 = 7
                    Assignment {
                        id: Id(7),
                        name: "a2".into(),
                        value: Expression::Number(7.0),
                    },
                    // a1 = 1
                    Assignment {
                        id: Id(8),
                        name: "a1".into(),
                        value: Expression::Number(1.0),
                    },
                    // a2 = 2
                    Assignment {
                        id: Id(9),
                        name: "a2".into(),
                        value: Expression::Number(2.0),
                    },
                    // a3 = 3
                    Assignment {
                        id: Id(10),
                        name: "a3".into(),
                        value: Expression::Number(3.0),
                    },
                    // a4 = 4
                    Assignment {
                        id: Id(11),
                        name: "a4".into(),
                        value: Expression::Number(4.0),
                    },
                    // b = a2
                    Assignment {
                        id: Id(12),
                        name: "b".into(),
                        value: Expression::Identifier(Id(7)),
                    },
                    // d = a4
                    Assignment {
                        id: Id(13),
                        name: "d".into(),
                        value: Expression::Identifier(Id(11)),
                    },
                    // [a1, b, c, d]
                    Assignment {
                        id: Id(14),
                        name: "<anonymous>".into(),
                        value: Expression::List(vec![
                            Expression::Identifier(Id(8)),
                            Expression::Identifier(Id(12)),
                            Expression::Identifier(Id(3)),
                            Expression::Identifier(Id(13)),
                        ]),
                    },
                ],
                vec![
                    ExpressionResult::None,
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(1),
                        parameters: vec![Id(0)],
                        domain: Some(Domain {
                            min: Ok(Id(15)),
                            max: Ok(Id(16)),
                        }),
                    },
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)],
                        domain: Some(Domain {
                            min: Ok(Id(17)),
                            max: Ok(Id(18)),
                        }),
                    },
                    ExpressionResult::Value(Id(14)),
                ],
                HashMap::from([("a2".into(), Id(0)), ("a4".into(), Id(4))]),
            ),
        );
    }

    #[test]
    fn wackscope() {
        assert_eq(
            resolve_names_ti(&[
                // f(a) = b + b
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a".into()],
                    body: AOp {
                        operation: OpName::Add,
                        args: vec![AId("b".into()), AId("b".into())],
                    },
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
                // f(1)
                ElExpr(ACallMul {
                    callee: "f".into(),
                    args: vec![ANum(1.0)],
                }),
            ]),
            (
                vec![
                    // freevar <anonymous function argument>: 0
                    // a = <anonymous function argument>
                    Assignment {
                        id: Id(1),
                        name: "a".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // b = a
                    Assignment {
                        id: Id(2),
                        name: "b".into(),
                        value: Expression::Identifier(Id(1)),
                    },
                    // f(<anonymous function argument>)
                    Assignment {
                        id: Id(3),
                        name: "<anonymous function plot>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(2)),
                                Expression::Identifier(Id(2)),
                            ],
                        },
                    },
                    // freevar a: 4
                    // b = a
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    Assignment {
                        id: Id(9),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(10),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // a = 1
                    Assignment {
                        id: Id(6),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    // b = a
                    Assignment {
                        id: Id(7),
                        name: "b".into(),
                        value: Expression::Identifier(Id(6)),
                    },
                    // f(1)
                    Assignment {
                        id: Id(8),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(7)),
                                Expression::Identifier(Id(7)),
                            ],
                        },
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(3),
                        parameters: vec![Id(0)],
                        domain: None,
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)],
                        domain: Some(Domain {
                            min: Ok(Id(9)),
                            max: Ok(Id(10)),
                        }),
                    },
                    ExpressionResult::Value(Id(8)),
                ],
                HashMap::from([
                    ("<anonymous function argument>".into(), Id(0)),
                    ("a".into(), Id(4)),
                ]),
            ),
        );
    }

    #[test]
    fn more_function_v1_10() {
        assert_eq(
            resolve_names_ti(&[
                // f(a) = b
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a".into()],
                    body: AId("b".into()),
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
                // b + f(1) with a = 2
                ElExpr(AWith {
                    body: bx(AOp {
                        operation: OpName::Add,
                        args: vec![
                            AId("b".into()),
                            ACallMul {
                                callee: "f".into(),
                                args: vec![ANum(1.0)],
                            },
                        ],
                    }),
                    substitutions: vec![("a".into(), ANum(2.0))],
                }),
            ]),
            (
                vec![
                    // freevar <anonymous function argument>: 0
                    // a = <anonymous function argument>
                    Assignment {
                        id: Id(1),
                        name: "a".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // b = a
                    Assignment {
                        id: Id(2),
                        name: "b".into(),
                        value: Expression::Identifier(Id(1)),
                    },
                    // f(<anonymous function argument>)
                    Assignment {
                        id: Id(3),
                        name: "<anonymous function plot>".into(),
                        value: Expression::Identifier(Id(2)),
                    },
                    // freevar a: 4
                    // b = a
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    Assignment {
                        id: Id(10),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(11),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // with a = 2
                    Assignment {
                        id: Id(6),
                        name: "a".into(),
                        value: Expression::Number(2.0),
                    },
                    // b = a
                    Assignment {
                        id: Id(7),
                        name: "b".into(),
                        value: Expression::Identifier(Id(6)),
                    },
                    // a = 1
                    Assignment {
                        id: Id(8),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    // b + f(1) with a = 2
                    Assignment {
                        id: Id(9),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(7)),
                                Expression::Identifier(Id(7)),
                            ],
                        },
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(3),
                        parameters: vec![Id(0)],
                        domain: None,
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)],
                        domain: Some(Domain {
                            min: Ok(Id(10)),
                            max: Ok(Id(11)),
                        }),
                    },
                    ExpressionResult::Value(Id(9)),
                ],
                HashMap::from([
                    ("<anonymous function argument>".into(), Id(0)),
                    ("a".into(), Id(4)),
                ]),
            ),
        );
    }

    #[test]
    fn even_more_function_v1_10() {
        assert_eq(
            resolve_names_ti(&[
                // f(a) = b + b
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a".into()],
                    body: AOp {
                        operation: OpName::Add,
                        args: vec![AId("b".into()), AId("b".into())],
                    },
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
                // g(a) = b + f(1) + b
                ElFunction {
                    name: "g".into(),
                    parameters: vec!["a".into()],
                    body: AOp {
                        operation: OpName::Add,
                        args: vec![
                            AOp {
                                operation: OpName::Add,
                                args: vec![
                                    AId("b".into()),
                                    ACallMul {
                                        callee: "f".into(),
                                        args: vec![ANum(1.0)],
                                    },
                                ],
                            },
                            AId("b".into()),
                        ],
                    },
                },
                // g(2)
                ElExpr(ACallMul {
                    callee: "g".into(),
                    args: vec![ANum(2.0)],
                }),
            ]),
            (
                vec![
                    // freevar <anonymous function argument>: 0
                    // a = <anonymous function argument>
                    Assignment {
                        id: Id(1),
                        name: "a".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // b = a
                    Assignment {
                        id: Id(2),
                        name: "b".into(),
                        value: Expression::Identifier(Id(1)),
                    },
                    // f(<anonymous function argument>)
                    Assignment {
                        id: Id(3),
                        name: "<anonymous function plot>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(2)),
                                Expression::Identifier(Id(2)),
                            ],
                        },
                    },
                    // freevar a: 4
                    // b = a
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    Assignment {
                        id: Id(16),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(17),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // a = <anonymous function argument>
                    Assignment {
                        id: Id(6),
                        name: "a".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // b = a
                    Assignment {
                        id: Id(7),
                        name: "b".into(),
                        value: Expression::Identifier(Id(6)),
                    },
                    // a = 1
                    Assignment {
                        id: Id(8),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    // b = a
                    Assignment {
                        id: Id(9),
                        name: "b".into(),
                        value: Expression::Identifier(Id(8)),
                    },
                    // g(<anonymous function argument>)
                    Assignment {
                        id: Id(10),
                        name: "<anonymous function plot>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Op {
                                    operation: OpName::Add,
                                    args: vec![
                                        Expression::Identifier(Id(7)),
                                        Expression::Op {
                                            operation: OpName::Add,
                                            args: vec![
                                                Expression::Identifier(Id(9)),
                                                Expression::Identifier(Id(9)),
                                            ],
                                        },
                                    ],
                                },
                                Expression::Identifier(Id(7)),
                            ],
                        },
                    },
                    // a = 2
                    Assignment {
                        id: Id(11),
                        name: "a".into(),
                        value: Expression::Number(2.0),
                    },
                    // b = a
                    Assignment {
                        id: Id(12),
                        name: "b".into(),
                        value: Expression::Identifier(Id(11)),
                    },
                    // a = 1
                    Assignment {
                        id: Id(13),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    // b = a
                    Assignment {
                        id: Id(14),
                        name: "b".into(),
                        value: Expression::Identifier(Id(13)),
                    },
                    // g(2) = b + f(1) + b
                    Assignment {
                        id: Id(15),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Op {
                                    operation: OpName::Add,
                                    args: vec![
                                        Expression::Identifier(Id(12)),
                                        Expression::Op {
                                            operation: OpName::Add,
                                            args: vec![
                                                Expression::Identifier(Id(14)),
                                                Expression::Identifier(Id(14)),
                                            ],
                                        },
                                    ],
                                },
                                Expression::Identifier(Id(12)),
                            ],
                        },
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(3),
                        parameters: vec![Id(0)],
                        domain: None,
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)],
                        domain: Some(Domain {
                            min: Ok(Id(16)),
                            max: Ok(Id(17)),
                        }),
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(10),
                        parameters: vec![Id(0)],
                        domain: None,
                    },
                    ExpressionResult::Value(Id(15)),
                ],
                HashMap::from([
                    ("<anonymous function argument>".into(), Id(0)),
                    ("a".into(), Id(4)),
                ]),
            ),
        );
    }

    #[test]
    fn function_with() {
        assert_eq(
            resolve_names_ti(&[
                // f(a) = a
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a".into()],
                    body: AId("a".into()),
                },
                // g(b) = f(b)
                ElFunction {
                    name: "g".into(),
                    parameters: vec!["b".into()],
                    body: ACallMul {
                        callee: "f".into(),
                        args: vec![AId("b".into())],
                    },
                },
                // g(1) with b = 2
                ElExpr(AWith {
                    body: bx(ACallMul {
                        callee: "g".into(),
                        args: vec![ANum(1.0)],
                    }),
                    substitutions: vec![("b".into(), ANum(2.0))],
                }),
            ]),
            (
                vec![
                    // freevar <anonymous function argument>: 0
                    // a = <anonymous function argument>
                    Assignment {
                        id: Id(1),
                        name: "a".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // f(<anonymous function argument>)
                    Assignment {
                        id: Id(2),
                        name: "<anonymous function plot>".into(),
                        value: Expression::Identifier(Id(1)),
                    },
                    // b = <anonymous function argument>
                    Assignment {
                        id: Id(3),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // a = b
                    Assignment {
                        id: Id(4),
                        name: "a".into(),
                        value: Expression::Identifier(Id(3)),
                    },
                    // g(<anonymous function argument>)
                    Assignment {
                        id: Id(5),
                        name: "<anonymous function plot>".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    // with b = 2
                    Assignment {
                        id: Id(6),
                        name: "b".into(),
                        value: Expression::Number(2.0),
                    },
                    // b = 1
                    Assignment {
                        id: Id(7),
                        name: "b".into(),
                        value: Expression::Number(1.0),
                    },
                    // a = b
                    Assignment {
                        id: Id(8),
                        name: "a".into(),
                        value: Expression::Identifier(Id(7)),
                    },
                    // g(1) with b=2
                    Assignment {
                        id: Id(9),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(8)),
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(2),
                        parameters: vec![Id(0)],
                        domain: None,
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(5),
                        parameters: vec![Id(0)],
                        domain: None,
                    },
                    ExpressionResult::Value(Id(9)),
                ],
                HashMap::from([("<anonymous function argument>".into(), Id(0))]),
            ),
        );
    }

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
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(2.0)
                    },
                    // a = 1
                    Assignment {
                        id: Id(1),
                        name: "a".into(),
                        value: Expression::Number(1.0)
                    },
                    // c = 3
                    Assignment {
                        id: Id(2),
                        name: "c".into(),
                        value: Expression::Number(3.0)
                    },
                    // a
                    Assignment {
                        id: Id(3),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(1))
                    }
                ],
                vec![
                    ExpressionResult::Value(Id(1)),
                    ExpressionResult::Value(Id(3))
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn list_comp() {
        assert_eq(
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
                                args: vec![AId("i".into()), AId("k".into())],
                            },
                        ],
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
                        args: vec![AId("j".into()), AId("j".into())],
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
                        id: Id(0),
                        name: "c".into(),
                        value: Expression::List(vec![Expression::Number(2.0)]),
                    },
                    // k = 3
                    Assignment {
                        id: Id(4),
                        name: "k".into(),
                        value: Expression::Number(3.0),
                    },
                    // p for j=c, i=[1]
                    Assignment {
                        id: Id(6),
                        name: "<anonymous>".into(),
                        value: Expression::For {
                            body: Body {
                                assignments: vec![
                                    // q = jj
                                    Assignment {
                                        id: Id(3),
                                        name: "q".into(),
                                        value: Expression::Op {
                                            operation: OpName::Mul,
                                            args: vec![
                                                Expression::Identifier(Id(1)),
                                                Expression::Identifier(Id(1)),
                                            ],
                                        },
                                    },
                                    // p = (q,i+k)
                                    Assignment {
                                        id: Id(5),
                                        name: "p".into(),
                                        value: Expression::Op {
                                            operation: OpName::Point,
                                            args: vec![
                                                Expression::Identifier(Id(3)),
                                                Expression::Op {
                                                    operation: OpName::Add,
                                                    args: vec![
                                                        Expression::Identifier(Id(2)),
                                                        Expression::Identifier(Id(4)),
                                                    ],
                                                },
                                            ],
                                        },
                                    },
                                ],
                                value: bx(Expression::Identifier(Id(5))),
                            },
                            lists: vec![
                                // j=c
                                Assignment {
                                    id: Id(1),
                                    name: "j".into(),
                                    value: Expression::Identifier(Id(0)),
                                },
                                // i=[1]
                                Assignment {
                                    id: Id(2),
                                    name: "i".into(),
                                    value: Expression::List(vec![Expression::Number(1.0)]),
                                },
                            ],
                        },
                    },
                    // freevar j: 7,
                    // q = jj
                    Assignment {
                        id: Id(8),
                        name: "q".into(),
                        value: Expression::Op {
                            operation: OpName::Mul,
                            args: vec![
                                Expression::Identifier(Id(7)),
                                Expression::Identifier(Id(7)),
                            ],
                        },
                    },
                    // freevar i: 9,
                    // p = (q,i+k)
                    Assignment {
                        id: Id(10),
                        name: "p".into(),
                        value: Expression::Op {
                            operation: OpName::Point,
                            args: vec![
                                Expression::Identifier(Id(8)),
                                Expression::Op {
                                    operation: OpName::Add,
                                    args: vec![
                                        Expression::Identifier(Id(9)),
                                        Expression::Identifier(Id(4)),
                                    ],
                                },
                            ],
                        },
                    },
                    // parametric bounds for q=jj
                    Assignment {
                        id: Id(11),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(12),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(6)),
                    ExpressionResult::Err(NameError::undefined(["i", "j"])),
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(8),
                        parameters: vec![Id(7)],
                        domain: Some(Domain {
                            min: Ok(Id(11)),
                            max: Ok(Id(12)),
                        }),
                    },
                    ExpressionResult::Value(Id(4)),
                ],
                HashMap::from([("j".into(), Id(7)), ("i".into(), Id(9))]),
            ),
        );
    }

    #[test]
    fn nested_list_comps() {
        assert_eq(
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
                                },
                            ],
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
                                    args: vec![AId("B".into()), AId("A".into())],
                                },
                                AId("F".into()),
                            ],
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
                        args: vec![AId("i".into()), ANum(2.0)],
                    },
                },
                // F = i + j
                ElAssign {
                    // TODO: change this back to "J" and see why it was panicking
                    name: "F".into(),
                    value: AOp {
                        operation: OpName::Add,
                        args: vec![AId("i".into()), AId("j".into())],
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
                        id: Id(3),
                        name: "C".into(),
                        value: Expression::For {
                            body: Body {
                                assignments: vec![
                                    // B = i^2
                                    Assignment {
                                        id: Id(2),
                                        name: "B".into(),
                                        value: Expression::Op {
                                            operation: OpName::Pow,
                                            args: vec![
                                                Expression::Identifier(Id(1)),
                                                Expression::Number(2.0),
                                            ],
                                        },
                                    },
                                ],
                                // B
                                value: bx(Expression::Identifier(Id(2))),
                            },
                            lists: vec![
                                // i=[1...5]
                                Assignment {
                                    id: Id(1),
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
                        id: Id(6),
                        name: "A".into(),
                        value: Expression::Number(5.0),
                    },
                    // E = C[i] + D[i] for j=[1...4]
                    Assignment {
                        id: Id(9),
                        name: "E".into(),
                        value: Expression::For {
                            body: Body {
                                assignments: vec![
                                    // D = B + A + F for i=[1...3]
                                    Assignment {
                                        id: Id(8),
                                        name: "D".into(),
                                        value: Expression::For {
                                            body: Body {
                                                assignments: vec![
                                                    // B = i^2
                                                    Assignment {
                                                        id: Id(5),
                                                        name: "B".into(),
                                                        value: Expression::Op {
                                                            operation: OpName::Pow,
                                                            args: vec![
                                                                Expression::Identifier(Id(4)),
                                                                Expression::Number(2.0),
                                                            ],
                                                        },
                                                    },
                                                    // F = i + j
                                                    Assignment {
                                                        id: Id(7),
                                                        name: "F".into(),
                                                        value: Expression::Op {
                                                            operation: OpName::Add,
                                                            args: vec![
                                                                Expression::Identifier(Id(4)),
                                                                Expression::Identifier(Id(0)),
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
                                                                Expression::Identifier(Id(5)),
                                                                Expression::Identifier(Id(6)),
                                                            ],
                                                        },
                                                        Expression::Identifier(Id(7)),
                                                    ],
                                                }),
                                            },
                                            lists: vec![
                                                // i=[1...3]
                                                Assignment {
                                                    id: Id(4),
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
                                            args: vec![Expression::Identifier(Id(3))],
                                        },
                                        Expression::Op {
                                            operation: OpName::Total,
                                            args: vec![Expression::Identifier(Id(8))],
                                        },
                                    ],
                                }),
                            },
                            lists: vec![
                                // j=[1...4]
                                Assignment {
                                    id: Id(0),
                                    name: "j".into(),
                                    value: Expression::ListRange {
                                        before_ellipsis: vec![Expression::Number(1.0)],
                                        after_ellipsis: vec![Expression::Number(4.0)],
                                    },
                                },
                            ],
                        },
                    },
                    // freevar j: 12
                    // D = B + A + F for i=[1...3]
                    Assignment {
                        id: Id(14),
                        name: "D".into(),
                        value: Expression::For {
                            body: Body {
                                assignments: vec![
                                    // B = i^2
                                    Assignment {
                                        id: Id(11),
                                        name: "B".into(),
                                        value: Expression::Op {
                                            operation: OpName::Pow,
                                            args: vec![
                                                Expression::Identifier(Id(10)),
                                                Expression::Number(2.0),
                                            ],
                                        },
                                    },
                                    // F = i + j
                                    Assignment {
                                        id: Id(13),
                                        name: "F".into(),
                                        value: Expression::Op {
                                            operation: OpName::Add,
                                            args: vec![
                                                Expression::Identifier(Id(10)),
                                                Expression::Identifier(Id(12)),
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
                                                Expression::Identifier(Id(11)),
                                                Expression::Identifier(Id(6)),
                                            ],
                                        },
                                        Expression::Identifier(Id(13)),
                                    ],
                                }),
                            },
                            lists: vec![
                                // i=[1...3]
                                Assignment {
                                    id: Id(10),
                                    name: "i".into(),
                                    value: Expression::ListRange {
                                        before_ellipsis: vec![Expression::Number(1.0)],
                                        after_ellipsis: vec![Expression::Number(3.0)],
                                    },
                                },
                            ],
                        },
                    },
                    Assignment {
                        id: Id(18),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(19),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // freevar i: 15
                    // B = i^2
                    Assignment {
                        id: Id(16),
                        name: "B".into(),
                        value: Expression::Op {
                            operation: OpName::Pow,
                            args: vec![Expression::Identifier(Id(15)), Expression::Number(2.0)],
                        },
                    },
                    Assignment {
                        id: Id(20),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(21),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // F = i + j
                    Assignment {
                        id: Id(17),
                        name: "F".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(15)),
                                Expression::Identifier(Id(12)),
                            ],
                        },
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(9)),
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(14),
                        parameters: vec![Id(12)],
                        domain: Some(Domain {
                            min: Ok(Id(18)),
                            max: Ok(Id(19)),
                        }),
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(16),
                        parameters: vec![Id(15)],
                        domain: Some(Domain {
                            min: Ok(Id(20)),
                            max: Ok(Id(21)),
                        }),
                    },
                    ExpressionResult::Err(NameError::undefined(["i", "j"])),
                    ExpressionResult::Value(Id(6)),
                ],
                HashMap::from([("j".into(), Id(12)), ("i".into(), Id(15))]),
            ),
        );
    }

    #[test]
    fn proper_cleanup() {
        assert_eq(
            resolve_names_ti(&[
                // b + c with a = 1
                ElExpr(AWith {
                    body: bx(AOp {
                        operation: OpName::Add,
                        args: vec![AId("b".into()), AId("c".into())],
                    }),
                    substitutions: vec![("a".into(), ANum(1.0))],
                }),
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into()),
                },
            ]),
            (
                vec![
                    // with a = 1
                    Assignment {
                        id: Id(0),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    // b = a
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // freevar c: 2
                    // b + c with a = 1
                    Assignment {
                        id: Id(3),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(1)),
                                Expression::Identifier(Id(2)),
                            ],
                        },
                    },
                    Assignment {
                        id: Id(6),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(7),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // freevar a: 4
                    // b = a
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    Assignment {
                        id: Id(8),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(9),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::PARAMETRIC,
                        value: Id(3),
                        parameters: vec![Id(2)],
                        domain: Some(Domain {
                            min: Ok(Id(6)),
                            max: Ok(Id(7)),
                        }),
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)],
                        domain: Some(Domain {
                            min: Ok(Id(8)),
                            max: Ok(Id(9)),
                        }),
                    },
                ],
                HashMap::from([("c".into(), Id(2)), ("a".into(), Id(4))]),
            ),
        );
    }

    #[test]
    fn cache_errors() {
        assert_eq(
            resolve_names_ti(&[
                // a = c with b = 1
                ElAssign {
                    name: "a".into(),
                    value: AWith {
                        body: bx(AId("c".into())),
                        substitutions: vec![("b".into(), ANum(1.0))],
                    },
                },
                // a
                ElExpr(AId("a".into())),
            ]),
            (
                vec![
                    // with b = 1
                    Assignment {
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(1.0),
                    },
                    // freevar c: 1
                    // a = c with b = 1
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Identifier(Id(1)),
                    },
                    Assignment {
                        id: Id(4),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(5),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                    // a
                    Assignment {
                        id: Id(3),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(2)),
                    },
                    Assignment {
                        id: Id(6),
                        name: "<parametric min>".into(),
                        value: Expression::Number(0.0),
                    },
                    Assignment {
                        id: Id(7),
                        name: "<parametric max>".into(),
                        value: Expression::Number(1.0),
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(2),
                        parameters: vec![Id(1)],
                        domain: Some(Domain {
                            min: Ok(Id(4)),
                            max: Ok(Id(5)),
                        }),
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::PARAMETRIC,
                        value: Id(3),
                        parameters: vec![Id(1)],
                        domain: Some(Domain {
                            min: Ok(Id(6)),
                            max: Ok(Id(7)),
                        }),
                    },
                ],
                HashMap::from([("c".into(), Id(1))]),
            ),
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
                        id: Id(0),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    },
                    // b = a
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // c = b
                    Assignment {
                        id: Id(2),
                        name: "c".into(),
                        value: Expression::Identifier(Id(1)),
                    },
                    // with a = 5
                    Assignment {
                        id: Id(3),
                        name: "a".into(),
                        value: Expression::Number(5.0),
                    },
                    // b = a
                    Assignment {
                        id: Id(4),
                        name: "b".into(),
                        value: Expression::Identifier(Id(3)),
                    },
                    // c = b
                    Assignment {
                        id: Id(5),
                        name: "c".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    // c with a = 5
                    Assignment {
                        id: Id(6),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(5)),
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(1)),
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Value(Id(6))
                ],
                HashMap::from([]),
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
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(1.0)
                    },
                    Assignment {
                        id: Id(1),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(0))
                    },
                    Assignment {
                        id: Id(2),
                        name: "c".into(),
                        value: Expression::Number(1.0)
                    },
                    Assignment {
                        id: Id(3),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(2))
                    }
                ],
                vec![
                    ExpressionResult::None,
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(1)),
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Value(Id(3))
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn substituting_same_variable() {
        assert_eq!(
            resolve_names_ti(&[
                // a = (1 with b = b) with b = 2
                ElAssign {
                    name: "a".into(),
                    value: AWith {
                        body: bx(AWith {
                            body: bx(ANum(1.0)),
                            substitutions: vec![("b".into(), AId("b".into()))]
                        }),
                        substitutions: vec![("b".into(), ANum(2.0))]
                    }
                }
            ]),
            (
                vec![
                    // with b = 2
                    Assignment {
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(2.0),
                    },
                    // with b = b
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
                    },
                    // a = (1 with b = b) with b = 2
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Number(1.0),
                    }
                ],
                vec![ExpressionResult::Value(Id(2))],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn substituting_existing_variable() {
        assert_eq!(
            resolve_names_ti(&[
                // b = 1
                ElAssign {
                    name: "b".into(),
                    value: ANum(1.0)
                },
                // a = b + (b with b = 2)
                ElAssign {
                    name: "a".into(),
                    value: AOp {
                        operation: OpName::Add,
                        args: vec![
                            AId("b".into()),
                            AWith {
                                body: bx(AId("b".into())),
                                substitutions: vec![("b".into(), ANum(2.0))]
                            }
                        ]
                    }
                },
            ]),
            (
                vec![
                    // b = 1
                    Assignment {
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(1.0),
                    },
                    // with b = 2
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Number(2.0),
                    },
                    // a = b + (b with b = 2)
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(0)),
                                Expression::Identifier(Id(1))
                            ]
                        },
                    }
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(2))
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn comprehension_existing_variable() {
        assert_eq!(
            resolve_names_ti(&[
                // b = 1
                ElAssign {
                    name: "b".into(),
                    value: ANum(1.0)
                },
                // a = b + (b for b = [])
                ElAssign {
                    name: "a".into(),
                    value: AOp {
                        operation: OpName::Add,
                        args: vec![
                            AId("b".into()),
                            AFor {
                                body: bx(AId("b".into())),
                                lists: vec![("b".into(), AList(vec![]))]
                            }
                        ]
                    }
                },
            ]),
            (
                vec![
                    // b = 1
                    Assignment {
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(1.0),
                    },
                    // a = b + (b for b = [])
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(0)),
                                Expression::For {
                                    body: Body {
                                        assignments: vec![],
                                        value: bx(Expression::Identifier(Id(1))),
                                    },
                                    lists: vec![
                                        // b = []
                                        Assignment {
                                            id: Id(1),
                                            name: "b".into(),
                                            value: Expression::List(vec![]),
                                        },
                                    ],
                                }
                            ]
                        },
                    }
                ],
                vec![
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Value(Id(2))
                ],
                HashMap::from([]),
            ),
        );
    }

    #[test]
    fn function_transitive_dependency() {
        assert_eq!(
            resolve_names_ti(&[
                // f(a) = a + c + a
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a".into()],
                    body: AOp {
                        operation: OpName::Add,
                        args: vec![
                            AOp {
                                operation: OpName::Add,
                                args: vec![AId("a".into()), AId("c".into())]
                            },
                            AId("a".into())
                        ]
                    }
                },
                // a = 5
                ElAssign {
                    name: "a".into(),
                    value: ANum(5.0)
                },
                // c = a
                ElAssign {
                    name: "c".into(),
                    value: AId("a".into())
                },
                // f(3)
                ElExpr(ACallMul {
                    callee: "f".into(),
                    args: vec![ANum(3.0)]
                })
            ]),
            (
                vec![
                    // freevar <anonymous function argument>: 0
                    // a = <anonymous function argument>
                    Assignment {
                        id: Id(1),
                        name: "a".into(),
                        value: Expression::Identifier(Id(0))
                    },
                    // a = 5
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Number(5.0)
                    },
                    // c = a
                    Assignment {
                        id: Id(3),
                        name: "c".into(),
                        value: Expression::Identifier(Id(2))
                    },
                    // f(<anonymous function argument>)
                    Assignment {
                        id: Id(4),
                        name: "<anonymous function plot>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Op {
                                    operation: OpName::Add,
                                    args: vec![
                                        Expression::Identifier(Id(1)),
                                        Expression::Identifier(Id(3))
                                    ]
                                },
                                Expression::Identifier(Id(1))
                            ]
                        }
                    },
                    // a = 3
                    Assignment {
                        id: Id(5),
                        name: "a".into(),
                        value: Expression::Number(3.0)
                    },
                    // f(3) = a + c + a
                    Assignment {
                        id: Id(6),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Op {
                                    operation: OpName::Add,
                                    args: vec![
                                        Expression::Identifier(Id(5)),
                                        Expression::Identifier(Id(3))
                                    ]
                                },
                                Expression::Identifier(Id(5))
                            ]
                        }
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(4),
                        parameters: vec![Id(0)],
                        domain: None,
                    },
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(6))
                ],
                HashMap::from([("<anonymous function argument>".into(), Id(0))]),
            )
        );
    }

    #[test]
    fn parametric_domain() {
        let id = |s: &str| AId(s.into());
        let mut ids = IdGenerator::default();
        let a = resolve_names(
            [
                // (t, t); a < t < a + b
                ExpressionListEntry {
                    expression: &Statement::Expression(AOp {
                        operation: OpName::Point,
                        args: vec![id("t"), id("t")],
                    }),
                    parametric_domain: Domain {
                        min: &id("a"),
                        max: &AOp {
                            operation: OpName::Add,
                            args: vec![id("a"), id("b")],
                        },
                    },
                    slider: None,
                },
                // a = 5
                ExpressionListEntry {
                    expression: &Statement::Assignment {
                        name: "a".into(),
                        value: ANum(5.0),
                    },
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: None,
                },
            ]
            .as_slice()
            .as_ref(),
            &[],
            false,
        );
        assert_eq(
            (a.assignments, a.results.into(), a.freevars),
            (
                vec![
                    // (t, t)
                    Assignment {
                        id: ids.new_id("1"),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Point,
                            args: vec![
                                Expression::Identifier(ids.new_id("t")),
                                Expression::Identifier(ids["t"]),
                            ],
                        },
                    },
                    // a = 5
                    Assignment {
                        id: ids.new_id("a"),
                        name: "a".into(),
                        value: Expression::Number(5.0),
                    },
                    // min = a
                    Assignment {
                        id: ids.new_id("min"),
                        name: "<parametric min>".into(),
                        value: Expression::Identifier(ids["a"]),
                    },
                ],
                vec![
                    // (t, t); a < t < b undefined
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::PARAMETRIC,
                        value: ids["1"],
                        parameters: vec![ids["t"]],
                        domain: Some(Domain {
                            min: Ok(ids["min"]),
                            max: Err(NameError::Undefined(vec!["b".into()])),
                        }),
                    },
                    // a
                    ExpressionResult::Value(ids["a"]),
                ],
                HashMap::from([("t".into(), ids["t"]), ("b".into(), ids.new_id("b"))]),
            ),
        );
    }

    #[test]
    fn slider() {
        let id = |s: &str| AId(s.into());
        let mut ids = IdGenerator::default();
        let a = resolve_names(
            [
                // a = 4; min = b, max = none, step: 1
                ExpressionListEntry {
                    expression: &Statement::Assignment {
                        name: "a".into(),
                        value: ANum(4.0),
                    },
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: Some(Slider {
                        min: Some(&id("b")),
                        max: None,
                        step: Some(&ANum(1.0)),
                    }),
                },
                // b = 3
                ExpressionListEntry {
                    expression: &Statement::Assignment {
                        name: "b".into(),
                        value: ANum(3.0),
                    },
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: None,
                },
                // a with b = 5
                ExpressionListEntry {
                    expression: &Statement::Expression(AWith {
                        body: bx(id("a")),
                        substitutions: vec![("b".into(), ANum(5.0))],
                    }),
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: None,
                },
                // c
                ExpressionListEntry {
                    expression: &Statement::Expression(id("c")),
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: None,
                },
                // c with d = 6
                ExpressionListEntry {
                    expression: &Statement::Expression(AWith {
                        body: bx(id("c")),
                        substitutions: vec![("d".into(), ANum(6.0))],
                    }),
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: None,
                },
                // c = 1; min = none, max = d, step = none
                ExpressionListEntry {
                    expression: &Statement::Assignment {
                        name: "c".into(),
                        value: ANum(1.0),
                    },
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: Some(Slider {
                        min: None,
                        max: Some(&id("d")),
                        step: None,
                    }),
                },
                // d = 2; min = none, max = c, step = none
                ExpressionListEntry {
                    expression: &Statement::Assignment {
                        name: "d".into(),
                        value: ANum(2.0),
                    },
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: Some(Slider {
                        min: None,
                        max: Some(&id("c")),
                        step: None,
                    }),
                },
                // e = 6; min = none, max = f, step = none
                ExpressionListEntry {
                    expression: &Statement::Assignment {
                        name: "e".into(),
                        value: ANum(6.0),
                    },
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: Some(Slider {
                        min: None,
                        max: Some(&id("f")),
                        step: None,
                    }),
                },
                // e with f = 5
                ExpressionListEntry {
                    expression: &Statement::Expression(AWith {
                        body: bx(id("e")),
                        substitutions: vec![("f".into(), ANum(5.0))],
                    }),
                    parametric_domain: Domain::ZERO_TO_ONE,
                    slider: None,
                },
            ]
            .as_slice()
            .as_ref(),
            &[],
            false,
        );
        assert_eq(
            (a.assignments, a.results.into(), a.freevars),
            (
                vec![
                    // b = 3
                    Assignment {
                        id: ids.new_id("b"),
                        name: "b".into(),
                        value: Expression::Number(3.0),
                    },
                    // a.min = b
                    Assignment {
                        id: ids.new_id("a.min"),
                        name: "<slider min>".into(),
                        value: Expression::Identifier(ids["b"]),
                    },
                    // a.step = 1
                    Assignment {
                        id: ids.new_id("a.step"),
                        name: "<slider step>".into(),
                        value: Expression::Number(1.0),
                    },
                    // a = 4; min = b, max = none, step: 1
                    Assignment {
                        id: ids.new_id("a"),
                        name: "a".into(),
                        value: Expression::Slider {
                            value: bx(Expression::Number(4.0)),
                            slider: Slider {
                                min: Some(bx(Expression::Identifier(ids["a.min"]))),
                                max: None,
                                step: Some(bx(Expression::Identifier(ids["a.step"]))),
                            },
                        },
                    },
                    // with b = 5
                    Assignment {
                        id: ids.new_id("b1"),
                        name: "b".into(),
                        value: Expression::Number(5.0),
                    },
                    // a.min = b
                    Assignment {
                        id: ids.new_id("a.min1"),
                        name: "<slider min>".into(),
                        value: Expression::Identifier(ids["b1"]),
                    },
                    // a.step = 1
                    Assignment {
                        id: ids.new_id("a.step1"),
                        name: "<slider step>".into(),
                        value: Expression::Number(1.0),
                    },
                    // a = 4; min = b, max = none, step: 1
                    Assignment {
                        id: ids.new_id("a1"),
                        name: "a".into(),
                        value: Expression::Slider {
                            value: bx(Expression::Number(4.0)),
                            slider: Slider {
                                min: Some(bx(Expression::Identifier(ids["a.min1"]))),
                                max: None,
                                step: Some(bx(Expression::Identifier(ids["a.step1"]))),
                            },
                        },
                    },
                    // a with b = 5
                    Assignment {
                        id: ids.new_id("a with b = 5"),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(ids["a1"]),
                    },
                    // with d = 6
                    Assignment {
                        id: ids.new_id("d"),
                        name: "d".into(),
                        value: Expression::Number(6.0),
                    },
                    // c.max = d
                    Assignment {
                        id: ids.new_id("c.max"),
                        name: "<slider max>".into(),
                        value: Expression::Identifier(ids["d"]),
                    },
                    // c = 1; min = none, max = d, step = none
                    Assignment {
                        id: ids.new_id("c"),
                        name: "c".into(),
                        value: Expression::Slider {
                            value: bx(Expression::Number(1.0)),
                            slider: Slider {
                                min: None,
                                max: Some(bx(Expression::Identifier(ids["c.max"]))),
                                step: None,
                            },
                        },
                    },
                    // c with d = 6
                    Assignment {
                        id: ids.new_id("c with d = 6"),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(ids["c"]),
                    },
                    // e.max = f
                    Assignment {
                        id: ids.new_id("e.max"),
                        name: "<slider max>".into(),
                        value: Expression::Identifier(ids.new_id("f")),
                    },
                    // e = 6; min = none, max = f, step = none
                    Assignment {
                        id: ids.new_id("e"),
                        name: "e".into(),
                        value: Expression::Slider {
                            value: bx(Expression::Number(6.0)),
                            slider: Slider {
                                min: None,
                                max: Some(bx(Expression::Identifier(ids["e.max"]))),
                                step: None,
                            },
                        },
                    },
                    // with f = 5
                    Assignment {
                        id: ids.new_id("f1"),
                        name: "f".into(),
                        value: Expression::Number(5.0),
                    },
                    // e.max = f
                    Assignment {
                        id: ids.new_id("e1.max"),
                        name: "<slider max>".into(),
                        value: Expression::Identifier(ids["f1"]),
                    },
                    // e = 6; min = none, max = f, step = none
                    Assignment {
                        id: ids.new_id("e1"),
                        name: "e".into(),
                        value: Expression::Slider {
                            value: bx(Expression::Number(6.0)),
                            slider: Slider {
                                min: None,
                                max: Some(bx(Expression::Identifier(ids["e1.max"]))),
                                step: None,
                            },
                        },
                    },
                    // e with f = 5
                    Assignment {
                        id: ids.new_id("e with f = 5"),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(ids["e1"]),
                    },
                ],
                vec![
                    // a = 4; min = b, max = none, step: 1
                    ExpressionResult::Slider {
                        value: Some(ids["a"]),
                        slider: Slider {
                            min: Some(Ok(ids["a.min"])),
                            max: None,
                            step: Some(Ok(ids["a.step"])),
                        },
                    },
                    // b = 3
                    ExpressionResult::Value(ids["b"]),
                    // a with b = 5
                    ExpressionResult::Value(ids["a with b = 5"]),
                    // c
                    ExpressionResult::Err(NameError::CyclicDefinition(vec![
                        "c".into(),
                        "d".into(),
                    ])),
                    // c with d = 6
                    ExpressionResult::Value(ids["c with d = 6"]),
                    // c = 1; min = none, max = d, step = none
                    ExpressionResult::Slider {
                        value: None,
                        slider: Slider {
                            min: None,
                            max: Some(Err(NameError::CyclicDefinition(vec![
                                "c".into(),
                                "d".into(),
                            ]))),
                            step: None,
                        },
                    },
                    // d = 2; min = none, max = c, step = none
                    ExpressionResult::Slider {
                        value: None,
                        slider: Slider {
                            min: None,
                            max: Some(Err(NameError::CyclicDefinition(vec![
                                "c".into(),
                                "d".into(),
                            ]))),
                            step: None,
                        },
                    },
                    // e = 6; min = none, max = f, step = none
                    ExpressionResult::Slider {
                        value: None,
                        slider: Slider {
                            min: None,
                            max: Some(Err(NameError::Undefined(vec!["f".into()]))),
                            step: None,
                        },
                    },
                    // e with f = 5
                    ExpressionResult::Value(ids["e with f = 5"]),
                ],
                HashMap::from([("f".into(), ids["f"])]),
            ),
        );
    }

    #[test]
    fn builtins() {
        let id = |s: &str| AId(s.into());
        let mut ids = IdGenerator::default();
        assert_eq_with_builtins(
            resolve_names_ti_with_builtins(
                &[
                    // a = pi + 2
                    ElAssign {
                        name: "a".into(),
                        value: AOp {
                            operation: OpName::Add,
                            args: vec![id("pi"), ANum(2.0)],
                        },
                    },
                    // b = a
                    ElAssign {
                        name: "b".into(),
                        value: id("a"),
                    },
                    // pi = x
                    ElAssign {
                        name: "pi".into(),
                        value: id("x"),
                    },
                    // e = x
                    ElAssign {
                        name: "e".into(),
                        value: id("x"),
                    },
                    // d = x
                    ElAssign {
                        name: "d".into(),
                        value: id("x"),
                    },
                    // c = e
                    ElAssign {
                        name: "c".into(),
                        value: id("e"),
                    },
                    // f(e) = e^2
                    ElFunction {
                        name: "f".into(),
                        parameters: vec!["e".into()],
                        body: AOp {
                            operation: OpName::Pow,
                            args: vec![id("e"), ANum(2.0)],
                        },
                    },
                    // f(3)
                    ElExpr(ACall {
                        callee: "f".into(),
                        args: vec![ANum(3.0)],
                    }),
                ],
                &["pi", "e"],
            ),
            (
                vec![
                    // a = pi + 2
                    Assignment {
                        id: ids.new_id("a"),
                        name: "a".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(ids.new_id("pi")),
                                Expression::Number(2.0),
                            ],
                        },
                    },
                    // b = a
                    Assignment {
                        id: ids.new_id("b"),
                        name: "b".into(),
                        value: Expression::Identifier(ids["a"]),
                    },
                    // pi = x RHS
                    Assignment {
                        id: ids.new_id("pi = x RHS"),
                        name: "pi".into(),
                        value: Expression::Identifier(ids.new_id("x")),
                    },
                    // pi = x
                    Assignment {
                        id: ids.new_id("pi - x"),
                        name: "<implicit plot>".into(),
                        value: Expression::Op {
                            operation: OpName::Sub,
                            args: vec![
                                Expression::Identifier(ids["pi"]),
                                Expression::Identifier(ids["pi = x RHS"]),
                            ],
                        },
                    },
                    // e = x RHS
                    Assignment {
                        id: ids.new_id("e = x RHS"),
                        name: "e".into(),
                        value: Expression::Identifier(ids["x"]),
                    },
                    // e = x
                    Assignment {
                        id: ids.new_id("e - x"),
                        name: "<implicit plot>".into(),
                        value: Expression::Op {
                            operation: OpName::Sub,
                            args: vec![
                                Expression::Identifier(ids.new_id("e")),
                                Expression::Identifier(ids["e = x RHS"]),
                            ],
                        },
                    },
                    // d = x
                    Assignment {
                        id: ids.new_id("d"),
                        name: "d".into(),
                        value: Expression::Identifier(ids["x"]),
                    },
                    // c = e
                    Assignment {
                        id: ids.new_id("c"),
                        name: "c".into(),
                        value: Expression::Identifier(ids["e"]),
                    },
                    // e = anonymous from f(e)
                    Assignment {
                        id: ids.new_id("e1"),
                        name: "e".into(),
                        value: Expression::Identifier(ids.new_id("<anonymous function argument>")),
                    },
                    // f(e) = e^2
                    Assignment {
                        id: ids.new_id("f(e) plot"),
                        name: "<anonymous function plot>".into(),
                        value: Expression::Op {
                            operation: OpName::Pow,
                            args: vec![Expression::Identifier(ids["e1"]), Expression::Number(2.0)],
                        },
                    },
                    // e = 3 from f(e=3)
                    Assignment {
                        id: ids.new_id("e2"),
                        name: "e".into(),
                        value: Expression::Number(3.0),
                    },
                    // f(3)
                    Assignment {
                        id: ids.new_id("f(3)"),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Pow,
                            args: vec![Expression::Identifier(ids["e2"]), Expression::Number(2.0)],
                        },
                    },
                ],
                vec![
                    // a = pi + 2
                    ExpressionResult::Value(ids["a"]),
                    // b = a
                    ExpressionResult::Value(ids["b"]),
                    // pi = x
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::IMPLICIT,
                        value: ids["pi - x"],
                        parameters: vec![ids["x"], ids.new_id("y")],
                        domain: None,
                    },
                    // e = x
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::IMPLICIT,
                        value: ids["e - x"],
                        parameters: vec![ids["x"], ids["y"]],
                        domain: None,
                    },
                    // d = x
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: ids["d"],
                        parameters: vec![ids["x"]],
                        domain: None,
                    },
                    // c = e
                    ExpressionResult::Value(ids["c"]),
                    // f(e) = e^2
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: ids["f(e) plot"],
                        parameters: vec![ids["<anonymous function argument>"]],
                        domain: None,
                    },
                    // f(3)
                    ExpressionResult::Value(ids["f(3)"]),
                ],
                HashMap::from([
                    ("x".into(), ids["x"]),
                    ("y".into(), ids["y"]),
                    (
                        "<anonymous function argument>".into(),
                        ids["<anonymous function argument>"],
                    ),
                ]),
                HashMap::from([("pi".into(), ids["pi"]), ("e".into(), ids["e"])]),
            ),
        );
    }

    // TODO add tests for plot types

    // TODO add test for f(f(f(f(f(x)))))
}
