use std::{
    array,
    borrow::Borrow,
    collections::HashMap,
    fmt::Display,
    iter::zip,
    ops::{Deref, DerefMut},
    sync::OnceLock,
};

use derive_more::{From, Into};
use typed_index_collections::{TiSlice, TiVec, ti_vec};

pub use crate::ast::{ComparisonOperator, SumProdKind};
use crate::{
    ast::{self, ExpressionListEntry},
    op::OpName,
};

#[derive(Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    Identifier(Id),
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

#[derive(Debug, Default, Clone)]
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
    computed: HashMap<&'a str, (Result<Id, NameError>, Dependencies<'a>)>,
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
    definitions: HashMap<&'a str, Result<&'a ExpressionListEntry, NameError>>,
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
        list: &'a TiSlice<ExpressionIndex, impl Borrow<ExpressionListEntry>>,
        use_v1_9_scoping_rules: bool,
    ) -> Self {
        let mut definitions = HashMap::new();

        for entry in list {
            match entry.borrow() {
                ExpressionListEntry::Assignment { name, .. }
                | ExpressionListEntry::FunctionDeclaration { name, .. }
                    if name != "x" && name != "y" =>
                {
                    if let Some(result) = definitions.get_mut(name.as_str()) {
                        *result = Err(NameError::MultipleDefinitions(name.into()));
                    } else {
                        definitions.insert(name.as_str(), Ok(entry.borrow()));
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
            if let Some((id, deps)) = scope.computed.get(name) {
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
        let (id, deps) = if let Some(entry) = self.definitions.get(name) {
            let expr = match entry.as_ref().map_err(Clone::clone)? {
                ExpressionListEntry::Assignment { value, .. } => value,
                ExpressionListEntry::FunctionDeclaration { .. } => {
                    return Err(NameError::FunctionAsVariable(name.into()));
                }
                _ => unreachable!(),
            };

            self.cycle_detector.push(name)?;
            self.line_count += 1;
            let (value, deps) = self.resolve_expression_with_dependencies(expr, None);
            self.line_count -= 1;
            self.cycle_detector.pop();
            let level = deps.level();
            let id = value.map(|value| self.push_assignment(name, level, value));
            (id, deps)
        } else {
            (Ok(self.create_new_freevar(name)), Dependencies::default())
        };

        let _existing = self.scopes[deps.scope_index()]
            .computed
            .insert(name, (id.clone(), deps));
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
        kind: ScopeKind,
        bindings: impl Iterator<Item = (&'a String, &'a ast::Expression)>,
        error: impl FnOnce(String) -> NameError,
    ) -> Result<Expression, NameError> {
        let mut substitutions = HashMap::new();

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

        let (body, _) =
            self.resolve_expression_with_dependencies(body, Some((kind, substitutions)));
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
            Some(Ok(ExpressionListEntry::FunctionDeclaration {
                parameters, body, ..
            })) => (parameters, body),
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
        self.line_count += 1;
        let kind = if self.use_v1_9_scoping_rules {
            ScopeKind::Dynamic
        } else {
            ScopeKind::Lexical {
                line_count: self.line_count,
            }
        };
        let value = self.resolve_substitutions(
            body,
            kind,
            zip(parameters, args),
            NameError::DuplicateFunctionParameter,
        );
        self.line_count -= 1;
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
                ScopeKind::Dynamic,
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
            NameError::BadPointDimension => write!(f, "points may only have 2 coordinates"),
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

#[derive(Debug, PartialEq)]
pub enum ExpressionResult {
    None,
    Err(NameError),
    Value(Id),
    Plot {
        allowed_kinds: PlotKinds,
        value: Id,
        parameters: Vec<Id>,
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

pub fn resolve_names(
    list: &TiSlice<ExpressionIndex, impl Borrow<ExpressionListEntry>>,
    use_v1_9_scoping_rules: bool,
) -> (
    Vec<Assignment>,
    TiVec<ExpressionIndex, ExpressionResult>,
    HashMap<String, Id>,
) {
    let mut resolver = Resolver::new(list, use_v1_9_scoping_rules);

    let results = list
        .iter()
        .map(|e| {
            // When we start resolving a new expression, there shouldn't be any
            // variables that are in scope from a `for` or `with` clause
            assert_eq!(resolver.line_count, 0);
            assert_eq!(resolver.assignments.len(), 1);
            assert_eq!(resolver.scopes.len(), 1);
            assert_eq!(resolver.scopes[0].substitutions, HashMap::new());
            assert_eq!(resolver.cycle_detector.stack, Vec::<&str>::new());
            assert!(resolver.cycle_detector.counts.values().all(|&c| c == 0));

            match e.borrow() {
                ExpressionListEntry::Assignment { name, value } => {
                    let (id, deps) = if !resolver.freevars.contains_key(name.as_str())
                        && let Some((id, deps)) = resolver.scopes[0].computed.get(name.as_str())
                    {
                        (id.clone(), deps.clone())
                    } else {
                        resolver
                            .cycle_detector
                            .push(name)
                            .expect("can't have a cycle before you even begin");
                        let (value, deps) =
                            resolver.resolve_expression_with_dependencies(value, None);
                        resolver.cycle_detector.pop();
                        let level = deps.level();
                        assert_eq!(level, Level(0));
                        let id = value.map(|value| resolver.push_assignment(name, level, value));
                        if let Some(Ok(_)) = resolver.definitions.get(name.as_str()) {
                            resolver.scopes[0]
                                .computed
                                .insert(name, (id.clone(), deps.clone()));
                        }
                        (id, deps)
                    };
                    let mut freevars = deps
                        .keys()
                        .cloned()
                        .filter(|name| resolver.freevars.contains_key(name))
                        .collect::<Vec<_>>();
                    freevars.sort();
                    match id {
                        Ok(id) => {
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
                                }
                            } else if freevars.len() == 1 && freevars != [name]
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
                                }
                            } else if (name == "x" || name == "y")
                                && matches!(freevars[..], ["x"] | ["y"] | ["x", "y"])
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
                                }
                            } else if freevars.is_empty() {
                                ExpressionResult::Value(id)
                            } else {
                                ExpressionResult::Err(NameError::undefined(freevars))
                            }
                        }
                        Err(e) => ExpressionResult::Err(e),
                    }
                }
                ExpressionListEntry::FunctionDeclaration {
                    name,
                    parameters,
                    body,
                    ..
                } => {
                    if let [parameter] = parameters.as_slice() {
                        let arg = "<anonymous function argument>";
                        let (value, deps) = resolver.resolve_with_dependencies(
                            |resolver| {
                                static ARG: OnceLock<ast::Expression> = OnceLock::new();
                                resolver.cycle_detector.push(name).unwrap();
                                resolver.line_count += 1;
                                let kind = if resolver.use_v1_9_scoping_rules {
                                    ScopeKind::Dynamic
                                } else {
                                    ScopeKind::Lexical {
                                        line_count: resolver.line_count,
                                    }
                                };
                                let value = resolver.resolve_substitutions(
                                    body,
                                    kind,
                                    zip(
                                        parameters,
                                        [ARG.get_or_init(|| {
                                            ast::Expression::Identifier(arg.into())
                                        })],
                                    ),
                                    NameError::DuplicateFunctionParameter,
                                );
                                resolver.line_count -= 1;
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
                                },
                                _ => ExpressionResult::Err(NameError::undefined(freevars)),
                            },
                            Err(e) => ExpressionResult::Err(e),
                        }
                    } else {
                        ExpressionResult::None
                    }
                }
                ExpressionListEntry::Relation(ast::ChainedComparison {
                    operands,
                    operators,
                }) => {
                    let [operator] = &operators[..] else {
                        return ExpressionResult::Err(NameError::TodoChainedRelation);
                    };
                    if *operator != ast::ComparisonOperator::Equal {
                        return ExpressionResult::Err(NameError::TodoInequality);
                    }
                    let (value, deps) = resolver.resolve_with_dependencies(
                        |this| {
                            Ok(Expression::Op {
                                operation: OpName::Sub,
                                args: this.resolve_expressions(operands)?,
                            })
                        },
                        None,
                    );
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
                        Ok(id) => {
                            if matches!(freevars[..], ["x"] | ["y"] | ["x", "y"]) {
                                ExpressionResult::Plot {
                                    allowed_kinds: PlotKinds::IMPLICIT,
                                    value: id,
                                    parameters: vec![
                                        resolver.resolve_variable("x").unwrap(),
                                        resolver.resolve_variable("y").unwrap(),
                                    ],
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
                ExpressionListEntry::Expression(value) => {
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
                            },
                            [v] if v != "y" => ExpressionResult::Plot {
                                allowed_kinds: PlotKinds::PARAMETRIC,
                                value: id,
                                parameters: vec![resolver.freevars[v]],
                            },
                            ["y"] => ExpressionResult::Err(NameError::ExpressionWithFreeVariablY),
                            _ => ExpressionResult::Err(NameError::undefined(freevars)),
                        },
                        Err(e) => ExpressionResult::Err(e),
                    }
                }
            }
        })
        .collect();

    let assignments = resolver.assignments.pop().unwrap();
    assert!(resolver.assignments.is_empty());
    let freevars = resolver
        .freevars
        .iter()
        .map(|(&k, &v)| (k.into(), v))
        .collect();
    (assignments, results, freevars)
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
    ) -> (Vec<Assignment>, Vec<ExpressionResult>, HashMap<String, Id>) {
        let (a, b, c) = resolve_names(list.as_ref(), false);
        (a, b.into(), c)
    }

    fn resolve_names_ti_v1_9(
        list: &[ExpressionListEntry],
    ) -> (Vec<Assignment>, Vec<ExpressionResult>, HashMap<String, Id>) {
        let (a, b, c) = resolve_names(list.as_ref(), true);
        (a, b.into(), c)
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
        assert_eq!(
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
                        parameters: vec![Id(0)]
                    },
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)]
                    },
                    ExpressionResult::Value(Id(15)),
                ],
                HashMap::from([("a2".into(), Id(0)), ("a4".into(), Id(4))]),
            )
        );
    }

    #[test]
    fn function_v1_10() {
        // https://www.desmos.com/calculator/1jougp3ykk
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
                    // freevar a2: 0,
                    // b = a2
                    Assignment {
                        id: Id(1),
                        name: "b".into(),
                        value: Expression::Identifier(Id(0)),
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
                        parameters: vec![Id(0)]
                    },
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)]
                    },
                    ExpressionResult::Value(Id(14)),
                ],
                HashMap::from([("a2".into(), Id(0)), ("a4".into(), Id(4))]),
            )
        );
    }

    #[test]
    fn wackscope() {
        assert_eq!(
            resolve_names_ti(&[
                // f(a) = b + b
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a".into()],
                    body: AOp {
                        operation: OpName::Add,
                        args: vec![AId("b".into()), AId("b".into())]
                    }
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into())
                },
                // f(1)
                ElExpr(ACallMul {
                    callee: "f".into(),
                    args: vec![ANum(1.0)]
                })
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
                                Expression::Identifier(Id(2))
                            ]
                        }
                    },
                    // freevar a: 4
                    // b = a
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    // a = 1
                    Assignment {
                        id: Id(6),
                        name: "a".into(),
                        value: Expression::Number(1.0)
                    },
                    // b = a
                    Assignment {
                        id: Id(7),
                        name: "b".into(),
                        value: Expression::Identifier(Id(6))
                    },
                    // f(1)
                    Assignment {
                        id: Id(8),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(7)),
                                Expression::Identifier(Id(7))
                            ]
                        }
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(3),
                        parameters: vec![Id(0)]
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)]
                    },
                    ExpressionResult::Value(Id(8))
                ],
                HashMap::from([
                    ("<anonymous function argument>".into(), Id(0)),
                    ("a".into(), Id(4))
                ]),
            )
        );
    }

    #[test]
    fn more_function_v1_10() {
        assert_eq!(
            resolve_names_ti(&[
                // f(a) = b
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a".into()],
                    body: AId("b".into())
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into())
                },
                // b + f(1) with a = 2
                ElExpr(AWith {
                    body: bx(AOp {
                        operation: OpName::Add,
                        args: vec![
                            AId("b".into()),
                            ACallMul {
                                callee: "f".into(),
                                args: vec![ANum(1.0)]
                            }
                        ]
                    }),
                    substitutions: vec![("a".into(), ANum(2.0))]
                })
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
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                    // with a = 2
                    Assignment {
                        id: Id(6),
                        name: "a".into(),
                        value: Expression::Number(2.0)
                    },
                    // b = a
                    Assignment {
                        id: Id(7),
                        name: "b".into(),
                        value: Expression::Identifier(Id(6))
                    },
                    // a = 1
                    Assignment {
                        id: Id(8),
                        name: "a".into(),
                        value: Expression::Number(1.0)
                    },
                    // b + f(1) with a = 2
                    Assignment {
                        id: Id(9),
                        name: "<anonymous>".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(7)),
                                Expression::Identifier(Id(7))
                            ]
                        }
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(3),
                        parameters: vec![Id(0)]
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)]
                    },
                    ExpressionResult::Value(Id(9))
                ],
                HashMap::from([
                    ("<anonymous function argument>".into(), Id(0)),
                    ("a".into(), Id(4))
                ]),
            )
        );
    }

    #[test]
    fn even_more_function_v1_10() {
        assert_eq!(
            resolve_names_ti(&[
                // f(a) = b + b
                ElFunction {
                    name: "f".into(),
                    parameters: vec!["a".into()],
                    body: AOp {
                        operation: OpName::Add,
                        args: vec![AId("b".into()), AId("b".into())]
                    }
                },
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into())
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
                                        args: vec![ANum(1.0)]
                                    }
                                ]
                            },
                            AId("b".into()),
                        ]
                    }
                },
                // g(2)
                ElExpr(ACallMul {
                    callee: "g".into(),
                    args: vec![ANum(2.0)]
                })
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
                                Expression::Identifier(Id(2))
                            ]
                        }
                    },
                    // freevar a: 4
                    // b = a
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Identifier(Id(4)),
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
                        value: Expression::Number(1.0)
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
                                                Expression::Identifier(Id(9))
                                            ]
                                        }
                                    ]
                                },
                                Expression::Identifier(Id(7)),
                            ]
                        },
                    },
                    // a = 2
                    Assignment {
                        id: Id(11),
                        name: "a".into(),
                        value: Expression::Number(2.0)
                    },
                    // b = a
                    Assignment {
                        id: Id(12),
                        name: "b".into(),
                        value: Expression::Identifier(Id(11))
                    },
                    // a = 1
                    Assignment {
                        id: Id(13),
                        name: "a".into(),
                        value: Expression::Number(1.0)
                    },
                    // b = a
                    Assignment {
                        id: Id(14),
                        name: "b".into(),
                        value: Expression::Identifier(Id(13))
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
                                                Expression::Identifier(Id(14))
                                            ]
                                        }
                                    ]
                                },
                                Expression::Identifier(Id(12))
                            ]
                        }
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(3),
                        parameters: vec![Id(0)]
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)]
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL,
                        value: Id(10),
                        parameters: vec![Id(0)]
                    },
                    ExpressionResult::Value(Id(15))
                ],
                HashMap::from([
                    ("<anonymous function argument>".into(), Id(0)),
                    ("a".into(), Id(4))
                ]),
            )
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
                                                Expression::Identifier(Id(1))
                                            ]
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
                                                        Expression::Identifier(Id(4))
                                                    ]
                                                }
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
                                    value: Expression::Identifier(Id(0))
                                },
                                // i=[1]
                                Assignment {
                                    id: Id(2),
                                    name: "i".into(),
                                    value: Expression::List(vec![Expression::Number(1.0)])
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
                                Expression::Identifier(Id(7))
                            ]
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
                                        Expression::Identifier(Id(4))
                                    ]
                                }
                            ],
                        },
                    },
                ],
                vec![
                    ExpressionResult::Value(Id(6)),
                    ExpressionResult::Err(NameError::undefined(["i", "j"])),
                    ExpressionResult::Value(Id(0)),
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(8),
                        parameters: vec![Id(7)]
                    },
                    ExpressionResult::Value(Id(4)),
                ],
                HashMap::from([("j".into(), Id(7)), ("i".into(), Id(9))]),
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
                                                Expression::Number(2.0)
                                            ]
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
                                                                Expression::Number(2.0)
                                                            ]
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
                                                                Expression::Identifier(Id(0))
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
                                                                Expression::Identifier(Id(6))
                                                            ],
                                                        },
                                                        Expression::Identifier(Id(7))
                                                    ]
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
                                            args: vec![Expression::Identifier(Id(3))]
                                        },
                                        Expression::Op {
                                            operation: OpName::Total,
                                            args: vec![Expression::Identifier(Id(8))]
                                        }
                                    ]
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
                                                Expression::Number(2.0)
                                            ]
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
                                                Expression::Identifier(Id(12))
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
                                                Expression::Identifier(Id(6))
                                            ],
                                        },
                                        Expression::Identifier(Id(13))
                                    ]
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
                    // freevar i: 15
                    // B = i^2
                    Assignment {
                        id: Id(16),
                        name: "B".into(),
                        value: Expression::Op {
                            operation: OpName::Pow,
                            args: vec![Expression::Identifier(Id(15)), Expression::Number(2.0)]
                        },
                    },
                    // F = i + j
                    Assignment {
                        id: Id(17),
                        name: "F".into(),
                        value: Expression::Op {
                            operation: OpName::Add,
                            args: vec![
                                Expression::Identifier(Id(15)),
                                Expression::Identifier(Id(12))
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
                        parameters: vec![Id(12)]
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(16),
                        parameters: vec![Id(15)]
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
        assert_eq!(
            resolve_names_ti(&[
                // b + c with a = 1
                ElExpr(AWith {
                    body: bx(AOp {
                        operation: OpName::Add,
                        args: vec![AId("b".into()), AId("c".into()),]
                    }),
                    substitutions: vec![("a".into(), ANum(1.0))],
                }),
                // b = a
                ElAssign {
                    name: "b".into(),
                    value: AId("a".into())
                }
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
                                Expression::Identifier(Id(2))
                            ]
                        },
                    },
                    // freevar a: 4
                    // b = a
                    Assignment {
                        id: Id(5),
                        name: "b".into(),
                        value: Expression::Identifier(Id(4)),
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::PARAMETRIC,
                        value: Id(3),
                        parameters: vec![Id(2)]
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(5),
                        parameters: vec![Id(4)]
                    },
                ],
                HashMap::from([("c".into(), Id(2)), ("a".into(), Id(4))]),
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
                vec![
                    // with b = 1
                    Assignment {
                        id: Id(0),
                        name: "b".into(),
                        value: Expression::Number(1.0)
                    },
                    // freevar c: 1
                    // a = c with b = 1
                    Assignment {
                        id: Id(2),
                        name: "a".into(),
                        value: Expression::Identifier(Id(1))
                    },
                    // a
                    Assignment {
                        id: Id(3),
                        name: "<anonymous>".into(),
                        value: Expression::Identifier(Id(2))
                    },
                ],
                vec![
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::NORMAL | PlotKinds::PARAMETRIC,
                        value: Id(2),
                        parameters: vec![Id(1)]
                    },
                    ExpressionResult::Plot {
                        allowed_kinds: PlotKinds::PARAMETRIC,
                        value: Id(3),
                        parameters: vec![Id(1)]
                    },
                ],
                HashMap::from([("c".into(), Id(1))]),
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
                        parameters: vec![Id(0)]
                    },
                    ExpressionResult::Value(Id(2)),
                    ExpressionResult::Value(Id(3)),
                    ExpressionResult::Value(Id(6))
                ],
                HashMap::from([("<anonymous function argument>".into(), Id(0))]),
            )
        );
    }
}
