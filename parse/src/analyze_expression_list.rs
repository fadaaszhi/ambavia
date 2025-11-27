use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    fmt::Display,
    ops::ControlFlow,
};

use derive_more::{From, Into};
use typed_index_collections::{TiSlice, TiVec};

use crate::{
    name_resolver::{
        Domain, ExpressionIndex, ExpressionListEntry, ExpressionResult as NrEr, Id, NameError,
        PlotKinds, resolve_names,
    },
    type_checker::{Assignment, Type, TypeError, type_check, walk_assignment_ids},
};

#[derive(Debug, Copy, Clone, From, Into, PartialEq)]
pub struct AssignmentIndex(usize);

#[derive(Debug, Clone, PartialEq)]
pub enum PlotKind<T> {
    /// `y = f(x)`
    Normal,
    /// `x = f(y)`
    Inverse,
    /// `(x(t), y(t))`
    Parametric(Domain<T>),
    /// `f(x, y) = 0`
    Implicit,
}

#[derive(Debug, PartialEq)]
pub enum AnalysisError {
    NameError(NameError),
    TypeError(TypeError),
    DomainBoundNotANumber(Type),
    TodoListPlot,
}

impl Display for AnalysisError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AnalysisError::NameError(e) => e.fmt(f),
            AnalysisError::TypeError(e) => e.fmt(f),
            AnalysisError::DomainBoundNotANumber(ty) => {
                write!(f, "domain bound should be {}, not {ty}", Type::Number)
            }
            AnalysisError::TodoListPlot => write!(f, "todo: plotting lists is not implemented yet"),
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum ExpressionResult {
    None,
    Err(AnalysisError),
    Value(Id, Type),
    Plot {
        kind: PlotKind<Result<Id, AnalysisError>>,
        value: Id,
        ty: Type,
        parameters: Vec<Id>,
        assignments: Vec<AssignmentIndex>,
    },
}

pub struct AnalysisResult {
    pub results: TiVec<ExpressionIndex, ExpressionResult>,
    pub assignments: TiVec<AssignmentIndex, Assignment>,
    pub constants: Vec<AssignmentIndex>,
    pub freevars: HashMap<Id, String>,
}

pub fn analyze_expression_list<'a>(
    list: &TiSlice<ExpressionIndex, impl Borrow<ExpressionListEntry<'a>>>,
    use_v1_9_scoping_rules: bool,
) -> AnalysisResult {
    let (assignments, results, freevars) = resolve_names(list, use_v1_9_scoping_rules);
    let (assignments, types) = type_check(&assignments, &freevars);
    let assignments: TiVec<AssignmentIndex, _> = assignments.into();

    // We want to find the list of assignments that represent constants in our
    // expression list. We'll do that by finding the assignments that
    // transitively depend on free variables and then excluding those ones
    let mut freevar_dependencies = freevars
        .values()
        .map(|&id| (id, HashSet::from([id])))
        .collect::<HashMap<_, _>>();
    for assignment in &assignments {
        let mut seen = HashSet::new();
        let mut dependencies = HashSet::new();
        _ = walk_assignment_ids(assignment, &mut |id| {
            if !seen.contains(&id)
                && let Some(other) = freevar_dependencies.get(&id)
            {
                dependencies.extend(other);
                seen.insert(id);
            }
            ControlFlow::Continue::<()>(())
        });
        freevar_dependencies.insert(assignment.id, dependencies);
    }

    let constants = assignments
        .iter_enumerated()
        .filter_map(|(i, a)| freevar_dependencies[&a.id].is_empty().then_some(i))
        .collect();

    let freevar_names = freevars
        .iter()
        .map(|(name, id)| (*id, name.as_str()))
        .collect::<HashMap<_, _>>();

    let results = results
        .into_iter()
        .map(|r| match r {
            NrEr::None => ExpressionResult::None,
            NrEr::Err(e) => ExpressionResult::Err(AnalysisError::NameError(e)),
            NrEr::Value(id) => match types[&id].clone() {
                Ok(ty) => ExpressionResult::Value(id, ty),
                Err(e) => ExpressionResult::Err(AnalysisError::TypeError(e)),
            },
            NrEr::Plot {
                allowed_kinds,
                value,
                parameters,
                domain,
            } => {
                let ty = match types[&value].clone() {
                    Ok(ty) => ty,
                    Err(e) => return ExpressionResult::Err(AnalysisError::TypeError(e)),
                };

                // Figure out plot kind, handling mismatches between expected
                // plot kind and received type
                let kind = match ty {
                    Type::Number | Type::NumberList => {
                        if allowed_kinds.intersects(
                            PlotKinds::NORMAL | PlotKinds::INVERSE | PlotKinds::IMPLICIT,
                        ) {
                            if ty.is_list() {
                                return ExpressionResult::Err(AnalysisError::TodoListPlot);
                            }
                            if allowed_kinds.contains(PlotKinds::NORMAL) {
                                PlotKind::Normal
                            } else if allowed_kinds.contains(PlotKinds::INVERSE) {
                                PlotKind::Inverse
                            } else {
                                PlotKind::Implicit
                            }
                        } else {
                            // a+2
                            return ExpressionResult::Err(AnalysisError::NameError(
                                NameError::undefined(parameters.iter().map(|p| freevar_names[p])),
                            ));
                        }
                    }
                    _ if ty != Type::EmptyList && allowed_kinds == PlotKinds::IMPLICIT => {
                        // (x,y) = (x,y)
                        return ExpressionResult::Err(AnalysisError::TypeError(
                            TypeError::CannotCompare(ty, ty),
                        ));
                    }
                    Type::Point | Type::PointList => {
                        if allowed_kinds.intersects(PlotKinds::PARAMETRIC) {
                            if ty.is_list() {
                                return ExpressionResult::Err(AnalysisError::TodoListPlot);
                            }
                            if parameters.is_empty() {
                                // y = (3,4)
                                return ExpressionResult::Value(value, ty);
                            }
                            let d = domain.unwrap();
                            let f = |m: Result<Id, NameError>| match m {
                                Ok(id) => match types[&id].clone() {
                                    Ok(Type::Number) => Ok(id),
                                    Ok(ty) => Err(AnalysisError::DomainBoundNotANumber(ty)),
                                    Err(e) => Err(AnalysisError::TypeError(e)),
                                },
                                Err(e) => Err(AnalysisError::NameError(e)),
                            };
                            PlotKind::Parametric(Domain {
                                min: f(d.min),
                                max: f(d.max),
                            })
                        } else {
                            // f(t) = (t,t)
                            // y = (x,x)
                            return ExpressionResult::None;
                        }
                    }
                    Type::Polygon | Type::PolygonList => {
                        return if allowed_kinds.contains(PlotKinds::PARAMETRIC) {
                            if parameters.is_empty() {
                                // x = polygon([0,1,1],[0,0,1])
                                ExpressionResult::Value(value, ty)
                            } else {
                                // a = polygon([(b,b)])
                                ExpressionResult::Err(AnalysisError::NameError(
                                    NameError::undefined(
                                        parameters.iter().map(|p| freevar_names[p]),
                                    ),
                                ))
                            }
                        } else {
                            // f(p) = polygon(p)
                            // x = polygon([(y,y)])
                            ExpressionResult::None
                        };
                    }
                    Type::Bool | Type::BoolList => unreachable!(),
                    Type::EmptyList => return ExpressionResult::Value(value, ty),
                };

                // We want to find the assignments necessary to compute this
                // value that depend on our parameter, since constants that we
                // depend on have already been computed and we don't want to
                // unnecessarily compute them again
                let assignments = if parameters
                    .iter()
                    .any(|p| freevar_dependencies[&value].contains(p))
                {
                    let mut value_dependencies = HashSet::from([value]);
                    for assignment in assignments.iter().rev() {
                        if value_dependencies.contains(&assignment.id) {
                            let _ = assignment.value.walk_ids(&mut |id| {
                                if let Some(d) = freevar_dependencies.get(&id)
                                    && parameters.iter().any(|p| d.contains(p))
                                {
                                    value_dependencies.insert(id);
                                }
                                ControlFlow::Continue::<()>(())
                            });
                        }
                    }
                    assignments
                        .iter_enumerated()
                        .filter_map(|(i, a)| value_dependencies.contains(&a.id).then_some(i))
                        .collect()
                } else {
                    vec![]
                };

                ExpressionResult::Plot {
                    kind,
                    value,
                    ty,
                    parameters,
                    assignments,
                }
            }
        })
        .collect();

    let freevars = freevars.into_iter().map(|(k, v)| (v, k)).collect();

    AnalysisResult {
        results,
        assignments,
        constants,
        freevars,
    }
}
