use std::{
    borrow::Borrow,
    collections::{HashMap, HashSet},
    ops::ControlFlow,
};

use derive_more::{From, Into};
use typed_index_collections::{TiSlice, TiVec};

use crate::{
    ast::ExpressionListEntry,
    name_resolver::{
        ExpressionIndex, ExpressionResult as NrEr, Id, PlotKinds, resolve_names,
        undefined_vars_error_msg,
    },
    type_checker::{Assignment, Type, type_check, walk_assignment_ids},
};

#[derive(Debug, Copy, Clone, From, Into, PartialEq)]
pub struct AssignmentIndex(usize);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PlotKind {
    /// `y = f(x)`
    Normal,
    /// `x = f(y)`
    Inverse,
    /// `(x(t), y(t))`
    Parametric,
    /// `f(x, y) = 0`
    Implicit,
}

#[derive(Debug, PartialEq)]
pub enum ExpressionResult {
    None,
    Err(String),
    Value(Id, Type),
    Plot {
        kind: PlotKind,
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
}

pub fn analyze_expression_list(
    list: &TiSlice<ExpressionIndex, impl Borrow<ExpressionListEntry>>,
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
            NrEr::Err(e) => ExpressionResult::Err(e),
            NrEr::Value(id) => match types[&id].clone() {
                Ok(ty) => ExpressionResult::Value(id, ty),
                Err(e) => ExpressionResult::Err(e),
            },
            NrEr::Plot {
                allowed_kinds,
                value,
                parameters,
            } => {
                let ty = match types[&value].clone() {
                    Ok(ty) => ty,
                    Err(e) => return ExpressionResult::Err(e),
                };

                // Figure out plot kind, handling mismatches between expected
                // plot kind and received type
                let kind = match ty {
                    Type::Number | Type::NumberList => {
                        if allowed_kinds.intersects(
                            PlotKinds::NORMAL | PlotKinds::INVERSE | PlotKinds::IMPLICIT,
                        ) {
                            if ty.is_list() {
                                return ExpressionResult::Err(
                                    "todo: plotting lists is not supported yet".into(),
                                );
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
                            return ExpressionResult::Err(
                                undefined_vars_error_msg(
                                    parameters.iter().map(|p| freevar_names[p]),
                                )
                                .unwrap(),
                            );
                        }
                    }
                    _ if ty != Type::EmptyList && allowed_kinds == PlotKinds::IMPLICIT => {
                        // (x,y) = (x,y)
                        return ExpressionResult::Err(format!("cannot compare {} to {}", ty, ty));
                    }
                    Type::Point | Type::PointList => {
                        if allowed_kinds.intersects(PlotKinds::PARAMETRIC) {
                            if ty.is_list() {
                                return ExpressionResult::Err(
                                    "todo: plotting lists is not supported yet".into(),
                                );
                            }
                            if parameters.is_empty() {
                                // y = (3,4)
                                return ExpressionResult::Value(value, ty);
                            }
                            PlotKind::Parametric
                        } else {
                            // f(t) = (t,t)
                            // y = (x,x)
                            return ExpressionResult::None;
                        }
                    }
                    Type::Polygon | Type::PolygonList => {
                        return if allowed_kinds.contains(PlotKinds::PARAMETRIC) {
                            if let Some(e) = undefined_vars_error_msg(
                                parameters.iter().map(|p| freevar_names[p]),
                            ) {
                                // a = polygon([(b,b)])
                                ExpressionResult::Err(e)
                            } else {
                                // x = polygon([0,1,1],[0,0,1])
                                ExpressionResult::Value(value, ty)
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

    AnalysisResult {
        results,
        assignments,
        constants,
    }
}
