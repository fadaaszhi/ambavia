use std::{borrow::Borrow, collections::HashMap, mem};

use derive_more::{From, Into};
use typed_index_collections::{TiSlice, TiVec, ti_vec};

pub use crate::name_resolver::{ComparisonOperator, SumProdKind};
use crate::{
    name_resolver::{self as nr},
    op::{Op, SigSatisfies},
};

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BaseType {
    Number,
    Point,
    Polygon,
    Bool,
    Empty,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum Type {
    Number,
    NumberList,
    Point,
    PointList,
    Polygon,
    PolygonList,
    Bool,
    BoolList,
    EmptyList,
}

impl Type {
    pub const fn base(self) -> BaseType {
        match self {
            Type::Number | Type::NumberList => BaseType::Number,
            Type::Point | Type::PointList => BaseType::Point,
            Type::Polygon | Type::PolygonList => BaseType::Polygon,
            Type::Bool | Type::BoolList => BaseType::Bool,
            Type::EmptyList => BaseType::Empty,
        }
    }
    pub const fn map_base(self, base: BaseType) -> Self {
        if self.is_list() {
            Self::list_of(base)
        } else {
            Self::single(base)
        }
    }
    pub const fn as_single(self) -> Self {
        Self::single(self.base())
    }
    pub const fn as_list(self) -> Self {
        if self.is_list() {
            self
        } else {
            Self::list_of(self.base())
        }
    }
    pub const fn list_of(base: BaseType) -> Type {
        match base {
            BaseType::Number => Type::NumberList,
            BaseType::Point => Type::PointList,
            BaseType::Polygon => Type::PolygonList,
            BaseType::Bool => Type::BoolList,
            BaseType::Empty => Type::EmptyList,
        }
    }

    pub const fn single(base: BaseType) -> Type {
        match base {
            BaseType::Number => Type::Number,
            BaseType::Point => Type::Point,
            BaseType::Polygon => Type::Polygon,
            BaseType::Bool => Type::Bool,
            BaseType::Empty => Type::Number,
        }
    }

    pub const fn is_list(self) -> bool {
        matches!(
            self,
            Type::NumberList
                | Type::PointList
                | Type::PolygonList
                | Type::BoolList
                | Type::EmptyList
        )
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Type::Number => "a number",
            Type::NumberList => "a list of numbers",
            Type::Point => "a point",
            Type::PointList => "a list of points",
            Type::Polygon => "a polygon",
            Type::PolygonList => "a list of polygons",
            Type::Bool => "a true/false value",
            Type::BoolList => "a list of true/false values",
            Type::EmptyList => "an empty list",
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct TypedExpression {
    pub ty: Type,
    pub e: Expression,
}

fn te(ty: Type, e: Expression) -> TypedExpression {
    TypedExpression { ty, e }
}

#[derive(Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    Identifier(usize),
    List(Vec<TypedExpression>),
    ListRange {
        before_ellipsis: Vec<TypedExpression>,
        after_ellipsis: Vec<TypedExpression>,
    },
    Broadcast {
        scalars: Vec<Assignment>,
        vectors: Vec<Assignment>,
        body: Box<TypedExpression>,
    },
    Op {
        operation: Op,
        args: Vec<TypedExpression>,
    },
    ChainedComparison {
        operands: Vec<TypedExpression>,
        operators: Vec<ComparisonOperator>,
    },
    Piecewise {
        test: Box<TypedExpression>,
        consequent: Box<TypedExpression>,
        alternate: Box<TypedExpression>,
    },
    SumProd {
        kind: SumProdKind,
        variable: usize,
        lower_bound: Box<TypedExpression>,
        upper_bound: Box<TypedExpression>,
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
    pub value: TypedExpression,
}

#[derive(Debug, PartialEq)]
pub struct Body {
    pub assignments: Vec<Assignment>,
    pub value: Box<TypedExpression>,
}

#[derive(Default)]
struct TypeChecker {
    names: HashMap<usize, Result<(Type, usize), String>>,
    id_counter: usize,
}
fn binary(operation: Op, left: TypedExpression, right: TypedExpression) -> Expression {
    Expression::Op {
        operation,
        args: vec![left, right],
    }
}
fn builtin(name: Op, args: Vec<TypedExpression>) -> Expression {
    Expression::Op {
        operation: name,
        args,
    }
}
impl TypeChecker {
    fn next_id(&mut self) -> usize {
        let id = self.id_counter;
        self.id_counter += 1;
        id
    }

    fn look_up(&self, id: impl Borrow<usize>) -> Result<(Type, usize), String> {
        self.names.get(id.borrow()).unwrap().clone()
    }

    fn create_assignment(&mut self, value: TypedExpression) -> Assignment {
        Assignment {
            id: self.next_id(),
            value,
        }
    }

    fn create_name_and_assignment(&mut self, id: usize, value: TypedExpression) -> Assignment {
        let new_id = self.next_id();
        self.names.insert(id, Ok((value.ty, new_id)));
        Assignment { id: new_id, value }
    }

    fn check_body(&mut self, body: &nr::Body) -> Result<Body, String> {
        Ok(Body {
            assignments: body
                .assignments
                .iter()
                .map(|l| {
                    let list = self.check_expression(&l.value)?;
                    Ok(self.create_name_and_assignment(l.id, list))
                })
                .collect::<Result<_, String>>()?,
            value: Box::new(self.check_expression(&body.value)?),
        })
    }

    fn check_expressions(&mut self, es: &[nr::Expression]) -> Result<Vec<TypedExpression>, String> {
        es.iter()
            .map(|e| self.check_expression(e))
            .collect::<Result<Vec<_>, _>>()
    }

    fn check_expression(&mut self, e: &nr::Expression) -> Result<TypedExpression, String> {
        use BaseType as B;
        fn empty_list<U>(b: B) -> Result<TypedExpression, U> {
            Ok(te(Type::list_of(b), Expression::List(vec![])))
        }

        match e {
            nr::Expression::Number(x) => Ok(te(Type::Number, Expression::Number(*x))),
            nr::Expression::Identifier(id) => {
                let (ty, id) = self.look_up(id)?;
                Ok(te(ty, Expression::Identifier(id)))
            }
            nr::Expression::List(list) => {
                let list = self.check_expressions(list)?;

                if list.is_empty() {
                    return Ok(te(Type::EmptyList, Expression::List(vec![])));
                }

                let ty = list[0].ty;

                if ty.is_list() {
                    return Err(format!("cannot store {ty} in a list"));
                }

                for v in &list {
                    if v.ty != ty {
                        return Err("all elements of a list must have the same type".into());
                    }
                }

                Ok(te(Type::list_of(ty.base()), Expression::List(list)))
            }
            nr::Expression::ListRange {
                before_ellipsis,
                after_ellipsis,
            } => {
                let before_ellipsis = self.check_expressions(before_ellipsis)?;
                let after_ellipsis = self.check_expressions(after_ellipsis)?;

                if before_ellipsis.iter().any(|e| e.ty != Type::Number)
                    || after_ellipsis.iter().any(|e| e.ty != Type::Number)
                {
                    return Err("ranges must be arithmetic sequences".into());
                }

                Ok(te(
                    Type::NumberList,
                    Expression::ListRange {
                        before_ellipsis,
                        after_ellipsis,
                    },
                ))
            }
            nr::Expression::ChainedComparison {
                operands,
                operators,
            } => {
                let operands = self.check_expressions(operands)?;

                for w in operands.windows(2) {
                    for o in w {
                        if !matches!(o.ty.base(), B::Number | B::Empty) {
                            return Err(format!("cannot compare {} to {}", w[0].ty, w[1].ty));
                        }
                    }
                }

                if operands.iter().any(|o| o.ty == Type::EmptyList) {
                    return empty_list(B::Bool);
                }

                if operands.iter().any(|o| o.ty.is_list()) {
                    let operands = operands
                        .into_iter()
                        .map(|o| self.create_assignment(o))
                        .collect::<Vec<_>>();
                    let body = Box::new(te(
                        Type::Bool,
                        Expression::ChainedComparison {
                            operands: operands
                                .iter()
                                .map(|o| te(Type::Number, Expression::Identifier(o.id)))
                                .collect(),
                            operators: operators.clone(),
                        },
                    ));
                    let (vectors, scalars) =
                        operands.into_iter().partition(|o| o.value.ty.is_list());
                    Ok(te(
                        Type::BoolList,
                        Expression::Broadcast {
                            scalars,
                            vectors,
                            body,
                        },
                    ))
                } else {
                    Ok(te(
                        Type::Bool,
                        Expression::ChainedComparison {
                            operands,
                            operators: operators.clone(),
                        },
                    ))
                }
            }
            nr::Expression::Piecewise {
                test,
                consequent,
                alternate,
            } => {
                let t = self.check_expression(test)?;
                assert_eq!(t.ty.base(), B::Bool);
                let mut c = self.check_expression(consequent)?;
                let mut a = alternate
                    .as_ref()
                    .map(|a| self.check_expression(a))
                    .transpose()?
                    .unwrap_or_else(|| match c.ty.base() {
                        B::Number => te(Type::Number, Expression::Number(f64::NAN)),
                        B::Point => te(
                            Type::Point,
                            binary(
                                Op::Point,
                                te(Type::Number, Expression::Number(f64::NAN)),
                                te(Type::Number, Expression::Number(f64::NAN)),
                            ),
                        ),
                        B::Polygon => te(
                            Type::Polygon,
                            builtin(
                                Op::Polygon,
                                vec![te(Type::PointList, Expression::List(vec![]))],
                            ),
                        ),
                        B::Bool => te(
                            Type::Bool,
                            Expression::ChainedComparison {
                                operands: vec![
                                    te(Type::Number, Expression::Number(1.0)),
                                    te(Type::Number, Expression::Number(2.0)),
                                ],
                                operators: vec![ComparisonOperator::Equal],
                            },
                        ),
                        B::Empty => te(Type::EmptyList, Expression::List(vec![])),
                    });

                match (c.ty == Type::EmptyList, a.ty == Type::EmptyList) {
                    (true, true) => return empty_list(B::Empty),
                    (true, false) => c.ty = Type::list_of(a.ty.base()),
                    (false, true) => a.ty = Type::list_of(c.ty.base()),
                    (false, false) => (),
                }

                if a.ty.base() != c.ty.base() {
                    return Err(format!(
                        "cannot use {} and {} as the branches in a piecewise, every branch must have the same type",
                        c.ty, a.ty
                    ));
                }

                if !t.ty.is_list() && (c.ty.is_list() == a.ty.is_list()) {
                    return Ok(te(
                        c.ty,
                        Expression::Piecewise {
                            test: Box::new(t),
                            consequent: Box::new(c),
                            alternate: Box::new(a),
                        },
                    ));
                }

                let ty = c.ty.base();
                let t = self.create_assignment(t);
                let c = self.create_assignment(c);
                let a = self.create_assignment(a);
                let body = Box::new(te(
                    Type::single(c.value.ty.base()),
                    Expression::Piecewise {
                        test: Box::new(te(Type::Bool, Expression::Identifier(t.id))),
                        consequent: Box::new(te(Type::single(ty), Expression::Identifier(c.id))),
                        alternate: Box::new(te(Type::single(ty), Expression::Identifier(a.id))),
                    },
                ));
                let (vectors, scalars) = [t, c, a].into_iter().partition(|x| x.value.ty.is_list());
                Ok(te(
                    Type::list_of(ty),
                    Expression::Broadcast {
                        scalars,
                        vectors,
                        body,
                    },
                ))
            }
            nr::Expression::SumProd { .. } => todo!(),
            nr::Expression::For { body, lists } => {
                let lists = lists
                            .iter()
                            .map(|l| {
                                let list = self.check_expression(&l.value)?;

                                if !list.ty.is_list() {
                                    return Err(format!(
                                        "a definition on the right-hand side of 'for' must be a list, but '{}' is {}",
                                        l.name, list.ty
                                    ));
                                }

                                let assignment = self.create_assignment(list);
                                self.names.insert(
                                    l.id,
                                    Ok((Type::single(assignment.value.ty.base()), assignment.id)),
                                );
                                Ok(assignment)
                            })
                            .collect::<Result<_, String>>()?;

                let body = self.check_body(body)?;

                if body.value.ty.is_list() {
                    return Err(format!("cannot store {} in a list", body.value.ty));
                }

                Ok(te(
                    Type::list_of(body.value.ty.base()),
                    Expression::For { body, lists },
                ))
            }
            nr::Expression::Op { operation, args } => {
                use crate::op::OpName;

                let mut checked_args = self.check_expressions(args)?;
                match operation {
                    OpName::Join => {
                        let mut first_non_empty: Option<Type> = None;
                        let mut new_args = vec![];

                        for a in checked_args {
                            if a.ty == Type::EmptyList {
                                continue;
                            }
                            let ty = first_non_empty.get_or_insert(a.ty);
                            if ty.base() != a.ty.base() {
                                return Err(format!("cannot join {ty} and {}", a.ty));
                            }
                            new_args.push(a);
                        }

                        let Some(ty) = first_non_empty else {
                            assert_eq!(new_args, vec![]);
                            return empty_list(B::Empty);
                        };
                        return Ok(te(
                            Type::list_of(ty.base()),
                            Expression::Op {
                                operation: match ty.base() {
                                    B::Number => Op::JoinNumber,
                                    B::Point => Op::JoinPoint,
                                    B::Polygon => Op::JoinPolygon,
                                    _ => unreachable!(),
                                },
                                args: new_args,
                            },
                        ));
                    }
                    OpName::Polygon => {
                        if checked_args.len() == 2
                            && let (Type::NumberList, Type::NumberList)
                            | (Type::NumberList, Type::Number)
                            | (Type::NumberList, Type::EmptyList)
                            | (Type::Number, Type::NumberList)
                            | (Type::Number, Type::EmptyList)
                            | (Type::EmptyList, Type::NumberList)
                            | (Type::EmptyList, Type::Number)
                            | (Type::EmptyList, Type::EmptyList) =
                                (checked_args[0].ty, checked_args[1].ty)
                        {
                            let [x, y] = checked_args.try_into().unwrap();
                            if x.ty == Type::EmptyList || y.ty == Type::EmptyList {
                                return Ok(te(
                                    Type::Polygon,
                                    Expression::Op {
                                        operation: Op::Polygon,
                                        args: vec![te(Type::PointList, Expression::List(vec![]))],
                                    },
                                ));
                            } else {
                                let broadcast_x = x.ty.is_list();
                                let broadcast_y = y.ty.is_list();
                                let x = self.create_assignment(x);
                                let y = self.create_assignment(y);
                                let body = Box::new(te(
                                    Type::Point,
                                    binary(
                                        Op::Point,
                                        te(Type::Number, Expression::Identifier(x.id)),
                                        te(Type::Number, Expression::Identifier(y.id)),
                                    ),
                                ));
                                let (scalars, vectors) = match (broadcast_x, broadcast_y) {
                                    (true, true) => (vec![], vec![x, y]),
                                    (true, false) => (vec![y], vec![x]),
                                    (false, true) => (vec![x], vec![y]),
                                    (false, false) => unreachable!(),
                                };
                                return Ok(te(
                                    Type::Polygon,
                                    Expression::Op {
                                        operation: Op::Polygon,
                                        args: vec![te(
                                            Type::PointList,
                                            Expression::Broadcast {
                                                scalars,
                                                vectors,
                                                body,
                                            },
                                        )],
                                    },
                                ));
                            }
                        } else if checked_args.is_empty() {
                            return Ok(te(
                                Type::Polygon,
                                Expression::Op {
                                    operation: Op::Polygon,
                                    args: vec![te(Type::PointList, Expression::List(vec![]))],
                                },
                            ));
                        }
                    }
                    // handle List[Bool] => Bool ? List : Empty
                    OpName::Index => {
                        let [a, b] = checked_args
                            .first_chunk()
                            .expect("Index should have two arguments");
                        if a.ty.is_list() && b.ty == Type::Bool {
                            let mut it = checked_args.into_iter();
                            let (Some(list), Some(test)) = (it.next(), it.next()) else {
                                unreachable!()
                            };
                            let ty = list.ty;
                            return Ok(te(
                                ty,
                                Expression::Piecewise {
                                    test: Box::new(test),
                                    consequent: Box::new(list),
                                    alternate: Box::new(te(ty, Expression::List(vec![]))),
                                },
                            ));
                        }
                    }
                    // canonicalize (Point, Number) order for MulNumberPoint
                    name if name.overloads().contains(&Op::MulNumberPoint) => {
                        if checked_args.len() == 2
                            && let Some([a, b]) = checked_args.first_chunk_mut()
                            && [a.ty.base(), b.ty.base()] == [B::Point, B::Number]
                        {
                            mem::swap(a, b);
                        }
                    }
                    _ => {}
                }

                // Narrow EmptyList to concrete types for functions that don't
                // map EmptyList to EmptyList
                match operation {
                    OpName::Index
                        if checked_args[0].ty == Type::EmptyList
                            && checked_args[1].ty.base() == B::Number =>
                    {
                        checked_args[0].ty = Type::NumberList;
                    }
                    OpName::Min
                    | OpName::Max
                    | OpName::Median
                    | OpName::Argmin
                    | OpName::Argmax
                    | OpName::Mean
                    | OpName::Count
                    | OpName::Total
                        if checked_args.len() == 1 && checked_args[0].ty == Type::EmptyList =>
                    {
                        checked_args[0].ty = Type::NumberList;
                    }
                    OpName::Polygon
                        if checked_args.len() == 1 && checked_args[0].ty == Type::EmptyList =>
                    {
                        checked_args[0].ty = Type::PointList;
                    }
                    _ => {}
                }
                let (
                    op,
                    SigSatisfies {
                        return_ty,
                        meta,
                        splat,
                    },
                ) = operation.overload_for(checked_args.iter().map(|v| v.ty))?;
                Ok(match meta {
                    // TODO: make it a runtime error if something like total([])=0 isn't handled
                    crate::op::SatisfyMeta::Empty => return empty_list(return_ty.base()),

                    crate::op::SatisfyMeta::NeedsBroadcast(..) => {
                        // we need a "function" because only drop glue can drop partially moved values
                        // (the compiler is not smart enough to see that we move the only Drop type out of meta)
                        let destructure_and_drop = {
                            #[inline]
                            move || {
                                let crate::op::SatisfyMeta::NeedsBroadcast(transform) = meta else {
                                    unreachable!()
                                };
                                (transform, splat)
                                // drop_glue(meta)
                            }
                        };
                        let (transform, splat) = destructure_and_drop();
                        let mut transform = transform.collect::<Vec<_>>().into_iter();

                        let mut inner_args = Vec::new();
                        let (vectors, scalars): (Vec<Assignment>, _) = checked_args
                            .into_iter()
                            .map(|expr| {
                                let a = self.create_assignment(expr);
                                inner_args
                                    .push(te(a.value.ty.as_single(), Expression::Identifier(a.id)));
                                a
                            })
                            .partition(|_| transform.next().expect("invalid transform iter"));
                        if splat {
                            inner_args =
                                vec![te(inner_args[0].ty.as_list(), Expression::List(inner_args))];
                        }
                        te(
                            return_ty,
                            Expression::Broadcast {
                                scalars,
                                vectors,
                                body: Box::new(te(
                                    return_ty.as_single(),
                                    Expression::Op {
                                        operation: op.expect("matched op should exist"),
                                        args: inner_args,
                                    },
                                )),
                            },
                        )
                    }
                    crate::op::SatisfyMeta::ExactMatch => {
                        drop(meta);
                        if splat {
                            checked_args = vec![te(
                                checked_args[0].ty.as_list(),
                                Expression::List(checked_args),
                            )];
                        }
                        te(
                            return_ty,
                            Expression::Op {
                                operation: op.expect("matched op should exist"),
                                args: checked_args,
                            },
                        )
                    }
                })
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone, From, Into, Hash)]
pub struct AssignmentIndex(usize);

pub fn type_check(
    assignments: &TiSlice<nr::AssignmentIndex, nr::Assignment>,
) -> (
    TiVec<AssignmentIndex, Assignment>,
    TiVec<nr::AssignmentIndex, Result<AssignmentIndex, String>>,
) {
    let mut tc = TypeChecker::default();
    let mut result_assignments = ti_vec![];
    let assignment_indices = assignments
        .iter()
        .map(|a| {
            let checked = tc
                .check_expression(&a.value)
                .map(|te| tc.create_assignment(te));
            tc.names.insert(
                a.id,
                checked
                    .as_ref()
                    .map(|c| (c.value.ty, c.id))
                    .map_err(|e| e.clone()),
            );
            checked.map(|a| {
                result_assignments.push(a);
                result_assignments.last_key().unwrap()
            })
        })
        .collect();
    (result_assignments, assignment_indices)
}

#[cfg(test)]
mod tests {
    use crate::op::OpName;

    use super::{
        Assignment as As,
        Expression::{Identifier as Id, Number as Num},
        nr::{
            Assignment as NAs,
            Expression::{Identifier as NId, Number as NNum, Op as NOp},
        },
        *,
    };

    use pretty_assertions::assert_eq;

    fn num(e: Expression) -> TypedExpression {
        te(Type::Number, e)
    }

    fn pt(e: Expression) -> TypedExpression {
        te(Type::Point, e)
    }

    fn type_check_ti(
        assignments: &[nr::Assignment],
    ) -> (Vec<Assignment>, Vec<Result<usize, String>>) {
        let (a, b) = type_check(assignments.as_ref());
        (a.into(), b.into_iter().map(|x| x.map(Into::into)).collect())
    }

    #[test]
    fn shrimple() {
        assert_eq!(
            type_check_ti(&[
                NAs {
                    id: 0,
                    name: "a".into(),
                    value: NNum(5.0)
                },
                NAs {
                    id: 2,
                    name: "b".into(),
                    value: NOp {
                        operation: OpName::Point,
                        args: vec![NNum(3.0), NNum(2.0)]
                    }
                },
                NAs {
                    id: 3,
                    name: "c".into(),
                    value: NOp {
                        operation: OpName::Dot,
                        args: vec![NId(2), NId(0)]
                    }
                },
                NAs {
                    id: 88,
                    name: "d".into(),
                    value: NOp {
                        operation: OpName::Add,
                        args: vec![NId(2), NId(0)]
                    }
                },
                NAs {
                    id: 89,
                    name: "e".into(),
                    value: NId(88)
                },
                NAs {
                    id: 999,
                    name: "f".into(),
                    value: NNum(9.0)
                }
            ]),
            (
                vec![
                    As {
                        id: 0,
                        value: num(Num(5.0))
                    },
                    As {
                        id: 1,
                        value: pt(binary(Op::Point, num(Num(3.0)), num(Num(2.0))))
                    },
                    As {
                        id: 2,
                        value: pt(binary(Op::MulNumberPoint, num(Id(0)), pt(Id(1))))
                    },
                    As {
                        id: 3,
                        value: num(Num(9.0))
                    }
                ],
                vec![
                    Ok(0),
                    Ok(1),
                    Ok(2),
                    Err("cannot Add a point and a number".into()),
                    Err("cannot Add a point and a number".into()),
                    Ok(3)
                ]
            )
        );
    }
}
