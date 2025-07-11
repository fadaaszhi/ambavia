use std::{borrow::Borrow, collections::HashMap, iter::zip};

use derive_more::{From, Into};
use typed_index_collections::{TiSlice, TiVec, ti_vec};

use crate::name_resolver::{self as nr};
pub use crate::name_resolver::{ComparisonOperator, SumProdKind};

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
    pub fn base(&self) -> BaseType {
        match self {
            Type::Number | Type::NumberList => BaseType::Number,
            Type::Point | Type::PointList => BaseType::Point,
            Type::Polygon | Type::PolygonList => BaseType::Polygon,
            Type::Bool | Type::BoolList => BaseType::Bool,
            Type::EmptyList => BaseType::Empty,
        }
    }

    pub fn list_of(base: BaseType) -> Type {
        match base {
            BaseType::Number => Type::NumberList,
            BaseType::Point => Type::PointList,
            BaseType::Polygon => Type::PolygonList,
            BaseType::Bool => Type::BoolList,
            BaseType::Empty => Type::EmptyList,
        }
    }

    pub fn single(base: BaseType) -> Type {
        match base {
            BaseType::Number => Type::Number,
            BaseType::Point => Type::Point,
            BaseType::Polygon => Type::Polygon,
            BaseType::Bool => Type::Bool,
            BaseType::Empty => Type::Number,
        }
    }

    pub fn is_list(&self) -> bool {
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
pub enum UnaryOperator {
    NegNumber,
    NegPoint,
    Fac,
    Sqrt,
    Abs,
    Mag,
    PointX,
    PointY,
}

#[derive(Debug, PartialEq)]
pub enum BinaryOperator {
    AddNumber,
    AddPoint,
    SubNumber,
    SubPoint,
    MulNumber,
    MulPointNumber,
    MulNumberPoint,
    DivNumber,
    DivPointNumber,
    Pow,
    Dot,
    Point,
    IndexNumberList,
    IndexPointList,
    IndexPolygonList,
    FilterNumberList,
    FilterPointList,
    FilterPolygonList,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BuiltIn {
    // Log
    /// ([`Type::Number`]) => [`Type::Number`]
    Ln,
    /// ([`Type::Number`]) => [`Type::Number`]
    Exp,
    /// ([`Type::Number`]) => [`Type::Number`]
    Erf,
    /// ([`Type::Number`]) => [`Type::Number`]
    Sin,
    /// ([`Type::Number`]) => [`Type::Number`]
    Cos,
    /// ([`Type::Number`]) => [`Type::Number`]
    Tan,
    /// ([`Type::Number`]) => [`Type::Number`]
    Sinh,
    /// ([`Type::Number`]) => [`Type::Number`]
    Cosh,
    /// ([`Type::Number`]) => [`Type::Number`]
    Tanh,
    /// ([`Type::Number`]) => [`Type::Number`]
    Asin,
    /// ([`Type::Number`]) => [`Type::Number`]
    Acos,
    /// ([`Type::Number`]) => [`Type::Number`]
    Atan,
    /// ([`Type::Number`], [`Type::Number`]) => [`Type::Number`]
    Atan2,
    /// ([`Type::Number`]) => [`Type::Number`]
    Asinh,
    /// ([`Type::Number`]) => [`Type::Number`]
    Acosh,
    /// ([`Type::Number`]) => [`Type::Number`]
    Atanh,
    /// ([`Type::Number`]) => [`Type::Number`]
    Abs,
    /// ([`Type::Number`]) => [`Type::Number`]
    Sgn,
    /// ([`Type::Number`]) => [`Type::Number`]
    Round,
    /// ([`Type::Number`]) => [`Type::Number`]
    Floor,
    /// ([`Type::Number`]) => [`Type::Number`]
    Ceil,
    /// ([`Type::Number`], [`Type::Number`]) => [`Type::Number`]
    Mod,
    /// ([`Type::Point`], [`Type::Point`]) => [`Type::Point`]
    Midpoint,
    /// ([`Type::Point`], [`Type::Point`]) => [`Type::Number`]
    Distance,
    ///  ([`Type::NumberList`]) => [`Type::Number`]
    Min,
    ///  ([`Type::NumberList`]) => [`Type::Number`]
    Max,
    ///  ([`Type::NumberList`]) => [`Type::Number`]
    Median,
    ///  ([`Type::NumberList`]) => [`Type::Number`]
    TotalNumber,
    ///  ([`Type::PointList`]) => [`Type::Point`]
    TotalPoint,
    ///  ([`Type::NumberList`]) => [`Type::Number`]
    MeanNumber,
    ///  ([`Type::PointList`]) => [`Type::Point`]
    MeanPoint,
    ///  ([`Type::NumberList`]) => [`Type::Number`]
    CountNumber,
    ///  ([`Type::PointList`]) => [`Type::Number`]
    CountPoint,
    ///  ([`Type::PolygonList`]) => [`Type::Number`]
    CountPolygon,
    ///  ([`Type::NumberList`]) => [`Type::NumberList`]
    UniqueNumber,
    ///  ([`Type::PointList`]) => [`Type::PointList`]
    UniquePoint,
    ///  ([`Type::PolygonList`]) => [`Type::PolygonList`]
    UniquePolygon,
    /// ([`Type::NumberList`]) => [`Type::NumberList`]
    Sort,
    /// ([`Type::NumberList`], [`Type::NumberList`]) => [`Type::NumberList`]
    SortKeyNumber,
    /// ([`Type::PointList`], [`Type::NumberList`]) => [`Type::PointList`]
    SortKeyPoint,
    /// ([`Type::PolygonList`], [`Type::NumberList`]) => [`Type::PolygonList`]
    SortKeyPolygon,
    /// ([`Type::PointList`]) => [`Type::Polygon`]
    Polygon,
    ///TODO
    Join,
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
    UnaryOperation {
        operation: UnaryOperator,
        arg: Box<TypedExpression>,
    },
    BinaryOperation {
        operation: BinaryOperator,
        left: Box<TypedExpression>,
        right: Box<TypedExpression>,
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
    BuiltIn {
        name: BuiltIn,
        args: Vec<TypedExpression>,
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
        let empty_list = |b: B| Ok(te(Type::list_of(b), Expression::List(vec![])));

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
            nr::Expression::UnaryOperation { operation, arg } => {
                let arg = self.check_expression(arg)?;

                if arg.ty == Type::EmptyList {
                    return empty_list(match operation {
                        nr::UnaryOperator::Neg => B::Empty,
                        nr::UnaryOperator::Fac
                        | nr::UnaryOperator::Sqrt
                        | nr::UnaryOperator::Norm
                        | nr::UnaryOperator::PointX
                        | nr::UnaryOperator::PointY => B::Number,
                    });
                }

                use UnaryOperator as O;
                let (ty, operation) = match operation {
                    nr::UnaryOperator::Neg => match arg.ty.base() {
                        B::Number => (B::Number, O::NegNumber),
                        B::Point => (B::Point, O::NegPoint),
                        _ => return Err(format!("cannot negate {}", arg.ty)),
                    },
                    nr::UnaryOperator::Fac => match arg.ty.base() {
                        B::Number => (B::Number, O::Fac),
                        _ => return Err(format!("cannot take the factorial of {}", arg.ty)),
                    },
                    nr::UnaryOperator::Sqrt => match arg.ty.base() {
                        B::Number => (B::Number, O::Sqrt),
                        _ => return Err(format!("cannot take the square root of {}", arg.ty)),
                    },
                    nr::UnaryOperator::Norm => match arg.ty.base() {
                        B::Number => (B::Number, O::Abs),
                        B::Point => (B::Number, O::Mag),
                        _ => return Err(format!("cannot take the absolute value of {}", arg.ty)),
                    },
                    nr::UnaryOperator::PointX => match arg.ty.base() {
                        B::Point => (B::Number, O::PointX),
                        _ => return Err(format!("cannot access coordinate '.x' of {}", arg.ty)),
                    },
                    nr::UnaryOperator::PointY => match arg.ty.base() {
                        B::Point => (B::Number, O::PointY),
                        _ => return Err(format!("cannot access coordinate '.y' of {}", arg.ty)),
                    },
                };

                if arg.ty.is_list() {
                    let arg_ty = Type::single(arg.ty.base());
                    let arg = self.create_assignment(arg);
                    let id = arg.id;
                    Ok(te(
                        Type::list_of(ty),
                        Expression::Broadcast {
                            scalars: vec![],
                            vectors: vec![arg],
                            body: Box::new(te(
                                Type::single(ty),
                                Expression::UnaryOperation {
                                    operation,
                                    arg: Box::new(te(arg_ty, Expression::Identifier(id))),
                                },
                            )),
                        },
                    ))
                } else {
                    Ok(te(
                        Type::single(ty),
                        Expression::UnaryOperation {
                            operation,
                            arg: Box::new(arg),
                        },
                    ))
                }
            }
            nr::Expression::BinaryOperation {
                operation,
                left,
                right,
            } => {
                let left = self.check_expression(left)?;
                let right = self.check_expression(right)?;

                let broadcast_left = left.ty.is_list() && *operation != nr::BinaryOperator::Index;
                let broadcast_right = right.ty.is_list();

                use BinaryOperator as O;
                let (ty, operation) = match operation {
                    nr::BinaryOperator::Add => match (left.ty.base(), right.ty.base()) {
                        (B::Number, B::Number) => (B::Number, O::AddNumber),
                        (B::Point, B::Point) => (B::Point, O::AddPoint),
                        (B::Number, B::Empty) | (B::Empty, B::Number) => {
                            return empty_list(B::Number);
                        }
                        (B::Point, B::Empty) | (B::Empty, B::Point) => return empty_list(B::Point),
                        _ => return Err(format!("cannot add {} and {}", left.ty, right.ty)),
                    },
                    nr::BinaryOperator::Sub => match (left.ty.base(), right.ty.base()) {
                        (B::Number, B::Number) => (B::Number, O::SubNumber),
                        (B::Point, B::Point) => (B::Point, O::SubPoint),
                        (B::Number, B::Empty) | (B::Empty, B::Number) => {
                            return empty_list(B::Number);
                        }
                        (B::Point, B::Empty) | (B::Empty, B::Point) => return empty_list(B::Point),
                        _ => return Err(format!("cannot subtract {} from {}", left.ty, right.ty)),
                    },
                    nr::BinaryOperator::Mul => match (left.ty.base(), right.ty.base()) {
                        (B::Number, B::Number) => (B::Number, O::MulNumber),
                        (B::Number, B::Point) => (B::Point, O::MulNumberPoint),
                        (B::Point, B::Number) => (B::Point, O::MulPointNumber),
                        (B::Number, B::Empty) | (B::Empty, B::Number) => {
                            return empty_list(B::Empty);
                        }
                        (B::Point, B::Empty) | (B::Empty, B::Point) => return empty_list(B::Point),
                        _ => return Err(format!("cannot multiply {} by {}", left.ty, right.ty)),
                    },
                    nr::BinaryOperator::Div => match (left.ty.base(), right.ty.base()) {
                        (B::Number, B::Number) => (B::Number, O::DivNumber),
                        (B::Point, B::Number) => (B::Point, O::DivPointNumber),
                        (B::Number, B::Empty) => return empty_list(B::Number),
                        (B::Empty, B::Number) => return empty_list(B::Empty),
                        (B::Point, B::Empty) => return empty_list(B::Point),
                        _ => return Err(format!("cannot divide {} by {}", left.ty, right.ty)),
                    },
                    nr::BinaryOperator::Pow => match (left.ty.base(), right.ty.base()) {
                        (B::Number, B::Number) => (B::Number, O::Pow),
                        (B::Number, B::Empty) | (B::Empty, B::Number) => {
                            return empty_list(B::Number);
                        }
                        _ => return Err(format!("cannot raise {} to {}", left.ty, right.ty)),
                    },
                    nr::BinaryOperator::Dot => match (left.ty.base(), right.ty.base()) {
                        (B::Number, B::Number) => (B::Number, O::MulNumber),
                        (B::Number, B::Point) => (B::Point, O::MulNumberPoint),
                        (B::Point, B::Number) => (B::Point, O::MulPointNumber),
                        (B::Point, B::Point) => (B::Number, O::Dot),
                        (B::Number, B::Empty)
                        | (B::Empty, B::Number)
                        | (B::Point, B::Empty)
                        | (B::Empty, B::Point) => return empty_list(B::Empty),
                        _ => return Err(format!("cannot multiply {} by {}", left.ty, right.ty)),
                    },
                    nr::BinaryOperator::Cross => match (left.ty.base(), right.ty.base()) {
                        (B::Number, B::Number) => (B::Number, O::MulNumber),
                        (B::Number, B::Point) => (B::Point, O::MulNumberPoint),
                        (B::Point, B::Number) => (B::Point, O::MulPointNumber),
                        (B::Number, B::Empty) | (B::Empty, B::Number) => {
                            return empty_list(B::Empty);
                        }
                        (B::Point, B::Empty) | (B::Empty, B::Point) => return empty_list(B::Point),
                        (B::Point, B::Point) => {
                            return Err(format!(
                                "cannot take the cross product of {} and {}",
                                left.ty, right.ty,
                            ));
                        }
                        _ => return Err(format!("cannot multiply {} by {}", left.ty, right.ty)),
                    },
                    nr::BinaryOperator::Point => match (left.ty.base(), right.ty.base()) {
                        (B::Number, B::Number) => (B::Point, O::Point),
                        (B::Number, B::Empty) | (B::Empty, B::Number) => {
                            return empty_list(B::Point);
                        }
                        _ => {
                            return Err(format!(
                                "cannot use {} and {} as the coordinates of a point",
                                left.ty, right.ty
                            ));
                        }
                    },
                    nr::BinaryOperator::Index => match (left.ty, right.ty.base()) {
                        (Type::NumberList, B::Number) => (B::Number, O::IndexNumberList),
                        (Type::NumberList, B::Empty) => return empty_list(B::Number),
                        (Type::PointList, B::Number) => (B::Point, O::IndexPointList),
                        (Type::PointList, B::Empty) => return empty_list(B::Point),
                        (Type::PolygonList, B::Number) => (B::Polygon, O::IndexPolygonList),
                        (Type::PolygonList, B::Empty) => return empty_list(B::Polygon),
                        (Type::EmptyList, B::Number) => (B::Number, O::IndexNumberList),
                        (Type::EmptyList, B::Bool) => return empty_list(B::Empty),
                        (Type::EmptyList, B::Empty) => return empty_list(B::Empty),
                        (
                            ty @ (Type::NumberList | Type::PointList | Type::PolygonList),
                            B::Bool,
                        ) => {
                            return Ok(if !right.ty.is_list() {
                                te(
                                    ty,
                                    Expression::Piecewise {
                                        test: Box::new(right),
                                        consequent: Box::new(left),
                                        alternate: Box::new(te(ty, Expression::List(vec![]))),
                                    },
                                )
                            } else {
                                te(
                                    ty,
                                    Expression::BinaryOperation {
                                        operation: match ty {
                                            Type::NumberList => O::FilterNumberList,
                                            Type::PointList => O::FilterPointList,
                                            Type::PolygonList => O::FilterPolygonList,
                                            _ => unreachable!(),
                                        },
                                        left: Box::new(left),
                                        right: Box::new(right),
                                    },
                                )
                            });
                        }
                        _ => return Err(format!("cannot index {} with {}", left.ty, right.ty)),
                    },
                };

                if broadcast_left || broadcast_right {
                    let left_ty = if broadcast_left {
                        Type::single(left.ty.base())
                    } else {
                        left.ty
                    };
                    let right_ty = if broadcast_right {
                        Type::single(right.ty.base())
                    } else {
                        right.ty
                    };
                    let left = self.create_assignment(left);
                    let right = self.create_assignment(right);
                    let body = Box::new(te(
                        Type::single(ty),
                        Expression::BinaryOperation {
                            operation,
                            left: Box::new(te(left_ty, Expression::Identifier(left.id))),
                            right: Box::new(te(right_ty, Expression::Identifier(right.id))),
                        },
                    ));
                    let (scalars, vectors) = match (broadcast_left, broadcast_right) {
                        (true, true) => (vec![], vec![left, right]),
                        (true, false) => (vec![right], vec![left]),
                        (false, true) => (vec![left], vec![right]),
                        (false, false) => unreachable!(),
                    };
                    Ok(te(
                        Type::list_of(ty),
                        Expression::Broadcast {
                            scalars,
                            vectors,
                            body,
                        },
                    ))
                } else {
                    Ok(te(
                        Type::single(ty),
                        Expression::BinaryOperation {
                            operation,
                            left: Box::new(left),
                            right: Box::new(right),
                        },
                    ))
                }
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
                            Expression::BinaryOperation {
                                operation: BinaryOperator::Point,
                                left: Box::new(te(Type::Number, Expression::Number(f64::NAN))),
                                right: Box::new(te(Type::Number, Expression::Number(f64::NAN))),
                            },
                        ),
                        B::Polygon => te(
                            Type::Polygon,
                            Expression::BuiltIn {
                                name: BuiltIn::Polygon,
                                args: vec![te(Type::PointList, Expression::List(vec![]))],
                            },
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
            nr::Expression::BuiltIn { name, args } => {
                use BuiltIn as Bi;
                use Type::{
                    EmptyList as EL, Number as N, NumberList as NL, Point as P, PointList as PL,
                    Polygon as Pg, PolygonList as PgL,
                };
                use nr::BuiltIn as Nb;
                let args = self.check_expressions(args)?;

                if args.len() == 1 && args[0].ty == Type::EmptyList {
                    match name {
                        Nb::Min | Nb::Max | Nb::Median | Nb::Mean => {
                            return Ok(te(N, Expression::Number(f64::NAN)));
                        }
                        Nb::Count | Nb::Total => return Ok(te(N, Expression::Number(0.0))),
                        Nb::Unique => return empty_list(B::Empty),
                        Nb::Sort => return empty_list(B::Number),
                        Nb::Polygon => {
                            return Ok(te(
                                Pg,
                                Expression::BuiltIn {
                                    name: Bi::Polygon,
                                    args: vec![te(PL, Expression::List(vec![]))],
                                },
                            ));
                        }
                        _ => {}
                    }
                }

                if *name == Nb::Polygon
                    && args.len() == 2
                    && let (Type::NumberList, Type::NumberList)
                    | (Type::NumberList, Type::Number)
                    | (Type::NumberList, Type::EmptyList)
                    | (Type::Number, Type::NumberList)
                    | (Type::Number, Type::EmptyList)
                    | (Type::EmptyList, Type::NumberList)
                    | (Type::EmptyList, Type::Number)
                    | (Type::EmptyList, Type::EmptyList) = (args[0].ty, args[1].ty)
                {
                    let [x, y] = args.try_into().unwrap();
                    if x.ty == Type::EmptyList || y.ty == Type::EmptyList {
                        return Ok(te(
                            Type::Polygon,
                            Expression::BuiltIn {
                                name: Bi::Polygon,
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
                            Expression::BinaryOperation {
                                operation: BinaryOperator::Point,
                                left: Box::new(te(Type::Number, Expression::Identifier(x.id))),
                                right: Box::new(te(Type::Number, Expression::Identifier(y.id))),
                            },
                        ));
                        let (scalars, vectors) = match (broadcast_x, broadcast_y) {
                            (true, true) => (vec![], vec![x, y]),
                            (true, false) => (vec![y], vec![x]),
                            (false, true) => (vec![x], vec![y]),
                            (false, false) => unreachable!(),
                        };
                        return Ok(te(
                            Type::Polygon,
                            Expression::BuiltIn {
                                name: BuiltIn::Polygon,
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
                }

                if *name == Nb::Sort && args.len() == 2 {
                    match (args[0].ty, args[1].ty) {
                        (EL, EL | NL) => return empty_list(B::Empty),
                        (l @ (NL | PL | PgL), EL) => return empty_list(l.base()),
                        _ => {}
                    }
                }

                if *name == Nb::Join {
                    let mut first_non_empty: Option<Type> = None;
                    let mut new_args = vec![];

                    for a in args {
                        if a.ty == EL {
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
                    if new_args.iter().all(|a| !a.ty.is_list()) {
                        return Ok(te(Type::list_of(ty.base()), Expression::List(new_args)));
                    }
                    if new_args.len() == 1 {
                        let a = new_args.pop().unwrap();
                        assert!(a.ty.is_list());
                        return Ok(a);
                    }
                    return Ok(te(
                        Type::list_of(ty.base()),
                        Expression::BuiltIn {
                            name: Bi::Join,
                            args: new_args,
                        },
                    ));
                }

                let overloads: &[(&[Type], Type, BuiltIn)] = match name {
                    Nb::Ln => &[(&[N], N, Bi::Ln)],
                    Nb::Exp => &[(&[N], N, Bi::Exp)],
                    Nb::Erf => &[(&[N], N, Bi::Erf)],
                    Nb::Sin => &[(&[N], N, Bi::Sin)],
                    Nb::Cos => &[(&[N], N, Bi::Cos)],
                    Nb::Tan => &[(&[N], N, Bi::Tan)],
                    Nb::Sec => todo!(),
                    Nb::Csc => todo!(),
                    Nb::Cot => todo!(),
                    Nb::Sinh => &[(&[N], N, Bi::Sinh)],
                    Nb::Cosh => &[(&[N], N, Bi::Cosh)],
                    Nb::Tanh => &[(&[N], N, Bi::Tanh)],
                    Nb::Sech => todo!(),
                    Nb::Csch => todo!(),
                    Nb::Coth => todo!(),
                    Nb::Asin => &[(&[N], N, Bi::Asin)],
                    Nb::Acos => &[(&[N], N, Bi::Acos)],
                    Nb::Atan => &[(&[N], N, Bi::Atan), (&[N, N], N, Bi::Atan2)],
                    Nb::Asec => todo!(),
                    Nb::Acsc => todo!(),
                    Nb::Acot => todo!(),
                    Nb::Asinh => &[(&[N], N, Bi::Asinh)],
                    Nb::Acosh => &[(&[N], N, Bi::Acosh)],
                    Nb::Atanh => &[(&[N], N, Bi::Atanh)],
                    Nb::Asech => todo!(),
                    Nb::Acsch => todo!(),
                    Nb::Acoth => todo!(),
                    Nb::Abs => &[(&[N], N, Bi::Abs)],
                    Nb::Sgn => &[(&[N], N, Bi::Sgn)],
                    Nb::Round => &[(&[N], N, Bi::Round)],
                    Nb::Floor => &[(&[N], N, Bi::Floor)],
                    Nb::Ceil => &[(&[N], N, Bi::Ceil)],
                    Nb::Mod => &[(&[N, N], N, Bi::Mod)],
                    Nb::Midpoint => &[(&[P, P], P, Bi::Midpoint)],
                    Nb::Distance => &[(&[P, P], N, Bi::Distance)],
                    Nb::Min => &[(&[NL], N, Bi::Min)],
                    Nb::Max => &[(&[NL], N, Bi::Max)],
                    Nb::Median => &[(&[NL], N, Bi::Median)],
                    Nb::Total => &[(&[NL], N, Bi::TotalNumber), (&[PL], P, Bi::TotalPoint)],
                    Nb::Mean => &[(&[NL], N, Bi::MeanNumber), (&[PL], P, Bi::MeanPoint)],
                    Nb::Count => &[
                        (&[NL], N, Bi::CountNumber),
                        (&[PL], N, Bi::CountPoint),
                        (&[PgL], N, Bi::CountPolygon),
                    ],
                    Nb::Unique => &[
                        (&[NL], NL, Bi::UniqueNumber),
                        (&[PL], PL, Bi::UniquePoint),
                        (&[PgL], PgL, Bi::UniquePolygon),
                    ],
                    Nb::Sort => &[
                        (&[NL], NL, Bi::Sort),
                        (&[NL, NL], NL, Bi::SortKeyNumber),
                        (&[PL, NL], PL, Bi::SortKeyPoint),
                        (&[PgL, NL], PgL, Bi::SortKeyPolygon),
                    ],
                    Nb::Polygon => &[(&[PL], Pg, Bi::Polygon)],
                    Nb::Join => unreachable!(),
                };

                'overload: for (arg_tys, ret_ty, builtin) in overloads {
                    'normal: {
                        if arg_tys.len() != args.len() {
                            break 'normal;
                        }

                        let broadcastable = !ret_ty.is_list();
                        let mut do_broadcast = false;
                        let mut empty = false;
                        for (a, t) in zip(&args, *arg_tys) {
                            if a.ty == *t {
                                continue;
                            }
                            if broadcastable && !t.is_list() {
                                if a.ty == Type::EmptyList {
                                    empty = true;
                                    continue;
                                }
                                if Type::single(a.ty.base()) == *t {
                                    do_broadcast = true;
                                    continue;
                                }
                            }
                            break 'normal;
                        }

                        if empty {
                            return empty_list(B::Empty);
                        }

                        if do_broadcast {
                            let assignments = args
                                .into_iter()
                                .map(|a| self.create_assignment(a))
                                .collect::<Vec<_>>();
                            let body = Box::new(te(
                                *ret_ty,
                                Expression::BuiltIn {
                                    name: *builtin,
                                    args: assignments
                                        .iter()
                                        .map(|a| te(Type::Point, Expression::Identifier(a.id)))
                                        .collect(),
                                },
                            ));
                            let mut vectors = vec![];
                            let mut scalars = vec![];
                            for (a, t) in zip(assignments, *arg_tys) {
                                if a.value.ty == *t {
                                    scalars.push(a);
                                } else {
                                    vectors.push(a);
                                }
                            }
                            return Ok(te(
                                Type::list_of(ret_ty.base()),
                                Expression::Broadcast {
                                    scalars,
                                    vectors,
                                    body,
                                },
                            ));
                        } else {
                            return Ok(te(
                                *ret_ty,
                                Expression::BuiltIn {
                                    name: *builtin,
                                    args,
                                },
                            ));
                        }
                    }

                    // splat
                    if !ret_ty.is_list()
                        && arg_tys.len() == 1
                        && let t = arg_tys[0]
                        && t.is_list()
                    {
                        let b = Type::single(t.base());
                        let mut do_broadcast = false;
                        let mut empty = false;
                        for a in &args {
                            if a.ty == b {
                                continue;
                            }
                            if a.ty == Type::EmptyList {
                                empty = true;
                                continue;
                            }
                            if Type::single(a.ty.base()) == b {
                                do_broadcast = true;
                                continue;
                            }
                            continue 'overload;
                        }

                        if empty {
                            return empty_list(B::Empty);
                        }

                        if do_broadcast {
                            let assignments = args
                                .into_iter()
                                .map(|a| self.create_assignment(a))
                                .collect::<Vec<_>>();
                            let body = Box::new(te(
                                *ret_ty,
                                Expression::BuiltIn {
                                    name: *builtin,
                                    args: vec![te(
                                        t,
                                        Expression::List(
                                            assignments
                                                .iter()
                                                .map(|a| te(b, Expression::Identifier(a.id)))
                                                .collect(),
                                        ),
                                    )],
                                },
                            ));
                            let (vectors, scalars) =
                                assignments.into_iter().partition(|a| a.value.ty.is_list());
                            return Ok(te(
                                Type::list_of(ret_ty.base()),
                                Expression::Broadcast {
                                    scalars,
                                    vectors,
                                    body,
                                },
                            ));
                        } else {
                            return Ok(te(
                                *ret_ty,
                                Expression::BuiltIn {
                                    name: *builtin,
                                    args: vec![te(t, Expression::List(args))],
                                },
                            ));
                        }
                    }
                }

                Err(format!(
                    "function '{name:?}' cannot be applied to these arguments"
                ))
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
    use super::{
        Assignment as As, BinaryOperator as Bo,
        Expression::{BinaryOperation as Bop, Identifier as Id, Number as Num},
        nr::{
            Assignment as NAs, BinaryOperator as NBo,
            Expression::{BinaryOperation as NBop, Identifier as NId, Number as NNum},
        },
        *,
    };

    use pretty_assertions::assert_eq;

    fn bx<T>(x: T) -> Box<T> {
        Box::new(x)
    }

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
                    value: NBop {
                        operation: NBo::Point,
                        left: bx(NNum(3.0)),
                        right: bx(NNum(2.0))
                    }
                },
                NAs {
                    id: 3,
                    name: "c".into(),
                    value: NBop {
                        operation: NBo::Dot,
                        left: bx(NId(2)),
                        right: bx(NId(0))
                    }
                },
                NAs {
                    id: 88,
                    name: "d".into(),
                    value: NBop {
                        operation: NBo::Add,
                        left: bx(NId(2)),
                        right: bx(NId(0))
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
                        value: pt(Bop {
                            operation: Bo::Point,
                            left: bx(num(Num(3.0))),
                            right: bx(num(Num(2.0)))
                        })
                    },
                    As {
                        id: 2,
                        value: pt(Bop {
                            operation: Bo::MulPointNumber,
                            left: bx(pt(Id(1))),
                            right: bx(num(Id(0)))
                        })
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
                    Err("cannot add a point and a number".into()),
                    Err("cannot add a point and a number".into()),
                    Ok(3)
                ]
            )
        );
    }
}
