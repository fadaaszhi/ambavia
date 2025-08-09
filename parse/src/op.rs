use crate::{
    ast, name_resolver,
    type_checker::{self, Type},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signature {
    param_types: &'static [Type],
    return_type: Type,
}

pub(crate) const AST_USE_OP: bool = false;
pub(crate) const NAMERESOLVE_USE_OP: bool = false;
pub(crate) const TYCK_USE_OP: bool = true;

macro_rules! declare_ops {
    (
        $( #[$meta:meta] )*
        // include the literal names of the enums to make searching for the definition easier
        $vis:vis enum Op => {
            $(
                $op:ident( $( $arg_ty:ident ),* ) -> $ret_ty:ident
            ),+ $(,)?
        }
    ) => {
        $( #[$meta] )*
        $vis enum Op {
            $(
                $op
            ),+
        }
        impl Op {
            $vis const fn sig(self) -> Signature {
                match self {
                    $(
                        Op::$op => {
                            Signature {
                                param_types: &[
                                    $(
                                        $arg_ty
                                    ),*
                                ],
                                return_type: $ret_ty
                            }
                        }
                    ),+
                }
            }
        }
    };
}
impl From<ast::BinaryOperator> for OpName {
    fn from(value: ast::BinaryOperator) -> Self {
        use OpName::*;
        match value {
            ast::BinaryOperator::Add => Add,
            ast::BinaryOperator::Sub => Sub,
            ast::BinaryOperator::Mul => Mul,
            ast::BinaryOperator::Div => Div,
            ast::BinaryOperator::Pow => Pow,
            ast::BinaryOperator::Dot => Dot,
            ast::BinaryOperator::Cross => Cross,
            ast::BinaryOperator::Point => Point,
            ast::BinaryOperator::Index => Index,
        }
    }
}
impl From<ast::UnaryOperator> for OpName {
    fn from(value: ast::UnaryOperator) -> Self {
        use OpName::*;
        match value {
            ast::UnaryOperator::Neg => Neg,
            ast::UnaryOperator::Fac => Fac,
            ast::UnaryOperator::Sqrt => Sqrt,
            ast::UnaryOperator::Norm => Norm,
            ast::UnaryOperator::PointX => PointX,
            ast::UnaryOperator::PointY => PointY,
        }
    }
}
impl From<name_resolver::BuiltIn> for OpName {
    fn from(value: name_resolver::BuiltIn) -> Self {
        use OpName::*;
        match value {
            name_resolver::BuiltIn::Ln => Ln,
            name_resolver::BuiltIn::Exp => Exp,
            name_resolver::BuiltIn::Erf => Erf,
            name_resolver::BuiltIn::Sin => Sin,
            name_resolver::BuiltIn::Cos => Cos,
            name_resolver::BuiltIn::Tan => Tan,
            name_resolver::BuiltIn::Sec => Sec,
            name_resolver::BuiltIn::Csc => Csc,
            name_resolver::BuiltIn::Cot => Cot,
            name_resolver::BuiltIn::Sinh => Sinh,
            name_resolver::BuiltIn::Cosh => Cosh,
            name_resolver::BuiltIn::Tanh => Tanh,
            name_resolver::BuiltIn::Sech => Sech,
            name_resolver::BuiltIn::Csch => Csch,
            name_resolver::BuiltIn::Coth => Coth,
            name_resolver::BuiltIn::Asin => Asin,
            name_resolver::BuiltIn::Acos => Acos,
            name_resolver::BuiltIn::Atan => Atan,
            name_resolver::BuiltIn::Asec => Asec,
            name_resolver::BuiltIn::Acsc => Acsc,
            name_resolver::BuiltIn::Acot => Acot,
            name_resolver::BuiltIn::Asinh => Asinh,
            name_resolver::BuiltIn::Acosh => Acosh,
            name_resolver::BuiltIn::Atanh => Atanh,
            name_resolver::BuiltIn::Asech => Asech,
            name_resolver::BuiltIn::Acsch => Acsch,
            name_resolver::BuiltIn::Acoth => Acoth,
            name_resolver::BuiltIn::Abs => Abs,
            name_resolver::BuiltIn::Sgn => Sgn,
            name_resolver::BuiltIn::Round => Round,
            name_resolver::BuiltIn::Floor => Floor,
            name_resolver::BuiltIn::Ceil => Ceil,
            name_resolver::BuiltIn::Mod => Mod,
            name_resolver::BuiltIn::Midpoint => Midpoint,
            name_resolver::BuiltIn::Distance => Distance,
            name_resolver::BuiltIn::Min => Min,
            name_resolver::BuiltIn::Max => Max,
            name_resolver::BuiltIn::Median => Median,
            name_resolver::BuiltIn::Total => Total,
            name_resolver::BuiltIn::Mean => Mean,
            name_resolver::BuiltIn::Count => Count,
            name_resolver::BuiltIn::Unique => Unique,
            name_resolver::BuiltIn::Sort => Sort,
            name_resolver::BuiltIn::Polygon => Polygon,
            name_resolver::BuiltIn::Join => Join,
        }
    }
}
impl From<type_checker::BinaryOperator> for Op {
    fn from(value: type_checker::BinaryOperator) -> Self {
        use Op::*;
        match value {
            type_checker::BinaryOperator::AddNumber => AddNumber,
            type_checker::BinaryOperator::AddPoint => AddPoint,
            type_checker::BinaryOperator::SubNumber => SubNumber,
            type_checker::BinaryOperator::SubPoint => SubPoint,
            type_checker::BinaryOperator::MulNumber => MulNumber,
            type_checker::BinaryOperator::MulNumberPoint => MulNumberPoint,
            type_checker::BinaryOperator::DivNumber => DivNumber,
            type_checker::BinaryOperator::DivPointNumber => DivPointNumber,
            type_checker::BinaryOperator::Pow => Pow,
            type_checker::BinaryOperator::Dot => Dot,
            type_checker::BinaryOperator::Point => Point,
            type_checker::BinaryOperator::IndexNumberList => IndexNumberList,
            type_checker::BinaryOperator::IndexPointList => IndexPointList,
            type_checker::BinaryOperator::IndexPolygonList => IndexPolygonList,
            type_checker::BinaryOperator::FilterNumberList => FilterNumberList,
            type_checker::BinaryOperator::FilterPointList => FilterPointList,
            type_checker::BinaryOperator::FilterPolygonList => FilterPolygonList,
        }
    }
}
impl From<type_checker::UnaryOperator> for Op {
    fn from(value: type_checker::UnaryOperator) -> Self {
        use Op::*;
        match value {
            type_checker::UnaryOperator::NegNumber => NegNumber,
            type_checker::UnaryOperator::NegPoint => NegPoint,
            type_checker::UnaryOperator::Fac => Fac,
            type_checker::UnaryOperator::Sqrt => Sqrt,
            type_checker::UnaryOperator::Abs => Abs,
            type_checker::UnaryOperator::Mag => Mag,
            type_checker::UnaryOperator::PointX => PointX,
            type_checker::UnaryOperator::PointY => PointY,
        }
    }
}
impl From<type_checker::BuiltIn> for Op {
    fn from(value: type_checker::BuiltIn) -> Self {
        use Op::*;
        match value {
            type_checker::BuiltIn::Ln => Ln,
            type_checker::BuiltIn::Exp => Exp,
            type_checker::BuiltIn::Erf => Erf,
            type_checker::BuiltIn::Sin => Sin,
            type_checker::BuiltIn::Cos => Cos,
            type_checker::BuiltIn::Tan => Tan,
            type_checker::BuiltIn::Sec => Sec,
            type_checker::BuiltIn::Csc => Csc,
            type_checker::BuiltIn::Cot => Cot,
            type_checker::BuiltIn::Sinh => Sinh,
            type_checker::BuiltIn::Cosh => Cosh,
            type_checker::BuiltIn::Tanh => Tanh,
            type_checker::BuiltIn::Sech => Sech,
            type_checker::BuiltIn::Csch => Csch,
            type_checker::BuiltIn::Coth => Coth,
            type_checker::BuiltIn::Asin => Asin,
            type_checker::BuiltIn::Acos => Acos,
            type_checker::BuiltIn::Atan => Atan,
            type_checker::BuiltIn::Atan2 => Atan2,
            type_checker::BuiltIn::Asec => Asec,
            type_checker::BuiltIn::Acsc => Acsc,
            type_checker::BuiltIn::Acot => Acot,
            type_checker::BuiltIn::Asinh => Asinh,
            type_checker::BuiltIn::Acosh => Acosh,
            type_checker::BuiltIn::Atanh => Atanh,
            type_checker::BuiltIn::Asech => Asech,
            type_checker::BuiltIn::Acsch => Acsch,
            type_checker::BuiltIn::Acoth => Acoth,
            type_checker::BuiltIn::Abs => Abs,
            type_checker::BuiltIn::Sgn => Sgn,
            type_checker::BuiltIn::Round => Round,
            type_checker::BuiltIn::RoundWithPrecision => RoundWithPrecision,
            type_checker::BuiltIn::Floor => Floor,
            type_checker::BuiltIn::Ceil => Ceil,
            type_checker::BuiltIn::Mod => Mod,
            type_checker::BuiltIn::Midpoint => Midpoint,
            type_checker::BuiltIn::Distance => Distance,
            type_checker::BuiltIn::Min => Min,
            type_checker::BuiltIn::Max => Max,
            type_checker::BuiltIn::Median => Median,
            type_checker::BuiltIn::TotalNumber => TotalNumber,
            type_checker::BuiltIn::TotalPoint => TotalPoint,
            type_checker::BuiltIn::MeanNumber => MeanNumber,
            type_checker::BuiltIn::MeanPoint => MeanPoint,
            type_checker::BuiltIn::CountNumber => CountNumber,
            type_checker::BuiltIn::CountPoint => CountPoint,
            type_checker::BuiltIn::CountPolygon => CountPolygon,
            type_checker::BuiltIn::UniqueNumber => UniqueNumber,
            type_checker::BuiltIn::UniquePoint => UniquePoint,
            type_checker::BuiltIn::UniquePolygon => UniquePolygon,
            type_checker::BuiltIn::Sort => Sort,
            type_checker::BuiltIn::SortKeyNumber => SortKeyNumber,
            type_checker::BuiltIn::SortKeyPoint => SortKeyPoint,
            type_checker::BuiltIn::SortKeyPolygon => SortKeyPolygon,
            type_checker::BuiltIn::Polygon => Polygon,
            type_checker::BuiltIn::JoinNumber => JoinNumber,
            type_checker::BuiltIn::JoinPoint => JoinPoint,
            type_checker::BuiltIn::JoinPolygon => JoinPolygon,
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpName {
    //
    Neg,
    Fac,
    Sqrt,
    Norm,
    PointX,
    PointY,
    //
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Dot,
    Cross,
    Point,
    Index,
    //
    Ln,
    Exp,
    Erf,
    Sin,
    Cos,
    Tan,
    Sec,
    Csc,
    Cot,
    Sinh,
    Cosh,
    Tanh,
    Sech,
    Csch,
    Coth,
    Asin,
    Acos,
    Atan,
    Asec,
    Acsc,
    Acot,
    Asinh,
    Acosh,
    Atanh,
    Asech,
    Acsch,
    Acoth,
    Abs,
    Sgn,
    Round,
    Floor,
    Ceil,
    Mod,
    Midpoint,
    Distance,
    Min,
    Max,
    Median,
    Total,
    Mean,
    Count,
    Unique,
    Sort,
    Polygon,
    Join,
}
use Type::{
    Bool as B, BoolList as BL, Number as N, NumberList as NL, Point as P, PointList as PL,
    Polygon as Pg, PolygonList as PgL,
};
declare_ops! {
    #[derive(Debug, Copy, Clone, Eq, PartialEq)]
    pub enum Op => {
        // special unary
        NegNumber(N) -> N,
        NegPoint(P) -> P,
        Fac(N) -> N,
        Sqrt(N) -> N,
        Mag(P) -> P,
        PointX(P) -> N,
        PointY(P) -> N,

        // binary
        AddNumber(N, N) -> N,
        AddPoint(P, P) -> P,
        SubNumber(N, N) -> N,
        SubPoint(P, P) -> P,
        MulNumber(N, N) -> N,
        MulNumberPoint(N, P) -> P,
        DivNumber(N, N) -> N,
        DivPointNumber(P, N) -> P,
        Pow(N, N) -> N,
        Dot(P, P) -> P,
        Point(N, N) -> P,
        IndexNumberList(NL, N) -> N,
        IndexPointList(PL, N) -> P,
        IndexPolygonList(Pg, N) -> Pg,
        FilterNumberList(BL, NL) -> N,
        FilterPointList(BL, PL) -> P,
        FilterPolygonList(BL, PgL) -> Pg,

        // builtins
        Ln(N) -> N,
        Exp(N) -> N,
        Erf(N) -> N,
        Sin(N) -> N,
        Cos(N) -> N,
        Tan(N) -> N,
        Sec(N) -> N,
        Csc(N) -> N,
        Cot(N) -> N,
        Sinh(N) -> N,
        Cosh(N) -> N,
        Tanh(N) -> N,
        Sech(N) -> N,
        Csch(N) -> N,
        Coth(N) -> N,
        Asin(N) -> N,
        Acos(N) -> N,
        Atan(N) -> N,
        Atan2(N, N) -> N,
        Asec(N) -> N,
        Acsc(N) -> N,
        Acot(N) -> N,
        Asinh(N) -> N,
        Acosh(N) -> N,
        Atanh(N) -> N,
        Asech(N) -> N,
        Acsch(N) -> N,
        Acoth(N) -> N,
        Abs(N) -> N,
        Sgn(N) -> N,
        Round(N) -> N,
        RoundWithPrecision(N, N) -> N,
        Floor(N) -> N,
        Ceil(N) -> N,
        Mod(N, N) -> N,
        Midpoint(P, P) -> P,
        Distance(P, P) -> N,
        Min(NL) -> N,
        Max(NL) -> N,
        Median(NL) -> N,
        TotalNumber(NL) -> N,
        TotalPoint(PL) -> P,
        MeanNumber(NL) -> N,
        MeanPoint(PL) -> P,
        CountNumber(NL) -> N,
        CountPoint(PL) -> N,
        CountPolygon(PgL) -> N,
        UniqueNumber(NL) -> NL,
        UniquePoint(PL) -> PL,
        UniquePolygon(PgL) -> PgL,
        Sort(NL) -> NL,
        SortKeyNumber(NL, NL) -> NL,
        SortKeyPoint(PL, NL) -> PL,
        SortKeyPolygon(PgL, NL) -> PgL,
        Polygon(PL) -> Pg,
        // These have more complicated type signatures than what we can represent (due to potential list of list)
        // and are thus left taking "no" input and handled as a special case
        JoinNumber() -> NL,
        JoinPoint() -> PL,
        JoinPolygon() -> PgL,
    }
}
impl OpName {
    const fn overloads(self) -> &'static [Op] {
        use Op::*;
        match self {
            OpName::Neg => &[NegNumber, NegPoint],
            OpName::Fac => &[Fac],
            OpName::Sqrt => &[Sqrt],
            OpName::Norm => &[Abs, Mag],
            OpName::PointX => &[PointX],
            OpName::PointY => &[PointY],
            OpName::Add => &[AddNumber, AddPoint],
            OpName::Sub => &[SubNumber, SubPoint],
            OpName::Mul => &[MulNumber, MulNumberPoint],
            OpName::Div => &[DivNumber, DivPointNumber],
            OpName::Pow => &[Pow],
            OpName::Dot => &[MulNumber, MulNumberPoint, Dot],
            OpName::Cross => &[MulNumber, MulNumberPoint, MulNumberPoint],
            OpName::Point => &[Point],
            OpName::Index => &[
                IndexNumberList,
                IndexPointList,
                IndexPolygonList,
                FilterNumberList,
                FilterPointList,
                FilterPolygonList,
            ],
            OpName::Ln => &[Ln],
            OpName::Exp => &[Exp],
            OpName::Erf => &[Erf],
            OpName::Sin => &[Sin],
            OpName::Cos => &[Cos],
            OpName::Tan => &[Tan],
            OpName::Sec => &[Sec],
            OpName::Csc => &[Csc],
            OpName::Cot => &[Cot],
            OpName::Sinh => &[Sinh],
            OpName::Cosh => &[Cosh],
            OpName::Tanh => &[Tanh],
            OpName::Sech => &[Sech],
            OpName::Csch => &[Csch],
            OpName::Coth => &[Coth],
            OpName::Asin => &[Asin],
            OpName::Acos => &[Acos],
            OpName::Atan => &[Atan, Atan2],
            OpName::Asec => &[Asec],
            OpName::Acsc => &[Acsc],
            OpName::Acot => &[Acot],
            OpName::Asinh => &[Asinh],
            OpName::Acosh => &[Acosh],
            OpName::Atanh => &[Atanh],
            OpName::Asech => &[Asech],
            OpName::Acsch => &[Acsch],
            OpName::Acoth => &[Acoth],
            OpName::Abs => &[Abs],
            OpName::Sgn => &[Sgn],
            OpName::Round => &[Round, RoundWithPrecision],
            OpName::Floor => &[Floor],
            OpName::Ceil => &[Ceil],
            OpName::Mod => &[Mod],
            OpName::Midpoint => &[Midpoint],
            OpName::Distance => &[Distance],
            OpName::Min => &[Min],
            OpName::Max => &[Max],
            OpName::Median => &[Median],
            OpName::Total => &[TotalNumber, TotalPoint],
            OpName::Mean => &[MeanNumber, MeanPoint],
            OpName::Count => &[CountNumber, CountPoint, CountPolygon],
            OpName::Unique => &[CountNumber, CountPoint, CountPolygon],
            OpName::Sort => &[Sort, SortKeyNumber, SortKeyPoint, SortKeyPolygon],
            OpName::Polygon => &[Polygon],
            OpName::Join => &[],
        }
    }
}

pub(crate) enum ParameterTransform {
    BroadcastOver, // coercions etc
}

pub(crate) struct SignatureSatisfies<T: Iterator<Item = Option<ParameterTransform>>> {
    ret_ty: Type,
    transform: Option<T>,
}

impl Signature {
    fn satisfies(
        self,
        ptypes: &[Type],
    ) -> Option<SignatureSatisfies<impl Iterator<Item = Option<ParameterTransform>>>> {
        // satisfy directly
        if self.param_types == ptypes {
            return Some(SignatureSatisfies {
                ret_ty: self.return_type,
                transform: None,
            });
        }
        // check for brodcast
        for (param, this_param) in ptypes.iter().zip(self.param_types) {
            if param.base() != this_param.base() {
                return None;
            }
        }
        // no list of list
        if self.return_type.is_list() {
            return None;
        }
        // satisfy by broadcasting the required args
        Some(SignatureSatisfies {
            ret_ty: Type::list_of(self.return_type.base()),
            transform: Some(ptypes.iter().zip(self.param_types).map(|(param, this)| {
                if param != this {
                    Some(ParameterTransform::BroadcastOver)
                } else {
                    None
                }
            })),
        })
    }
}

impl OpName {
    pub(crate) fn overload_for(
        self,
        ptypes: &[Type],
    ) -> Result<
        (
            Op,
            SignatureSatisfies<impl Iterator<Item = Option<ParameterTransform>>>,
        ),
        String,
    > {
        self.overloads()
            .iter()
            .copied()
            .filter_map(|op| op.sig().satisfies(ptypes).map(|v| (op, v)))
            .next()
            .ok_or_else(|| {
                let s = if ptypes.len() > 1 {
                    &ptypes[0..(ptypes.len() - 1)]
                } else {
                    ptypes
                };
                format!(
                    "cannot {self:#?} {}{}",
                    s.iter()
                        .map(|a| format!("{a}"))
                        .collect::<Vec<_>>()
                        .join(", "),
                    if ptypes.len() > 1 {
                        format!("and {}", ptypes.last().unwrap())
                    } else {
                        String::new()
                    }
                )
            })
    }
}
