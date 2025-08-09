use crate::type_checker::Type;
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signature {
    param_types: &'static [Type],
    return_type: Type,
}

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

pub enum OpName {
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
    BoolList as BL, Number as N, NumberList as NL, Point as P, PointList as PL, Polygon as Pg,
    PolygonList as PgL,
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
        Polygon(PL) -> Pg
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
            OpName::Cross => &[MulNumber, MulNumberPoint],
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
            OpName::Join => todo!(),
        }
    }
}
