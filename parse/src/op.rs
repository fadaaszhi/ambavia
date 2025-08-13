use std::iter::{once, repeat, zip};

use crate::{
    ast, name_resolver,
    type_checker::{self, BaseType, Type},
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signature {
    pub param_types: &'static [Type],
    pub return_type: Type,
}

pub(crate) const AST_USE_OP: bool = true;
pub(crate) const NAMERESOLVE_USE_OP: bool = true;
pub(crate) const TYCK_USE_OP: bool = true;

macro_rules! declare_ops {
    (
        $( #[$meta:meta] )*
        // include the literal name of the enum to make searching for the definition easier
        $vis:vis enum Op => {
            $(
                $op:ident( $( $arg_ty:ident ),* ) -> $ret_ty:ident
            ),+ $(,)?
        }
    ) => {
        $( #[$meta] )*
        $vis enum Op {
            $(
                #[doc = concat!( "(", stringify!( $( [ $arg_ty ] ),*), ") => [", stringify!( $ret_ty ), "]") ]
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
        Dot(P, P) -> N,
        Point(N, N) -> P,
        IndexNumberList(NL, N) -> N,
        IndexPointList(PL, N) -> P,
        IndexPolygonList(Pg, N) -> Pg,
        FilterNumberList(NL, BL) -> NL,
        FilterPointList(PL, BL) -> PL,
        FilterPolygonList(PgL, BL) -> PgL,

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
    pub(crate) const fn overloads(self) -> &'static [Op] {
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
            OpName::Join => &[],
        }
    }
}

pub(crate) enum ParameterSatisfaction {
    Exact,
    NeedsBroadcast,
    FromEmpty,
}
impl ParameterSatisfaction {
    fn try_satisfy(supplied: Type, desired: Type) -> Option<Self> {
        // EmptyList "coerces" to any parameter type
        if supplied == Type::EmptyList {
            return Some(Self::FromEmpty);
        }
        match (supplied, desired) {
            (s, d) if s == d => Some(Self::Exact),
            (s, d) if Type::single(s.base()) == d => Some(Self::NeedsBroadcast),
            _ => None,
        }
    }
    fn needs_broadcast(self) -> bool {
        matches!(self, Self::NeedsBroadcast)
    }
}

impl Signature {
    fn satisfies(
        self,
        ptypes: impl Iterator<Item = Type> + Clone,
    ) -> Option<SigSatisfies<impl Iterator<Item = CoercionMeta> + Clone>> {
        // accumulate length of the supplied parameter list
        let mut len = 0usize;
        // Either Some(BaseType) if all supplied params have the same base type or None
        let mut all_supplied_base_ty = None;
        // first supplied type, if any
        let mut first_supplied = None;

        let is_splat_candidate = self.param_types.len() == 1
            && self.param_types.first().unwrap().is_list()
            && !self.return_type.is_list();

        let mut needs_broadcast = false;
        let mut num_empty_coercions = 0;
        let mut handle_satisfaction = |param| match param {
            ParameterSatisfaction::Exact => {}
            ParameterSatisfaction::NeedsBroadcast => needs_broadcast = true,
            ParameterSatisfaction::FromEmpty => {
                num_empty_coercions += 1;
            }
        };
        // scan the supplied parameter list
        for (candidate, desired) in zip(
            ptypes.clone(),
            self.param_types
                .iter()
                .copied()
                .map(Some)
                .chain(repeat(None)),
        ) {
            let candidate_base = candidate.base();
            len += 1;
            if len == 1 {
                all_supplied_base_ty = Some(candidate_base);
                first_supplied = Some(candidate);
                // skip the first parameter for now
                continue;
            } else if candidate_base != BaseType::Empty {
                if Some(BaseType::Empty) == all_supplied_base_ty {
                    all_supplied_base_ty = Some(candidate_base);
                } else if Some(candidate_base) != all_supplied_base_ty {
                    all_supplied_base_ty = None;
                }
            }
            // early return if we can't determine the (likely) desired type
            let actual_desired =
                desired.or(is_splat_candidate.then_some(self.param_types[0].as_single()))?;

            // early return if we can't satisfy this parameter
            let this_param = ParameterSatisfaction::try_satisfy(candidate, actual_desired)?;
            handle_satisfaction(this_param);
        }
        // functions that map (List) -> Scalar get special calling semantics when called with patterns of arguments
        // eg min(1,2,3) should produce Op::Min { args: [ List [N1, N2, N3] ] } instead of Op::Min { args: [ N1, N2, N3 ] }
        // further, this effect can be broadcast e.g min(1, [2,3]) produces
        // Broadcast { scalars: Assign A1 N1, vectors: Assign A2 [N2, N3], body: Op::Min {args: [ List [ A1, A2 ] ] } }
        // instead of Op::Min { args: [ N1, List [ N2, N3 ] ] }
        let needs_splat = is_splat_candidate
            && all_supplied_base_ty
                .is_some_and(|all_supplied_base_ty| all_supplied_base_ty == self.param_types[0].base())
            // first supplied type exists
            && first_supplied.is_some_and(|t|
                // it is scalar (non splat is list)
                !t.is_list()
                // ...or there is more than one supplied argument (non splat is 1)
                || len > 1
            );
        if len < self.param_types.len() {
            return None;
        }
        if self.param_types.len() > 0 {
            let first = first_supplied.expect("first not present but the len is > 0");
            // now that we know for sure whether this is a splat or not we can choose the correct desired type to test against
            handle_satisfaction(ParameterSatisfaction::try_satisfy(
                first,
                needs_splat
                    .then_some(self.param_types[0].as_single())
                    .unwrap_or(self.param_types[0]),
            )?);
        }
        Some(if num_empty_coercions > 0 {
            SigSatisfies {
                return_ty: if num_empty_coercions == len {
                    Type::EmptyList
                } else {
                    self.return_type
                },
                meta: SatisfyMeta::Empty,
            }
        } else if needs_broadcast {
            if self.return_type.is_list() {
                // no list of list
                return None;
            } else {
                SigSatisfies {
                    return_ty: Type::list_of(self.return_type.base()),
                    meta: SatisfyMeta::PartialMatch(
                        zip(
                            ptypes,
                            self.param_types
                                .iter()
                                .copied()
                                .map(Some)
                                .chain(repeat(None)),
                        )
                        .map(move |(param_ty, desired_ty)| {
                            ParameterSatisfaction::try_satisfy(
                                param_ty,
                                needs_splat
                                    .then_some(self.param_types[0].as_single())
                                    .or(desired_ty)
                                    .expect("desired type should be present"),
                            )
                            .expect(
                                "tried to check coercion for parameter but failed after precheck",
                            )
                            .needs_broadcast()
                        }),
                        needs_splat,
                    ),
                }
            }
        } else {
            SigSatisfies {
                return_ty: self.return_type,
                meta: SatisfyMeta::ExactMatch(needs_splat),
            }
        })
    }
}
#[derive(Debug, PartialEq)]
pub(crate) struct SigSatisfies<T> {
    pub(crate) return_ty: Type,
    pub(crate) meta: SatisfyMeta<T>,
}
#[derive(Debug, PartialEq)]
pub(crate) enum SatisfyMeta<T> {
    Empty,
    PartialMatch(T, bool),
    ExactMatch(bool),
}
pub(crate) type CoercionMeta = bool;

impl<T: Iterator<Item = CoercionMeta> + Clone> SigSatisfies<T> {
    fn unify(this: (Op, Self), other: (Option<Op>, Self)) -> Result<(Option<Op>, Self), String> {
        let this_op = Some(this.0);
        let other_op = other.0;
        let (this, other) = (this.1, other.1);
        match (this.meta, other.meta) {
            (SatisfyMeta::Empty, SatisfyMeta::Empty) => {
                Ok(if this.return_ty.as_list() == other.return_ty.as_list() {
                    (
                        None,
                        SigSatisfies {
                            return_ty: this.return_ty.as_list(),
                            meta: SatisfyMeta::Empty,
                        },
                    )
                } else {
                    (
                        None,
                        SigSatisfies {
                            return_ty: Type::EmptyList,
                            meta: SatisfyMeta::Empty,
                        },
                    )
                })
            }
            _ => Err(format!(
                "[internal] failed to unify ambiguous overloads {this_op:#?} and {other_op:#?}"
            )),
        }
    }
}

impl OpName {
    pub(crate) fn overload_for(
        self,
        mut ptypes: impl DoubleEndedIterator<Item = Type> + Clone,
    ) -> Result<
        (
            Option<Op>,
            SigSatisfies<impl Iterator<Item = CoercionMeta> + Clone>,
        ),
        String,
    > {
        self.overloads()
            .iter()
            .copied()
            .try_fold(None, |prev, op| {
                let this = op.sig().satisfies(ptypes.clone()).map(|v| (op, v));
                Ok(match (this, prev) {
                    (None, None) => None,
                    (None, Some(a)) => Some(a),
                    (Some((op, s)), None) => Some((Some(op), s)),
                    (Some(this), Some(prev)) => Some(SigSatisfies::unify(this, prev)?),
                })
            })
            .and_then(|v| {
                v.ok_or_else(|| {
                    let Some(first) = ptypes.next() else {
                        return format!("cannot {self:#?} nothing");
                    };
                    let last = ptypes.next_back();

                    let i = once(first).chain(ptypes);
                    format!(
                        "cannot {self:#?} {}{}",
                        i.map(|a| format!("{a}")).collect::<Vec<_>>().join(", "),
                        if let Some(last) = last {
                            &format!(" and {last}")
                        } else {
                            ""
                        }
                    )
                })
            })
    }
}

#[cfg(test)]
mod tests {
    impl<T> SatisfyMeta<T> {
        fn is_splat(&self) -> bool {
            match self {
                SatisfyMeta::Empty => false,
                SatisfyMeta::PartialMatch(_, s) => *s,
                SatisfyMeta::ExactMatch(s) => *s,
            }
        }
    }
    impl<T> SigSatisfies<T> {
        fn map_iter<U>(self, f: impl FnOnce(T) -> U) -> SigSatisfies<U> {
            let SigSatisfies { return_ty, meta } = self;
            let meta = match meta {
                SatisfyMeta::PartialMatch(iter, s) => SatisfyMeta::PartialMatch(f(iter), s),
                SatisfyMeta::Empty => SatisfyMeta::Empty,
                SatisfyMeta::ExactMatch(s) => SatisfyMeta::ExactMatch(s),
            };
            SigSatisfies { return_ty, meta }
        }
    }
    use crate::{
        op::{CoercionMeta, Op, OpName, SatisfyMeta, SigSatisfies},
        type_checker::Type,
    };

    type MatchedOverload = (Option<Op>, SigSatisfies<Vec<CoercionMeta>>);

    fn try_get_overload(op: OpName, req: &[Type]) -> Result<MatchedOverload, String> {
        op.overload_for(req.iter().copied())
            .map(|v| (v.0, v.1.map_iter(|a| a.collect())))
    }
    #[track_caller]
    fn get(op: OpName, req: &[Type]) -> MatchedOverload {
        try_get_overload(op, req).expect("expected a match")
    }
    #[track_caller]
    fn empty_list_of(v: &MatchedOverload, of: Type) {
        assert_eq!(
            &v.1,
            &SigSatisfies {
                return_ty: of,
                meta: crate::op::SatisfyMeta::Empty
            }
        )
    }
    #[track_caller]
    fn exact_match(v: &MatchedOverload, resolved: Op, splat: bool) {
        assert_eq!(
            v,
            &(
                Some(resolved),
                SigSatisfies {
                    return_ty: resolved.sig().return_type,
                    meta: crate::op::SatisfyMeta::ExactMatch(splat)
                }
            )
        );
    }
    #[track_caller]
    fn has_op(v: &MatchedOverload, op: Op) {
        assert_eq!(v.0, Some(op))
    }
    #[track_caller]
    fn splat(v: &MatchedOverload) {
        assert!(v.1.meta.is_splat())
    }
    #[track_caller]
    fn has_coercions(v: &MatchedOverload, broadcast: &[CoercionMeta]) {
        match &v.1.meta {
            super::SatisfyMeta::PartialMatch(c, _) => assert_eq!(c, broadcast),
            super::SatisfyMeta::Empty | super::SatisfyMeta::ExactMatch(_) => {
                panic!("expected a list of coercions")
            }
        }
    }
    use OpName::*;
    use Type as Ty;
    use Type::{EmptyList, Number, NumberList};
    #[test]
    fn shrimple_ops() {
        exact_match(&get(Add, &[Number, Number]), Op::AddNumber, false);
        exact_match(&get(Mul, &[Number, Ty::Point]), Op::MulNumberPoint, false);

        empty_list_of(&get(Add, &[Number, EmptyList]), NumberList);
        empty_list_of(&get(Add, &[EmptyList, Number]), NumberList);
        empty_list_of(&get(Add, &[EmptyList, EmptyList]), EmptyList);
        empty_list_of(&get(Min, &[Number, EmptyList]), NumberList);
        empty_list_of(&get(Min, &[EmptyList, EmptyList]), EmptyList);
        empty_list_of(&get(Min, &[EmptyList, Type::NumberList]), NumberList);

        let s = get(Min, &[NumberList, Number]);
        splat(&s);
        has_coercions(&s, &[true, false]);
        has_op(&s, Op::Min);

        let s = get(Min, &[Number, NumberList]);
        splat(&s);
        has_coercions(&s, &[false, true]);
        has_op(&s, Op::Min);

        let s = get(Min, &[EmptyList, EmptyList, EmptyList, EmptyList, Number]);
        empty_list_of(&s, NumberList);
        let s = get(Min, &[EmptyList, EmptyList, EmptyList, EmptyList]);
        empty_list_of(&s, EmptyList);
        let s = get(Index, &[EmptyList, Type::BoolList]);
        empty_list_of(&s, EmptyList);

        let s = get(Min, &[NumberList]);
        exact_match(&s, Op::Min, false);
        let s = get(Min, &[Number]);
        splat(&s);
        exact_match(&s, Op::Min, true);
        let s = get(Index, &[EmptyList, EmptyList]);
        empty_list_of(&s, EmptyList);
    }
}
