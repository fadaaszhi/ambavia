use std::iter::{once, repeat, zip};

use crate::type_checker::Type;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Signature {
    pub param_types: &'static [Type],
    pub return_type: Type,
}

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
    Argmin,
    Argmax,
    Total,
    Mean,
    Count,
    Unique,
    UniquePerm,
    Sort,
    SortPerm,
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
        IndexPolygonList(PgL, N) -> Pg,
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
        Argmin(NL) -> N,
        Argmax(NL) -> N,
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
        UniquePermNumber(NL) -> NL,
        UniquePermPoint(PL) -> NL,
        UniquePermPolygon(PgL) -> NL,
        Sort(NL) -> NL,
        SortKeyNumber(NL, NL) -> NL,
        SortKeyPoint(PL, NL) -> PL,
        SortKeyPolygon(PgL, NL) -> PgL,
        SortPerm(NL) -> NL,
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
            OpName::Argmin => &[Argmin],
            OpName::Argmax => &[Argmax],
            OpName::Total => &[TotalNumber, TotalPoint],
            OpName::Mean => &[MeanNumber, MeanPoint],
            OpName::Count => &[CountNumber, CountPoint, CountPolygon],
            OpName::Unique => &[UniqueNumber, UniquePoint, UniquePolygon],
            OpName::UniquePerm => &[UniquePermNumber, UniquePermPoint, UniquePermPolygon],
            OpName::Sort => &[Sort, SortKeyNumber, SortKeyPoint, SortKeyPolygon],
            OpName::SortPerm => &[SortPerm],
            OpName::Polygon => &[Polygon],
            OpName::Join => &[],
        }
    }
}
impl SatisfyMeta<()> {
    fn try_satisfy_param(supplied: Type, desired: Type) -> Option<Self> {
        // EmptyList "coerces" to any parameter type
        if supplied == Type::EmptyList {
            return Some(Self::Empty);
        }
        match (supplied, desired) {
            (s, d) if s == d => Some(Self::ExactMatch),
            (s, d) if Type::single(s.base()) == d => Some(Self::NeedsBroadcast(())),
            _ => None,
        }
    }
    fn combine(self, other: Self) -> Self {
        match other {
            SatisfyMeta::Empty => SatisfyMeta::Empty,
            SatisfyMeta::NeedsBroadcast(_) => match self {
                SatisfyMeta::ExactMatch | SatisfyMeta::NeedsBroadcast(()) => other,
                SatisfyMeta::Empty => self,
            },
            SatisfyMeta::ExactMatch => self,
        }
    }
    fn needs_broadcast(self) -> bool {
        matches!(self, Self::NeedsBroadcast(()))
    }
}
impl Signature {
    fn satisfies(
        self,
        candidate_types: impl Iterator<Item = Type> + Clone,
    ) -> Option<SigSatisfies<impl Iterator<Item = CoercionMeta> + Clone>> {
        fn try_satisfy_inner(
            candidates: impl Iterator<Item = Type> + Clone,
            desired: impl Iterator<Item = Type> + Clone,
        ) -> Option<(usize, SatisfyMeta<()>)> {
            // assume we have an exact match
            let mut s = SatisfyMeta::ExactMatch;
            let it = zip(candidates, desired.map(Some).chain(repeat(None)));

            let mut len = 0usize;
            for (supplied, desired) in it {
                len += 1;
                let desired = desired?;
                s = s.combine(SatisfyMeta::try_satisfy_param(supplied, desired)?);
            }
            Some((len, s))
        }
        // functions that map (List) -> Scalar get special calling semantics when called with patterns of arguments
        // eg min(1,2,3) should produce Op::Min { args: [ List [N1, N2, N3] ] } instead of Op::Min { args: [ N1, N2, N3 ] }
        // further, this effect can be broadcast e.g min(1, [2,3]) produces
        // Broadcast { scalars: Assign A1 N1, vectors: Assign A2 [N2, N3], body: Op::Min {args: [ List [ A1, A2 ] ] } }
        // instead of Op::Min { args: [ N1, List [ N2, N3 ] ] }
        let is_splat_candidate = self.param_types.len() == 1
            && self.param_types.first().unwrap().is_list()
            && !self.return_type.is_list();
        let mut needs_splat = false;

        let (len, meta) =
            try_satisfy_inner(candidate_types.clone(), self.param_types.iter().copied()).or_else(
                || {
                    is_splat_candidate
                        .then(|| {
                            let r = try_satisfy_inner(
                                candidate_types.clone(),
                                repeat(self.param_types[0].as_single()),
                            );
                            if r.is_some() {
                                needs_splat = true;
                            }
                            r
                        })
                        .flatten()
                },
            )?;
        if len < self.param_types.len() {
            return None;
        }
        let return_ty = match meta {
            SatisfyMeta::ExactMatch | SatisfyMeta::Empty => self.return_type,
            SatisfyMeta::NeedsBroadcast(_) => {
                if self.return_type.is_list() {
                    return None;
                } else {
                    self.return_type.as_list()
                }
            }
        };
        let meta = meta.map_iter(|_| {
            zip(
                candidate_types,
                self.param_types
                    .iter()
                    .copied()
                    .map(Some)
                    .chain(repeat(None)),
            )
            .map(move |(param_ty, desired_ty)| {
                SatisfyMeta::try_satisfy_param(
                    param_ty,
                    needs_splat
                        .then_some(self.param_types[0].as_single())
                        .or(desired_ty)
                        .expect("desired type should be present"),
                )
                .expect("tried to check coercion for parameter but failed after precheck")
                .needs_broadcast()
            })
        });
        Some(SigSatisfies {
            return_ty,
            splat: needs_splat,
            meta,
        })
    }
}
#[derive(Debug, PartialEq)]
pub(crate) struct SigSatisfies<T> {
    pub(crate) return_ty: Type,
    pub(crate) splat: bool,
    pub(crate) meta: SatisfyMeta<T>,
}
#[derive(Debug, PartialEq)]
pub(crate) enum SatisfyMeta<T> {
    Empty,
    NeedsBroadcast(T),
    ExactMatch,
}
impl<T> SatisfyMeta<T> {
    fn map_iter<U>(self, f: impl FnOnce(T) -> U) -> SatisfyMeta<U> {
        match self {
            SatisfyMeta::Empty => SatisfyMeta::Empty,
            SatisfyMeta::NeedsBroadcast(it) => SatisfyMeta::NeedsBroadcast(f(it)),
            SatisfyMeta::ExactMatch => SatisfyMeta::ExactMatch,
        }
    }
}
pub(crate) type CoercionMeta = bool;

impl<T: Iterator<Item = CoercionMeta> + Clone> SigSatisfies<T> {
    fn unify(this: (Op, Self), other: (Option<Op>, Self)) -> Result<(Option<Op>, Self), String> {
        let this_op = Some(this.0);
        let other_op = other.0;
        let (this, other) = (this.1, other.1);
        match (this.meta, other.meta) {
            (SatisfyMeta::Empty, SatisfyMeta::Empty) => {
                // preserve return type iff all matched overloads agree exactly
                Ok(if this.return_ty == other.return_ty {
                    (
                        None,
                        SigSatisfies {
                            return_ty: this.return_ty,
                            meta: SatisfyMeta::Empty,
                            splat: false,
                        },
                    )
                } else {
                    (
                        None,
                        SigSatisfies {
                            return_ty: Type::EmptyList,
                            meta: SatisfyMeta::Empty,
                            splat: false,
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
                    format!(
                        "cannot {self:#?} {}{}",
                        once(first)
                            .chain(ptypes)
                            .map(|a| format!("{a}"))
                            .collect::<Vec<_>>()
                            .join(", "),
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

    use crate::{
        op::{CoercionMeta, Op, OpName, SigSatisfies},
        type_checker::Type,
    };

    type MatchedOverload = (Option<Op>, SigSatisfies<Vec<CoercionMeta>>);
    impl<T> SigSatisfies<T> {
        fn map_iter<U>(self, f: impl FnOnce(T) -> U) -> SigSatisfies<U> {
            let Self {
                return_ty,
                splat,
                meta,
            } = self;
            SigSatisfies {
                return_ty,
                splat,
                meta: meta.map_iter(f),
            }
        }
    }
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
        assert!(
            matches!(
                &v.1,
                &SigSatisfies {
                    return_ty: _,
                    meta: crate::op::SatisfyMeta::Empty,
                    splat: _
                }
            ) && v.1.return_ty == of
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
                    meta: crate::op::SatisfyMeta::ExactMatch,
                    splat
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
        assert!(v.1.splat)
    }
    #[track_caller]
    fn has_coercions(v: &MatchedOverload, broadcast: &[CoercionMeta]) {
        match &v.1.meta {
            super::SatisfyMeta::NeedsBroadcast(c) => assert_eq!(c, broadcast),
            super::SatisfyMeta::Empty | super::SatisfyMeta::ExactMatch => {
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

        empty_list_of(&get(Add, &[Number, EmptyList]), Number);
        empty_list_of(&get(Add, &[EmptyList, Number]), Number);
        empty_list_of(&get(Add, &[EmptyList, EmptyList]), EmptyList);

        let s = get(Min, &[NumberList, Number]);
        splat(&s);
        has_coercions(&s, &[true, false]);
        has_op(&s, Op::Min);

        let s = get(Min, &[Number, NumberList]);
        splat(&s);
        has_coercions(&s, &[false, true]);
        has_op(&s, Op::Min);

        let s = get(Min, &[EmptyList, EmptyList, EmptyList, EmptyList, Number]);
        empty_list_of(&s, Number);
        let s = get(Min, &[EmptyList, EmptyList, EmptyList, EmptyList]);
        dbg!(&s);
        empty_list_of(&s, Number);
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
