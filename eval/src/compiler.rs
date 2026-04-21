use std::{collections::HashMap, iter::zip};

use crate::{
    instruction_builder::{BaseType as IbBaseType, InstructionBuilder, Type as IbType, Value},
    vm::{
        Instruction::{self, *},
        VarIndex,
    },
};
use parse::{
    name_resolver::Id,
    type_checker::{
        Assignment, Body, ComparisonOperator, Expression, Type as TcType, TypedExpression,
    },
};

fn tc_list_to_ib_base(ty: TcType) -> IbBaseType {
    match ty {
        TcType::NumberList => IbBaseType::Number,
        TcType::Point2List => IbBaseType::Point2,
        TcType::Point3List => IbBaseType::Point3,
        TcType::PolygonList => IbBaseType::Polygon,
        TcType::BoolList => IbBaseType::Bool,
        TcType::EmptyList => IbBaseType::Number,
        TcType::Number | TcType::Bool | TcType::Point2 | TcType::Point3 | TcType::Polygon => {
            unreachable!()
        }
    }
}

fn compile_expression(expression: &TypedExpression, builder: &mut InstructionBuilder) -> Value {
    let TypedExpression { ty, e: expression } = expression;
    match expression {
        Expression::Number(x) => builder.load_const(*x),
        Expression::Identifier(name) => builder.load(*name),
        Expression::List(list) => {
            let list = list
                .iter()
                .map(|e| compile_expression(e, builder))
                .collect::<Vec<_>>();
            builder.build_list(tc_list_to_ib_base(*ty), list)
        }
        Expression::ListRange {
            before_ellipsis,
            after_ellipsis,
        } => {
            let [start] = &before_ellipsis[..] else {
                todo!()
            };
            let [end] = &after_ellipsis[..] else { todo!() };
            let start = compile_expression(start, builder);
            let end = compile_expression(end, builder);
            builder.instr2(BuildListFromRange, start, end)
        }
        Expression::Broadcast {
            scalars,
            vectors,
            body,
        } => {
            for Assignment { id, value, .. } in scalars {
                let value = compile_expression(value, builder);
                builder.store(*id, value);
            }

            let vector_values = vectors
                .iter()
                .map(|a| compile_expression(&a.value, builder))
                .collect::<Vec<_>>();

            let mut result = builder.build_list(tc_list_to_ib_base(*ty), vec![]);

            let count = builder.count_specific(&vector_values[0]);

            for v in &vector_values[1..] {
                let c = builder.count_specific(v);
                builder.instr2_in_place(MinInternal, &count, c);
            }

            let i = builder.load_const(0.0);
            let loop_start = builder.label();
            let i_copy = builder.copy(&i);
            let count_copy = builder.copy(&count);
            let i_lt_count = builder.instr2(LessThan, i_copy, count_copy);
            let loop_jump_if_false = builder.jump_if_false(i_lt_count);

            for (Assignment { id, .. }, value) in zip(vectors, &vector_values) {
                let i_copy = builder.copy(&i);
                let value_i = builder.unchecked_index(value, i_copy);
                builder.store(*id, value_i);
            }

            let body = compile_expression(body, builder);
            builder.append(&result, body);

            let one = builder.load_const(1.0);
            builder.instr2_in_place(Add, &i, one);
            let loop_jump = builder.jump(vec![]);
            builder.set_jump_label(loop_jump, &loop_start);
            let loop_end = builder.label();
            builder.set_jump_label(loop_jump_if_false, &loop_end);
            builder.pop(i);
            builder.pop(count);
            builder.swap_pop(&mut result, vector_values);

            result
        }
        Expression::ChainedComparison {
            operands,
            operators,
        } => {
            if operands.len() > 2 {
                todo!();
            }
            let mut a = compile_expression(&operands[0], builder);
            let mut jifs = vec![];
            let mut old_c = None;

            for (b, op) in zip(&operands[1..], operators) {
                if let Some(c) = old_c {
                    builder.pop(c);
                }

                let mut b = compile_expression(b, builder);
                builder.swap(&mut b, &mut a);
                let b_copy = builder.copy(&b);
                let c = builder.instr2(
                    match op {
                        ComparisonOperator::Equal => Equal,
                        ComparisonOperator::Less => LessThan,
                        ComparisonOperator::LessEqual => LessThanEqual,
                        ComparisonOperator::Greater => GreaterThan,
                        ComparisonOperator::GreaterEqual => GreaterThanEqual,
                    },
                    a,
                    b_copy,
                );
                let c_copy = builder.copy(&c);
                jifs.push(builder.jump_if_false(c_copy));
                old_c = Some(c);
                a = b;
            }

            let end = builder.label();

            for jif in jifs {
                builder.set_jump_label(jif, &end);
            }

            let mut c = old_c.unwrap();
            builder.swap(&mut c, &mut a);
            builder.pop(a);
            c
        }
        Expression::Piecewise {
            test,
            consequent,
            alternate,
        } => {
            let test = compile_expression(test, builder);
            let jump_if_false = builder.jump_if_false(test);
            let result = compile_expression(consequent, builder);
            let jump = builder.jump(vec![result]);
            builder.set_jump_label(jump_if_false, &builder.label());
            let result = compile_expression(alternate, builder);
            builder.set_jump_label(jump, &builder.label());
            result
        }
        Expression::SumProd { .. } => todo!(),
        Expression::For {
            body: Body { assignments, value },
            lists,
        } => {
            let list_values = lists
                .iter()
                .rev()
                .map(|a| compile_expression(&a.value, builder))
                .collect::<Vec<_>>();

            let mut result = builder.build_list(tc_list_to_ib_base(*ty), vec![]);
            let mut variables = vec![];

            for (Assignment { id, .. }, value) in zip(lists.iter().rev(), &list_values) {
                let count = builder.count_specific(value);
                let i = builder.load_const(0.0);
                let loop_start = builder.label();
                let i_copy = builder.copy(&i);
                let count_copy = builder.copy(&count);
                let i_lt_count = builder.instr2(LessThan, i_copy, count_copy);
                let loop_jump_if_false = builder.jump_if_false(i_lt_count);
                let i_copy = builder.copy(&i);
                let value_i = builder.unchecked_index(value, i_copy);
                builder.store(*id, value_i);
                variables.push((count, i, loop_start, loop_jump_if_false));
            }

            for Assignment { id, value, .. } in assignments {
                let value = compile_expression(value, builder);
                builder.store(*id, value);
            }

            let value = compile_expression(value, builder);
            builder.append(&result, value);

            for (count, i, loop_start, loop_jump_if_false) in variables.into_iter().rev() {
                let one = builder.load_const(1.0);
                builder.instr2_in_place(Add, &i, one);
                let loop_jump = builder.jump(vec![]);
                builder.set_jump_label(loop_jump, &loop_start);
                let loop_end = builder.label();
                builder.set_jump_label(loop_jump_if_false, &loop_end);
                builder.pop(i);
                builder.pop(count);
            }

            builder.swap_pop(&mut result, list_values);
            result
        }
        Expression::Op { operation, args } => {
            use parse::op::Op;
            match operation {
                Op::JoinNumber | Op::JoinPoint2 | Op::JoinPoint3 | Op::JoinPolygon => {
                    let (base, push, concat) = match operation {
                        Op::JoinNumber => (IbBaseType::Number, Push, Concat),
                        Op::JoinPoint2 => (IbBaseType::Point2, Push2, Concat2),
                        Op::JoinPoint3 => (IbBaseType::Point3, Push3, Concat3),
                        Op::JoinPolygon => (IbBaseType::Polygon, PushPolygon, ConcatPolygon),
                        _ => unreachable!(),
                    };
                    let first = compile_expression(&args[0], builder);
                    let mut list = if args[0].ty.is_list() {
                        first
                    } else {
                        builder.build_list(base, vec![first])
                    };
                    for a in &args[1..] {
                        let instr = if a.ty.is_list() { concat } else { push };
                        let a = compile_expression(a, builder);
                        list = builder.instr2(instr, list, a);
                    }
                    list
                }
                _ => {
                    let args = args
                        .iter()
                        .map(|e| compile_expression(e, builder))
                        .collect::<Vec<_>>();
                    let mut args = args.into_iter();
                    let mut arg = || args.next().unwrap();
                    match operation {
                        Op::Ln => builder.instr1(Ln, arg()),
                        Op::Exp => builder.instr1(Exp, arg()),
                        Op::Erf => builder.instr1(Erf, arg()),
                        Op::Sin => builder.instr1(Sin, arg()),
                        Op::Cos => builder.instr1(Cos, arg()),
                        Op::Tan => builder.instr1(Tan, arg()),
                        Op::Sec => builder.instr1(Sec, arg()),
                        Op::Csc => builder.instr1(Csc, arg()),
                        Op::Cot => builder.instr1(Cot, arg()),
                        Op::Sinh => builder.instr1(Sinh, arg()),
                        Op::Cosh => builder.instr1(Cosh, arg()),
                        Op::Tanh => builder.instr1(Tanh, arg()),
                        Op::Sech => builder.instr1(Sech, arg()),
                        Op::Csch => builder.instr1(Csch, arg()),
                        Op::Coth => builder.instr1(Coth, arg()),
                        Op::Asin => builder.instr1(Asin, arg()),
                        Op::Acos => builder.instr1(Acos, arg()),
                        Op::Atan => builder.instr1(Atan, arg()),
                        Op::Atan2 => builder.instr2(Atan2, arg(), arg()),
                        Op::Asec => builder.instr1(Asec, arg()),
                        Op::Acsc => builder.instr1(Acsc, arg()),
                        Op::Acot => builder.instr1(Acot, arg()),
                        Op::Asinh => builder.instr1(Asinh, arg()),
                        Op::Acosh => builder.instr1(Acosh, arg()),
                        Op::Atanh => builder.instr1(Atanh, arg()),
                        Op::Asech => builder.instr1(Asech, arg()),
                        Op::Acsch => builder.instr1(Acsch, arg()),
                        Op::Acoth => builder.instr1(Acoth, arg()),
                        Op::Abs => builder.instr1(Abs, arg()),
                        Op::Sgn => builder.instr1(Sgn, arg()),
                        Op::Round => builder.instr1(Round, arg()),
                        Op::RoundWithPrecision => builder.instr2(RoundWithPrecision, arg(), arg()),
                        Op::Floor => builder.instr1(Floor, arg()),
                        Op::Ceil => builder.instr1(Ceil, arg()),
                        Op::Mod => builder.instr2(Mod, arg(), arg()),
                        Op::Midpoint2 => builder.instr2(Midpoint2, arg(), arg()),
                        Op::Midpoint3 => builder.instr2(Midpoint3, arg(), arg()),
                        Op::Distance2 => builder.instr2(Distance2, arg(), arg()),
                        Op::Distance3 => builder.instr2(Distance3, arg(), arg()),
                        Op::Min => builder.instr1(Min, arg()),
                        Op::Max => builder.instr1(Max, arg()),
                        Op::Median => builder.instr1(Median, arg()),
                        Op::Argmin => builder.instr1(Argmin, arg()),
                        Op::Argmax => builder.instr1(Argmax, arg()),
                        Op::TotalNumber => builder.instr1(Total, arg()),
                        Op::TotalPoint2 => builder.instr1(Total2, arg()),
                        Op::TotalPoint3 => builder.instr1(Total3, arg()),
                        Op::MeanNumber => builder.instr1(Mean, arg()),
                        Op::MeanPoint2 => builder.instr1(Mean2, arg()),
                        Op::MeanPoint3 => builder.instr1(Mean3, arg()),
                        Op::CountNumber => builder.instr1(Count, arg()),
                        Op::CountPoint2 => builder.instr1(Count2, arg()),
                        Op::CountPoint3 => builder.instr1(Count3, arg()),
                        Op::CountPolygon => builder.instr1(CountPolygonList, arg()),
                        Op::RepeatNumber => builder.instr2(Repeat, arg(), arg()),
                        Op::RepeatPoint2 => builder.instr2(Repeat2, arg(), arg()),
                        Op::RepeatPoint3 => builder.instr2(Repeat3, arg(), arg()),
                        Op::RepeatPolygon => builder.instr2(RepeatPolygon, arg(), arg()),
                        Op::RepeatNumberList => builder.instr2(RepeatList, arg(), arg()),
                        Op::RepeatPoint2List => builder.instr2(Repeat2List, arg(), arg()),
                        Op::RepeatPoint3List => builder.instr2(Repeat3List, arg(), arg()),
                        Op::RepeatPolygonList => builder.instr2(RepeatPolygonList, arg(), arg()),
                        Op::UniqueNumber => builder.instr1(Unique, arg()),
                        Op::UniquePoint2 => builder.instr1(Unique2, arg()),
                        Op::UniquePoint3 => builder.instr1(Unique3, arg()),
                        Op::UniquePolygon => builder.instr1(UniquePolygon, arg()),
                        Op::UniquePermNumber => builder.instr1(UniquePerm, arg()),
                        Op::UniquePermPoint2 => builder.instr1(UniquePerm2, arg()),
                        Op::UniquePermPoint3 => builder.instr1(UniquePerm3, arg()),
                        Op::UniquePermPolygon => builder.instr1(UniquePermPolygon, arg()),
                        Op::Sort => builder.instr1(Sort, arg()),
                        Op::SortKeyNumber => builder.instr2(SortKey, arg(), arg()),
                        Op::SortKeyPoint2 => builder.instr2(SortKey2, arg(), arg()),
                        Op::SortKeyPoint3 => builder.instr2(SortKey3, arg(), arg()),
                        Op::SortKeyPolygon => builder.instr2(SortKeyPolygon, arg(), arg()),
                        Op::SortPerm => builder.instr1(SortPerm, arg()),
                        Op::Polygon => builder.instr1(Polygon, arg()),
                        Op::AddNumber => builder.instr2(Add, arg(), arg()),
                        Op::AddPoint2 => builder.instr2(Add2, arg(), arg()),
                        Op::AddPoint3 => builder.instr2(Add3, arg(), arg()),
                        Op::SubNumber => builder.instr2(Sub, arg(), arg()),
                        Op::SubPoint2 => builder.instr2(Sub2, arg(), arg()),
                        Op::SubPoint3 => builder.instr2(Sub3, arg(), arg()),
                        Op::MulNumber => builder.instr2(Mul, arg(), arg()),
                        Op::MulNumberPoint2 => builder.instr2(Mul1_2, arg(), arg()),
                        Op::MulNumberPoint3 => builder.instr2(Mul1_3, arg(), arg()),
                        Op::DivNumber => builder.instr2(Div, arg(), arg()),
                        Op::DivPoint2Number => builder.instr2(Div2_1, arg(), arg()),
                        Op::DivPoint3Number => builder.instr2(Div3_1, arg(), arg()),
                        Op::Pow => builder.instr2(Pow, arg(), arg()),
                        Op::Dot2 => builder.instr2(Dot2, arg(), arg()),
                        Op::Dot3 => builder.instr2(Dot3, arg(), arg()),
                        Op::Cross => builder.instr2(Cross, arg(), arg()),
                        Op::Point2 => builder.instr2(Point2, arg(), arg()),
                        Op::Point3 => builder.instr3(Point3, arg(), arg(), arg()),
                        Op::IndexNumberList => builder.instr2(Index, arg(), arg()),
                        Op::IndexPoint2List => builder.instr2(Index2, arg(), arg()),
                        Op::IndexPoint3List => builder.instr2(Index3, arg(), arg()),
                        Op::IndexPolygonList => builder.instr2(IndexPolygonList, arg(), arg()),
                        Op::NegNumber => builder.instr1(Neg, arg()),
                        Op::NegPoint2 => builder.instr1(Neg2, arg()),
                        Op::NegPoint3 => builder.instr1(Neg3, arg()),
                        Op::Fac => todo!("factorial"), // builder.instr1(todo, arg()),
                        Op::Sqrt => builder.instr1(Sqrt, arg()),
                        Op::Mag2 => builder.instr1(Hypot2, arg()),
                        Op::Mag3 => builder.instr1(Hypot3, arg()),
                        Op::Point2X => builder.instr1(Point2X, arg()),
                        Op::Point2Y => builder.instr1(Point2Y, arg()),
                        Op::Point3X => builder.instr1(Point3X, arg()),
                        Op::Point3Y => builder.instr1(Point3Y, arg()),
                        Op::Point3Z => builder.instr1(Point3Z, arg()),
                        Op::FilterNumberList
                        | Op::FilterPoint2List
                        | Op::FilterPoint3List
                        | Op::FilterPolygonList => {
                            let mut result = builder.build_list(
                                match ty {
                                    TcType::NumberList => IbBaseType::Number,
                                    TcType::Point2List => IbBaseType::Point2,
                                    TcType::Point3List => IbBaseType::Point3,
                                    TcType::PolygonList => IbBaseType::Polygon,
                                    TcType::BoolList => IbBaseType::Bool,
                                    TcType::EmptyList => IbBaseType::Number,
                                    TcType::Number
                                    | TcType::Point2
                                    | TcType::Point3
                                    | TcType::Polygon
                                    | TcType::Bool => {
                                        unreachable!()
                                    }
                                },
                                vec![],
                            );
                            let (left, right) = (arg(), arg());
                            let left_count = builder.count_specific(&left);
                            let right_count = builder.count_specific(&right);
                            let count = builder.instr2(MinInternal, left_count, right_count);

                            let i = builder.load_const(0.0);
                            let loop_start = builder.label();
                            let i_copy = builder.copy(&i);
                            let count_copy = builder.copy(&count);
                            let i_lt_count = builder.instr2(LessThan, i_copy, count_copy);
                            let loop_jump_if_false = builder.jump_if_false(i_lt_count);

                            let i_copy = builder.copy(&i);
                            let right_i = builder.unchecked_index(&right, i_copy);

                            let filter_jump_if_false = builder.jump_if_false(right_i);
                            let i_copy = builder.copy(&i);
                            let left_i = builder.unchecked_index(&left, i_copy);
                            builder.append(&result, left_i);
                            let label = builder.label();
                            builder.set_jump_label(filter_jump_if_false, &label);

                            let one = builder.load_const(1.0);
                            builder.instr2_in_place(Add, &i, one);
                            let loop_jump = builder.jump(vec![]);
                            builder.set_jump_label(loop_jump, &loop_start);
                            let loop_end = builder.label();
                            builder.set_jump_label(loop_jump_if_false, &loop_end);
                            builder.pop(i);
                            builder.pop(count);
                            builder.swap_pop(&mut result, vec![left, right]);

                            result
                        }
                        Op::JoinNumber | Op::JoinPoint2 | Op::JoinPoint3 | Op::JoinPolygon => {
                            unreachable!()
                        }
                    }
                }
            }
        }
    }
}

pub fn compile_assignments<
    'a,
    I: IntoIterator<Item = Id>,
    A: IntoIterator<Item = &'a Assignment>,
>(
    constants: impl IntoIterator<Item = &'a Assignment>,
    functions: impl IntoIterator<Item = (I, A)>,
    builtin_constants: impl IntoIterator<Item = (Id, TcType)>,
) -> (
    Vec<Instruction>,
    Vec<Vec<Instruction>>,
    HashMap<Id, VarIndex>,
) {
    let mut builder = InstructionBuilder::default();

    for (id, ty) in builtin_constants {
        builder.define(
            id,
            match ty {
                TcType::Number => IbType::Number,
                TcType::NumberList => IbType::NumberList,
                TcType::Point2 => IbType::Point2,
                TcType::Point2List => IbType::Point2List,
                TcType::Point3 => IbType::Point3,
                TcType::Point3List => IbType::Point3List,
                TcType::Polygon => IbType::Polygon,
                TcType::PolygonList => IbType::PolygonList,
                TcType::Bool => IbType::Bool,
                TcType::BoolList => IbType::BoolList,
                TcType::EmptyList => panic!(),
            },
        );
    }

    for Assignment { id, value, .. } in constants {
        let value = compile_expression(value, &mut builder);
        builder.store(*id, value);
    }

    let constants_program = builder.clear_instructions();

    let mut function_programs = vec![];

    for (parameters, assignments) in functions {
        for id in parameters {
            builder.define(id, IbType::Number);
        }
        for Assignment { id, value, .. } in assignments {
            let value = compile_expression(value, &mut builder);
            builder.store(*id, value);
        }
        function_programs.push(builder.clear_instructions());
    }

    let vars = builder.defined_vars();
    (constants_program, function_programs, vars)
}

#[cfg(test)]
mod tests {
    use super::*;
    use Expression::{Identifier as TId, Number as TNum, Op as TOp};
    use InstructionBuilder as Ib;
    use parse::{name_resolver::Id, op::Op};
    use pretty_assertions::assert_eq;

    fn num(e: Expression) -> TypedExpression {
        TypedExpression {
            ty: TcType::Number,
            e,
        }
    }

    fn num_list(e: Expression) -> TypedExpression {
        TypedExpression {
            ty: TcType::NumberList,
            e,
        }
    }

    fn pt(e: Expression) -> TypedExpression {
        TypedExpression {
            ty: TcType::Point2,
            e,
        }
    }

    fn pt_list(e: Expression) -> TypedExpression {
        TypedExpression {
            ty: TcType::Point2List,
            e,
        }
    }

    #[test]
    fn number() {
        let mut b = Ib::default();
        compile_expression(&num(TNum(5.0)), &mut b);
        compile_expression(&num(TNum(3.0)), &mut b);
        assert_eq!(b.finish(), [LoadConst(5.0), LoadConst(3.0)]);
    }

    fn create_empty_variable(builder: &mut InstructionBuilder, name: Id, ty: TcType) {
        let v = match ty {
            TcType::Number => builder.load_const(0.0),
            TcType::NumberList => builder.build_list(IbBaseType::Number, vec![]),
            TcType::Point2 => {
                let x = builder.load_const(0.0);
                let y = builder.load_const(0.0);
                builder.instr2(Point2, x, y)
            }
            TcType::Point2List => builder.build_list(IbBaseType::Point2, vec![]),
            TcType::Point3 => {
                let x = builder.load_const(0.0);
                let y = builder.load_const(0.0);
                let z = builder.load_const(0.0);
                builder.instr3(Point3, x, y, z)
            }
            TcType::Point3List => builder.build_list(IbBaseType::Point3, vec![]),
            TcType::Polygon => {
                let p = builder.build_list(IbBaseType::Point2, vec![]);
                builder.instr1(Polygon, p)
            }
            TcType::PolygonList => builder.build_list(IbBaseType::Polygon, vec![]),
            TcType::EmptyList => panic!("why"),
            TcType::Bool | TcType::BoolList => panic!("bruh"),
        };
        builder.store(name, v);
    }

    #[test]
    fn identifiers() {
        let mut b = Ib::default();
        create_empty_variable(&mut b, Id(0), TcType::Number);
        compile_expression(&num(TId(Id(0))), &mut b);
        create_empty_variable(&mut b, Id(1), TcType::Point2);
        compile_expression(&pt(TId(Id(1))), &mut b);
        create_empty_variable(&mut b, Id(2), TcType::Point2List);
        compile_expression(&pt_list(TId(Id(2))), &mut b);
        create_empty_variable(&mut b, Id(3), TcType::NumberList);
        compile_expression(&num_list(TId(Id(3))), &mut b);
        compile_expression(&pt(TId(Id(1))), &mut b);
        assert_eq!(
            b.finish(),
            [
                LoadConst(0.0),
                Store(VarIndex(0)),
                Load(VarIndex(0)),
                LoadConst(0.0),
                LoadConst(0.0),
                Store2(VarIndex(3)),
                Load2(VarIndex(3)),
                BuildList(0),
                Store(VarIndex(6)),
                Load(VarIndex(6)),
                BuildList(0),
                Store(VarIndex(9)),
                Load(VarIndex(9)),
                Load2(VarIndex(3))
            ]
        );
    }

    #[test]
    fn bop() {
        let make_vars_and_builder = || {
            let mut b = Ib::default();
            create_empty_variable(&mut b, Id(0), TcType::Number);
            create_empty_variable(&mut b, Id(1), TcType::Point2);
            b
        };
        let number = || num(TId(Id(0)));
        let point = || pt(TId(Id(1)));

        let mut b = make_vars_and_builder();

        compile_expression(
            &num(TOp {
                operation: Op::AddNumber,
                args: vec![number(), number()],
            }),
            &mut b,
        );
        assert_eq!(b.finish()[5..], [Load(VarIndex(0)), Load(VarIndex(0)), Add]);

        let mut b = make_vars_and_builder();
        compile_expression(
            &pt(TOp {
                operation: Op::AddPoint2,
                args: vec![point(), point()],
            }),
            &mut b,
        );
        assert_eq!(
            b.finish()[5..],
            [Load2(VarIndex(3)), Load2(VarIndex(3)), Add2]
        );

        let mut b = make_vars_and_builder();
        compile_expression(
            &pt(TOp {
                operation: Op::MulNumberPoint2,
                args: vec![number(), point()],
            }),
            &mut b,
        );
        assert_eq!(
            b.finish()[5..],
            [Load(VarIndex(0)), Load2(VarIndex(3)), Mul1_2]
        );

        let mut b = make_vars_and_builder();
        compile_expression(
            &pt(TOp {
                operation: Op::DivPoint2Number,
                args: vec![point(), number()],
            }),
            &mut b,
        );
        assert_eq!(
            b.finish()[5..],
            [Load2(VarIndex(3)), Load(VarIndex(0)), Div2_1]
        );

        let mut b = make_vars_and_builder();
        compile_expression(
            &num(TOp {
                operation: Op::Pow,
                args: vec![number(), number()],
            }),
            &mut b,
        );
        assert_eq!(b.finish()[5..], [Load(VarIndex(0)), Load(VarIndex(0)), Pow]);

        let mut b = make_vars_and_builder();
        compile_expression(
            &num(TOp {
                operation: Op::Dot2,
                args: vec![point(), point()],
            }),
            &mut b,
        );
        assert_eq!(
            b.finish()[5..],
            [Load2(VarIndex(3)), Load2(VarIndex(3)), Dot2]
        );
    }
}
