use std::iter::zip;

use typed_index_collections::{TiSlice, TiVec};

use crate::{
    instruction_builder::{BaseType as IbBaseType, InstructionBuilder, Value},
    vm::Instruction::{self, *},
};
use parse::type_checker::{
    Assignment, AssignmentIndex, BinaryOperator, Body, BuiltIn, ComparisonOperator, Expression,
    Type as TcType, TypedExpression, UnaryOperator,
};

fn tc_list_to_ib_base(ty: TcType) -> IbBaseType {
    match ty {
        TcType::NumberList => IbBaseType::Number,
        TcType::PointList => IbBaseType::Point,
        TcType::PolygonList => IbBaseType::Polygon,
        TcType::BoolList => IbBaseType::Bool,
        TcType::EmptyList => IbBaseType::Number,
        TcType::Number | TcType::Bool | TcType::Point | TcType::Polygon => unreachable!(),
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
            for Assignment { id, value } in scalars {
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
        Expression::UnaryOperation { operation, arg } => {
            let arg = compile_expression(arg, builder);
            builder.instr1(
                match operation {
                    UnaryOperator::NegNumber => Neg,
                    UnaryOperator::NegPoint => Neg2,
                    UnaryOperator::Fac => todo!("factorial"),
                    UnaryOperator::Sqrt => Sqrt,
                    UnaryOperator::Abs => Abs,
                    UnaryOperator::Mag => Hypot,
                    UnaryOperator::PointX => PointX,
                    UnaryOperator::PointY => PointY,
                },
                arg,
            )
        }
        Expression::BinaryOperation {
            operation,
            left,
            right,
        } => {
            let left = compile_expression(left, builder);
            let right = compile_expression(right, builder);

            if matches!(
                operation,
                BinaryOperator::FilterNumberList
                    | BinaryOperator::FilterPointList
                    | BinaryOperator::FilterPolygonList
            ) {
                let mut result = builder.build_list(
                    match ty {
                        TcType::NumberList => IbBaseType::Number,
                        TcType::PointList => IbBaseType::Point,
                        TcType::PolygonList => IbBaseType::Polygon,
                        TcType::BoolList => IbBaseType::Bool,
                        TcType::EmptyList => IbBaseType::Number,
                        TcType::Number | TcType::Point | TcType::Polygon | TcType::Bool => {
                            unreachable!()
                        }
                    },
                    vec![],
                );

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

                return result;
            }

            builder.instr2(
                match operation {
                    BinaryOperator::AddNumber => Add,
                    BinaryOperator::AddPoint => Add2,
                    BinaryOperator::SubNumber => Sub,
                    BinaryOperator::SubPoint => Sub2,
                    BinaryOperator::MulNumber => Mul,
                    BinaryOperator::MulNumberPoint => Mul1_2,
                    BinaryOperator::DivNumber => Div,
                    BinaryOperator::DivPointNumber => Div2_1,
                    BinaryOperator::Pow => Pow,
                    BinaryOperator::Dot => Dot,
                    BinaryOperator::Point => Point,
                    BinaryOperator::IndexNumberList => Index,
                    BinaryOperator::IndexPointList => Index2,
                    BinaryOperator::IndexPolygonList => IndexPolygonList,
                    BinaryOperator::FilterNumberList
                    | BinaryOperator::FilterPointList
                    | BinaryOperator::FilterPolygonList => {
                        unreachable!()
                    }
                },
                left,
                right,
            )
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
        Expression::BuiltIn { name, args } => {
            if matches!(
                name,
                BuiltIn::JoinNumber | BuiltIn::JoinPoint | BuiltIn::JoinPolygon
            ) {
                let (base, push, concat) = match name {
                    BuiltIn::JoinNumber => (IbBaseType::Number, Push, Concat),
                    BuiltIn::JoinPoint => (IbBaseType::Point, Push2, Concat2),
                    BuiltIn::JoinPolygon => (IbBaseType::Polygon, PushPolygon, ConcatPolygon),
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
            } else {
                let args = args
                    .iter()
                    .map(|e| compile_expression(e, builder))
                    .collect::<Vec<_>>();
                let mut args = args.into_iter();
                let mut arg = || args.next().unwrap();
                match name {
                    BuiltIn::Ln => builder.instr1(Ln, arg()),
                    BuiltIn::Exp => builder.instr1(Exp, arg()),
                    BuiltIn::Erf => builder.instr1(Erf, arg()),
                    BuiltIn::Sin => builder.instr1(Sin, arg()),
                    BuiltIn::Cos => builder.instr1(Cos, arg()),
                    BuiltIn::Tan => builder.instr1(Tan, arg()),
                    BuiltIn::Sec => builder.instr1(Sec, arg()),
                    BuiltIn::Csc => builder.instr1(Csc, arg()),
                    BuiltIn::Cot => builder.instr1(Cot, arg()),
                    BuiltIn::Sinh => builder.instr1(Sinh, arg()),
                    BuiltIn::Cosh => builder.instr1(Cosh, arg()),
                    BuiltIn::Tanh => builder.instr1(Tanh, arg()),
                    BuiltIn::Sech => builder.instr1(Sech, arg()),
                    BuiltIn::Csch => builder.instr1(Csch, arg()),
                    BuiltIn::Coth => builder.instr1(Coth, arg()),
                    BuiltIn::Asin => builder.instr1(Asin, arg()),
                    BuiltIn::Acos => builder.instr1(Acos, arg()),
                    BuiltIn::Atan => builder.instr1(Atan, arg()),
                    BuiltIn::Atan2 => builder.instr2(Atan2, arg(), arg()),
                    BuiltIn::Asec => builder.instr1(Asec, arg()),
                    BuiltIn::Acsc => builder.instr1(Acsc, arg()),
                    BuiltIn::Acot => builder.instr1(Acot, arg()),
                    BuiltIn::Asinh => builder.instr1(Asinh, arg()),
                    BuiltIn::Acosh => builder.instr1(Acosh, arg()),
                    BuiltIn::Atanh => builder.instr1(Atanh, arg()),
                    BuiltIn::Asech => builder.instr1(Asech, arg()),
                    BuiltIn::Acsch => builder.instr1(Acsch, arg()),
                    BuiltIn::Acoth => builder.instr1(Acoth, arg()),
                    BuiltIn::Abs => builder.instr1(Abs, arg()),
                    BuiltIn::Sgn => builder.instr1(Sgn, arg()),
                    BuiltIn::Round => builder.instr1(Round, arg()),
                    BuiltIn::RoundWithPrecision => builder.instr2(RoundWithPrecision, arg(), arg()),
                    BuiltIn::Floor => builder.instr1(Floor, arg()),
                    BuiltIn::Ceil => builder.instr1(Ceil, arg()),
                    BuiltIn::Mod => builder.instr2(Mod, arg(), arg()),
                    BuiltIn::Midpoint => builder.instr2(Midpoint, arg(), arg()),
                    BuiltIn::Distance => builder.instr2(Distance, arg(), arg()),
                    BuiltIn::Min => builder.instr1(Min, arg()),
                    BuiltIn::Max => builder.instr1(Max, arg()),
                    BuiltIn::Median => builder.instr1(Median, arg()),
                    BuiltIn::TotalNumber => builder.instr1(Total, arg()),
                    BuiltIn::TotalPoint => builder.instr1(Total2, arg()),
                    BuiltIn::MeanNumber => builder.instr1(Mean, arg()),
                    BuiltIn::MeanPoint => builder.instr1(Mean2, arg()),
                    BuiltIn::CountNumber => builder.instr1(Count, arg()),
                    BuiltIn::CountPoint => builder.instr1(Count2, arg()),
                    BuiltIn::CountPolygon => builder.instr1(CountPolygonList, arg()),
                    BuiltIn::UniqueNumber => builder.instr1(Unique, arg()),
                    BuiltIn::UniquePoint => builder.instr1(Unique2, arg()),
                    BuiltIn::UniquePolygon => builder.instr1(UniquePolygon, arg()),
                    BuiltIn::Sort => builder.instr1(Sort, arg()),
                    BuiltIn::SortKeyNumber => builder.instr2(SortKey, arg(), arg()),
                    BuiltIn::SortKeyPoint => builder.instr2(SortKey2, arg(), arg()),
                    BuiltIn::SortKeyPolygon => builder.instr2(SortKeyPolygon, arg(), arg()),
                    BuiltIn::Polygon => builder.instr1(Polygon, arg()),
                    BuiltIn::JoinNumber | BuiltIn::JoinPoint | BuiltIn::JoinPolygon => {
                        unreachable!()
                    }
                }
            }
        }
    }
}

pub fn compile_assignments(
    assignments: &TiSlice<AssignmentIndex, Assignment>,
) -> (Vec<Instruction>, TiVec<AssignmentIndex, usize>) {
    let mut builder = InstructionBuilder::default();

    for Assignment { id, value, .. } in assignments {
        let value = compile_expression(value, &mut builder);
        builder.store(*id, value);
    }

    let vars = builder.defined_vars();
    let program = builder.finish();
    (
        program,
        assignments
            .iter()
            .map(|a| *vars.get(&a.id).unwrap())
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use BinaryOperator as TBo;
    use Expression::{BinaryOperation as TBop, Identifier as TId, Number as TNum};
    use InstructionBuilder as Ib;
    use pretty_assertions::assert_eq;

    fn bx<T>(x: T) -> Box<T> {
        Box::new(x)
    }

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
            ty: TcType::Point,
            e,
        }
    }

    fn pt_list(e: Expression) -> TypedExpression {
        TypedExpression {
            ty: TcType::PointList,
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

    fn create_empty_variable(builder: &mut InstructionBuilder, name: usize, ty: TcType) {
        let v = match ty {
            TcType::Number => builder.load_const(0.0),
            TcType::NumberList => builder.build_list(IbBaseType::Number, vec![]),
            TcType::Point => {
                let x = builder.load_const(0.0);
                let y = builder.load_const(0.0);
                builder.instr2(Point, x, y)
            }
            TcType::PointList => builder.build_list(IbBaseType::Point, vec![]),
            TcType::Polygon => {
                let p = builder.build_list(IbBaseType::Point, vec![]);
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
        create_empty_variable(&mut b, 0, TcType::Number);
        compile_expression(&num(TId(0)), &mut b);
        create_empty_variable(&mut b, 1, TcType::Point);
        compile_expression(&pt(TId(1)), &mut b);
        create_empty_variable(&mut b, 2, TcType::PointList);
        compile_expression(&pt_list(TId(2)), &mut b);
        create_empty_variable(&mut b, 3, TcType::NumberList);
        compile_expression(&num_list(TId(3)), &mut b);
        compile_expression(&pt(TId(1)), &mut b);
        assert_eq!(
            b.finish(),
            [
                LoadConst(0.0),
                Store(0),
                Load(0),
                LoadConst(0.0),
                LoadConst(0.0),
                Store2(2),
                Load2(2),
                BuildList(0),
                Store(4),
                Load(4),
                BuildList(0),
                Store(6),
                Load(6),
                Load2(2)
            ]
        );
    }

    #[test]
    fn bop() {
        let make_vars_and_builder = || {
            let mut b = Ib::default();
            create_empty_variable(&mut b, 0, TcType::Number);
            create_empty_variable(&mut b, 1, TcType::Point);
            b
        };
        let number = || bx(num(TId(0)));
        let point = || bx(pt(TId(1)));

        let mut b = make_vars_and_builder();

        compile_expression(
            &num(TBop {
                operation: TBo::AddNumber,
                left: number(),
                right: number(),
            }),
            &mut b,
        );
        assert_eq!(b.finish()[5..], [Load(0), Load(0), Add]);

        let mut b = make_vars_and_builder();
        compile_expression(
            &pt(TBop {
                operation: TBo::AddPoint,
                left: point(),
                right: point(),
            }),
            &mut b,
        );
        assert_eq!(b.finish()[5..], [Load2(2), Load2(2), Add2]);

        let mut b = make_vars_and_builder();
        compile_expression(
            &pt(TBop {
                operation: TBo::MulNumberPoint,
                left: number(),
                right: point(),
            }),
            &mut b,
        );
        assert_eq!(b.finish()[5..], [Load(0), Load2(2), Mul1_2]);

        let mut b = make_vars_and_builder();
        compile_expression(
            &pt(TBop {
                operation: TBo::DivPointNumber,
                left: point(),
                right: number(),
            }),
            &mut b,
        );
        assert_eq!(b.finish()[5..], [Load2(2), Load(0), Div2_1]);

        let mut b = make_vars_and_builder();
        compile_expression(
            &num(TBop {
                operation: TBo::Pow,
                left: number(),
                right: number(),
            }),
            &mut b,
        );
        assert_eq!(b.finish()[5..], [Load(0), Load(0), Pow]);

        let mut b = make_vars_and_builder();
        compile_expression(
            &num(TBop {
                operation: TBo::Dot,
                left: point(),
                right: point(),
            }),
            &mut b,
        );
        assert_eq!(b.finish()[5..], [Load2(2), Load2(2), Dot]);
    }
}
