use std::collections::HashMap;

use crate::{
    instruction_builder::{BaseType, InstructionBuilder, Type, Value},
    resolver::{Assignment, BinaryOperator, ComparisonOperator, Expression, UnaryOperator},
    vm::Instruction::{self, *},
};

#[derive(Debug, Default)]
pub struct Names {
    map: HashMap<usize, Type>,
}

impl Names {
    fn create(&mut self, name: usize, typ: Type) -> Result<(), String> {
        if self.map.contains_key(&name) {
            return Err(format!("variable '{name}' is already defined"));
        }

        self.map.insert(name, typ);
        Ok(())
    }

    fn get(&self, name: usize) -> Result<Type, String> {
        self.map
            .get(&name)
            .copied()
            .ok_or_else(|| format!("variable '{name}' is not defined"))
    }
}

pub fn compile_expression(
    expression: &Expression,
    builder: &mut InstructionBuilder,
    names: &Names,
) -> Result<Value, String> {
    match expression {
        Expression::Number(x) => Ok(builder.load_const(*x)),
        Expression::Identifier(name) => {
            let ty = names.get(*name)?;
            let value = builder.load(*name);
            assert_eq!(value.ty(), ty);
            Ok(value)
        }
        Expression::List(list) => {
            if list.is_empty() {
                Ok(builder.empty_list())
            } else {
                let list = list
                    .iter()
                    .map(|e| compile_expression(e, builder, names))
                    .collect::<Result<Vec<_>, _>>()?;
                let ty = list[0].ty();

                if ty.is_list() {
                    return Err(format!("cannot store {ty} in a list"));
                }

                for v in &list {
                    if v.ty() != ty {
                        return Err("all elements of a list must have the same type".into());
                    }
                }

                Ok(builder.build_list(ty.base(), list))
            }
        }
        Expression::ListRange {
            before_ellipsis,
            after_ellipsis,
        } => {
            let [start] = &before_ellipsis[..] else {
                todo!()
            };
            let [end] = &after_ellipsis[..] else { todo!() };
            let start = compile_expression(start, builder, names)?;
            let end = compile_expression(end, builder, names)?;
            if start.ty() != Type::Number || end.ty() != Type::Number {
                return Err("ranges must be arithmetic sequences".into());
            }
            Ok(builder.instr2(BuildListFromRange, start, end))
        }
        Expression::UnaryOperation { operation, arg } => {
            let arg = compile_expression(arg, builder, names)?;
            let broadcast = arg.ty().is_list();

            if broadcast && arg.ty() == Type::EmptyList {
                return Ok(arg);
            }

            let (base, instruction) = match operation {
                UnaryOperator::Neg => match arg.ty().base() {
                    BaseType::Number => (BaseType::Number, Neg),
                    BaseType::Point => (BaseType::Point, Neg2),
                    _ => return Err(format!("cannot negate {}", arg.ty())),
                },
                UnaryOperator::Fac => todo!(),
                UnaryOperator::Sqrt => match arg.ty().base() {
                    BaseType::Number => (BaseType::Number, Sqrt),
                    _ => return Err(format!("cannot take the square root of {}", arg.ty())),
                },
                UnaryOperator::Norm => match arg.ty().base() {
                    BaseType::Number => (BaseType::Number, Abs),
                    BaseType::Point => (BaseType::Number, Hypot),
                    _ => return Err(format!("cannot take the absolute value of {}", arg.ty())),
                },
                UnaryOperator::PointX => match arg.ty().base() {
                    BaseType::Point => (BaseType::Number, PointX),
                    _ => return Err(format!("cannot access coordinate '.x' of {}", arg.ty())),
                },
                UnaryOperator::PointY => match arg.ty().base() {
                    BaseType::Point => (BaseType::Number, PointY),
                    _ => return Err(format!("cannot access coordinate '.y' of {}", arg.ty())),
                },
            };

            if broadcast {
                let mut list = builder.build_list(base, vec![]);
                let count = builder.count_specific(&arg);
                let i = builder.load_const(0.0);
                let loop_start = builder.label();
                let i_copy = builder.copy(&i);
                let count_copy = builder.copy(&count);
                let i_lt_count = builder.instr2(LessThan, i_copy, count_copy);
                let jump_if_false = builder.jump_if_false(i_lt_count);
                let i_copy = builder.copy(&i);
                let a = builder.unchecked_index(&arg, i_copy);
                let result = builder.instr1(instruction, a);
                builder.append(&list, result);
                let one = builder.load_const(1.0);
                builder.instr2_in_place(Add, &i, one);
                let jump = builder.jump(vec![]);
                builder.set_jump_label(jump, &loop_start);
                let loop_end = builder.label();
                builder.set_jump_label(jump_if_false, &loop_end);
                builder.pop(i);
                builder.pop(count);
                builder.swap_pop(&mut list, vec![arg]);
                Ok(list)
            } else {
                Ok(builder.instr1(instruction, arg))
            }
        }
        Expression::BinaryOperation {
            operation,
            left,
            right,
        } => {
            let mut left = compile_expression(left, builder, names)?;
            let right = compile_expression(right, builder, names)?;

            let broadcast_left = left.ty().is_list()
                && (operation != &BinaryOperator::Index || right.ty().base() == BaseType::Bool);
            let broadcast_right = right.ty().is_list();

            let broadcast_empty_left = broadcast_left && left.ty() == Type::EmptyList;
            let broadcast_empty_right = broadcast_right && right.ty() == Type::EmptyList;

            if broadcast_empty_left || broadcast_empty_right {
                if !broadcast_empty_left {
                    let base = left.ty().base();
                    match operation {
                        BinaryOperator::Add => match base {
                            BaseType::Number | BaseType::Point => {}
                            _ => {
                                return Err(format!("cannot add {} and {}", left.ty(), right.ty()))
                            }
                        },
                        BinaryOperator::Sub => match base {
                            BaseType::Number | BaseType::Point => {}
                            _ => {
                                return Err(format!(
                                    "cannot subtract {} from {}",
                                    left.ty(),
                                    right.ty()
                                ))
                            }
                        },
                        BinaryOperator::Mul | BinaryOperator::Dot | BinaryOperator::Cross => {
                            match base {
                                BaseType::Number | BaseType::Point => {}
                                _ => {
                                    return Err(format!(
                                        "cannot multiply {} by {}",
                                        left.ty(),
                                        right.ty()
                                    ))
                                }
                            }
                        }
                        BinaryOperator::Div => match base {
                            BaseType::Number | BaseType::Point => {}
                            _ => {
                                return Err(format!(
                                    "cannot divide {} by {}",
                                    left.ty(),
                                    right.ty()
                                ))
                            }
                        },
                        BinaryOperator::Pow => match base {
                            BaseType::Number => {}
                            _ => {
                                return Err(format!("cannot raise {} to {}", left.ty(), right.ty()))
                            }
                        },
                        BinaryOperator::Point => match base {
                            BaseType::Number => {}
                            _ => {
                                return Err(format!(
                                    "cannot use {} and {} as the coordinates of a point",
                                    left.ty(),
                                    right.ty()
                                ))
                            }
                        },
                        BinaryOperator::Index => match left.ty() {
                            Type::NumberList
                            | Type::PointList
                            | Type::BoolList
                            | Type::EmptyList => {}
                            _ => {
                                return Err(format!(
                                    "cannot index {} with {}",
                                    left.ty(),
                                    right.ty()
                                ))
                            }
                        },
                    }
                }
                if !broadcast_empty_right {
                    let base = right.ty().base();
                    match operation {
                        BinaryOperator::Add => match base {
                            BaseType::Number | BaseType::Point => {}
                            _ => {
                                return Err(format!("cannot add {} and {}", left.ty(), right.ty()))
                            }
                        },
                        BinaryOperator::Sub => match base {
                            BaseType::Number | BaseType::Point => {}
                            _ => {
                                return Err(format!(
                                    "cannot subtract {} from {}",
                                    left.ty(),
                                    right.ty()
                                ))
                            }
                        },
                        BinaryOperator::Mul | BinaryOperator::Dot | BinaryOperator::Cross => {
                            match base {
                                BaseType::Number | BaseType::Point => {}
                                _ => {
                                    return Err(format!(
                                        "cannot multiply {} by {}",
                                        left.ty(),
                                        right.ty()
                                    ))
                                }
                            }
                        }
                        BinaryOperator::Div => match base {
                            BaseType::Number => {}
                            _ => {
                                return Err(format!(
                                    "cannot divide {} by {}",
                                    left.ty(),
                                    right.ty()
                                ))
                            }
                        },
                        BinaryOperator::Pow => match base {
                            BaseType::Number => {}
                            _ => {
                                return Err(format!("cannot raise {} to {}", left.ty(), right.ty()))
                            }
                        },
                        BinaryOperator::Point => match base {
                            BaseType::Number => {}
                            _ => {
                                return Err(format!(
                                    "cannot use {} and {} as the coordinates of a point",
                                    left.ty(),
                                    right.ty()
                                ))
                            }
                        },
                        BinaryOperator::Index => match base {
                            BaseType::Bool => {}
                            _ => unreachable!(),
                        },
                    }
                }
                builder.pop(right);
                builder.pop(left);
                return Ok(builder.empty_list());
            }

            let (base, instruction) = match operation {
                BinaryOperator::Add => match (left.ty().base(), right.ty().base()) {
                    (BaseType::Number, BaseType::Number) => (BaseType::Number, Add),
                    (BaseType::Point, BaseType::Point) => (BaseType::Number, Add2),
                    _ => return Err(format!("cannot add {} and {}", left.ty(), right.ty())),
                },
                BinaryOperator::Sub => match (left.ty().base(), right.ty().base()) {
                    (BaseType::Number, BaseType::Number) => (BaseType::Number, Sub),
                    (BaseType::Point, BaseType::Point) => (BaseType::Number, Sub2),
                    _ => return Err(format!("cannot subtract {} from {}", right.ty(), left.ty())),
                },
                BinaryOperator::Mul => match (left.ty().base(), right.ty().base()) {
                    (BaseType::Number, BaseType::Number) => (BaseType::Number, Mul),
                    (BaseType::Number, BaseType::Point) => (BaseType::Point, Mul1_2),
                    (BaseType::Point, BaseType::Number) => (BaseType::Point, Mul2_1),
                    _ => return Err(format!("cannot multiply {} by {}", left.ty(), right.ty())),
                },
                BinaryOperator::Div => match (left.ty().base(), right.ty().base()) {
                    (BaseType::Number, BaseType::Number) => (BaseType::Number, Div),
                    (BaseType::Point, BaseType::Number) => (BaseType::Point, Div2_1),
                    _ => return Err(format!("cannot divide {} by {}", left.ty(), right.ty())),
                },
                BinaryOperator::Pow => match (left.ty().base(), right.ty().base()) {
                    (BaseType::Number, BaseType::Number) => (BaseType::Number, Pow),
                    _ => return Err(format!("cannot raise {} to {}", left.ty(), right.ty())),
                },
                BinaryOperator::Dot => match (left.ty().base(), right.ty().base()) {
                    (BaseType::Number, BaseType::Number) => (BaseType::Number, Mul),
                    (BaseType::Number, BaseType::Point) => (BaseType::Point, Mul1_2),
                    (BaseType::Point, BaseType::Number) => (BaseType::Point, Mul2_1),
                    (BaseType::Point, BaseType::Point) => (BaseType::Number, Dot),
                    _ => return Err(format!("cannot multiply {} by {}", left.ty(), right.ty())),
                },
                BinaryOperator::Cross => match (left.ty().base(), right.ty().base()) {
                    (BaseType::Number, BaseType::Number) => (BaseType::Number, Mul),
                    (BaseType::Number, BaseType::Point) => (BaseType::Point, Mul1_2),
                    (BaseType::Point, BaseType::Number) => (BaseType::Point, Mul2_1),
                    (BaseType::Point, BaseType::Point) => {
                        return Err(format!(
                            "cannot take the cross product of {} and {}",
                            left.ty(),
                            right.ty()
                        ))
                    }
                    _ => return Err(format!("cannot multiply {} by {}", left.ty(), right.ty())),
                },
                BinaryOperator::Point => match (left.ty().base(), right.ty().base()) {
                    (BaseType::Number, BaseType::Number) => (BaseType::Point, Point),
                    _ => {
                        return Err(format!(
                            "cannot use {} and {} as the coordinates of a point",
                            left.ty(),
                            right.ty()
                        ))
                    }
                },
                BinaryOperator::Index => match (left.ty(), right.ty().base()) {
                    (Type::EmptyList, BaseType::Number) => {
                        builder.collapse(&mut left, BaseType::Number);
                        (BaseType::Number, Index)
                    }
                    (Type::NumberList, BaseType::Number) => (BaseType::Number, Index),
                    (Type::PointList, BaseType::Number) => (BaseType::Point, Index2),
                    (Type::NumberList, BaseType::Bool) => (BaseType::Number, Unreachable),
                    (Type::PointList, BaseType::Bool) => (BaseType::Point, Unreachable),
                    _ => return Err(format!("cannot index {} with {}", left.ty(), right.ty())),
                },
            };

            let result = if broadcast_left || broadcast_right {
                let mut list = builder.build_list(base, vec![]);
                let count = match (broadcast_left, broadcast_right) {
                    (true, true) => {
                        let left_count = builder.count_specific(&left);
                        let right_count = builder.count_specific(&right);
                        builder.instr2(Min, left_count, right_count)
                    }
                    (true, false) => builder.count_specific(&left),
                    (false, true) => builder.count_specific(&right),
                    (false, false) => unreachable!(),
                };
                let i = builder.load_const(0.0);
                let loop_start = builder.label();
                let i_copy = builder.copy(&i);
                let count_copy = builder.copy(&count);
                let i_lt_count = builder.instr2(LessThan, i_copy, count_copy);
                let loop_jump_if_false = builder.jump_if_false(i_lt_count);
                if operation == &BinaryOperator::Index && right.ty().base() == BaseType::Bool {
                    let r = if broadcast_right {
                        let i_copy = builder.copy(&i);
                        builder.unchecked_index(&right, i_copy)
                    } else {
                        builder.copy(&right)
                    };
                    let filter_jump_if_false = builder.jump_if_false(r);
                    let l = if broadcast_left {
                        let i_copy = builder.copy(&i);
                        builder.unchecked_index(&left, i_copy)
                    } else {
                        builder.copy(&left)
                    };
                    builder.append(&list, l);
                    builder.set_jump_label(filter_jump_if_false, &builder.label());
                } else {
                    let l = if broadcast_left {
                        let i_copy = builder.copy(&i);
                        builder.unchecked_index(&left, i_copy)
                    } else {
                        builder.copy(&left)
                    };
                    let r = if broadcast_right {
                        let i_copy = builder.copy(&i);
                        builder.unchecked_index(&right, i_copy)
                    } else {
                        builder.copy(&right)
                    };
                    let result = builder.instr2(instruction, l, r);
                    builder.append(&list, result);
                }
                let one = builder.load_const(1.0);
                builder.instr2_in_place(Add, &i, one);
                let loop_jump = builder.jump(vec![]);
                builder.set_jump_label(loop_jump, &loop_start);
                let loop_end = builder.label();
                builder.set_jump_label(loop_jump_if_false, &loop_end);
                builder.pop(i);
                builder.pop(count);
                builder.swap_pop(&mut list, vec![left, right]);
                list
            } else {
                builder.instr2(instruction, left, right)
            };

            Ok(result)
        }
        Expression::ChainedComparison {
            operands,
            operators,
        } => {
            let ([left, right], [operator]) = (&operands[..], &operators[..]) else {
                todo!()
            };
            let left = compile_expression(left, builder, names)?;
            let right = compile_expression(right, builder, names)?;

            let broadcast_left = left.ty().is_list();
            let broadcast_right = right.ty().is_list();

            let broadcast_empty_left = broadcast_left && left.ty() == Type::EmptyList;
            let broadcast_empty_right = broadcast_right && right.ty() == Type::EmptyList;

            if broadcast_empty_left || broadcast_empty_right {
                if !broadcast_empty_left {
                    match left.ty().base() {
                        BaseType::Number => {}
                        _ => return Err(format!("cannot compare {} to {}", left.ty(), right.ty())),
                    }
                }
                if !broadcast_empty_right {
                    match right.ty().base() {
                        BaseType::Number => {}
                        _ => return Err(format!("cannot compare {} to {}", left.ty(), right.ty())),
                    }
                }
                builder.pop(right);
                builder.pop(left);
                return Ok(builder.build_list(BaseType::Bool, vec![]));
            }

            let base = match (left.ty().base(), right.ty().base()) {
                (BaseType::Number, BaseType::Number) => BaseType::Bool,
                _ => return Err(format!("cannot compare {} and {}", left.ty(), right.ty())),
            };
            let instruction = match operator {
                ComparisonOperator::Equal => Equal,
                ComparisonOperator::Less => LessThan,
                ComparisonOperator::LessEqual => LessThanEqual,
                ComparisonOperator::Greater => GreaterThan,
                ComparisonOperator::GreaterEqual => GreaterThanEqual,
            };

            let result = if broadcast_left || broadcast_right {
                let mut list = builder.build_list(base, vec![]);
                let count = match (broadcast_left, broadcast_right) {
                    (true, true) => {
                        let left_count = builder.count_specific(&left);
                        let right_count = builder.count_specific(&right);
                        builder.instr2(Min, left_count, right_count)
                    }
                    (true, false) => builder.count_specific(&left),
                    (false, true) => builder.count_specific(&right),
                    (false, false) => unreachable!(),
                };
                let i = builder.load_const(0.0);
                let loop_start = builder.label();
                let i_copy = builder.copy(&i);
                let count_copy = builder.copy(&count);
                let i_lt_count = builder.instr2(LessThan, i_copy, count_copy);
                let jump_if_false = builder.jump_if_false(i_lt_count);
                let l = if broadcast_left {
                    let i_copy = builder.copy(&i);
                    builder.unchecked_index(&left, i_copy)
                } else {
                    builder.copy(&left)
                };
                let r = if broadcast_right {
                    let i_copy = builder.copy(&i);
                    builder.unchecked_index(&right, i_copy)
                } else {
                    builder.copy(&right)
                };
                let result = builder.instr2(instruction, l, r);
                builder.append(&list, result);
                let one = builder.load_const(1.0);
                builder.instr2_in_place(Add, &i, one);
                let jump = builder.jump(vec![]);
                builder.set_jump_label(jump, &loop_start);
                let loop_end = builder.label();
                builder.set_jump_label(jump_if_false, &loop_end);
                builder.pop(i);
                builder.pop(count);
                builder.swap_pop(&mut list, vec![left, right]);
                list
            } else {
                builder.instr2(instruction, left, right)
            };

            Ok(result)
        }
        Expression::Piecewise {
            test,
            consequent,
            alternate,
        } => {
            let mut consequent = compile_expression(consequent, builder, names)?; // T, ListOfT, EmptyList
            let mut alternate = if let Some(alternate) = alternate {
                let alternate = compile_expression(alternate, builder, names)?;
                if consequent.ty() != Type::EmptyList
                    && alternate.ty() != Type::EmptyList
                    && alternate.ty().base() != consequent.ty().base()
                {
                    return Err(format!(
                        "cannot use {} and {} as the branches in a piecewise, every branch must have the same type",
                        consequent.ty(),
                        alternate.ty()
                    ));
                }
                alternate
            } else {
                match consequent.ty() {
                    Type::Number | Type::NumberList => builder.load_const(f64::NAN),
                    Type::Point | Type::PointList => {
                        let x = builder.load_const(f64::NAN);
                        let y = builder.load_const(f64::NAN);
                        builder.instr2(Point, x, y)
                    }
                    Type::EmptyList => builder.empty_list(),
                    Type::Bool | Type::BoolList => unreachable!(),
                }
            };
            let test = compile_expression(test, builder, names)?;
            assert!(test.ty().base() == BaseType::Bool);

            let broadcast_test = test.ty().is_list();
            let broadcast_consequent =
                consequent.ty().is_list() && (broadcast_test || !alternate.ty().is_list());
            let broadcast_alternate =
                alternate.ty().is_list() && (broadcast_test || !consequent.ty().is_list());

            if (broadcast_consequent && consequent.ty() == Type::EmptyList)
                || (broadcast_alternate && alternate.ty() == Type::EmptyList)
                || (consequent.ty() == Type::EmptyList && alternate.ty() == Type::EmptyList)
            {
                builder.pop(test);
                builder.pop(alternate);
                builder.pop(consequent);
                return Ok(builder.empty_list());
            }

            if consequent.ty() == Type::EmptyList {
                assert!(alternate.ty().is_list());
                builder.collapse(&mut consequent, alternate.ty().base());
            }

            if alternate.ty() == Type::EmptyList {
                assert!(consequent.ty().is_list());
                builder.collapse(&mut alternate, consequent.ty().base());
            }

            if broadcast_test || broadcast_consequent || broadcast_alternate {
                assert_ne!(consequent.ty(), Type::EmptyList);
                let mut list = builder.build_list(consequent.ty().base(), vec![]);
                let mut counts = [
                    (broadcast_test, &test),
                    (broadcast_consequent, &consequent),
                    (broadcast_alternate, &alternate),
                ]
                .iter()
                .filter(|(b, _)| *b)
                .map(|(_, l)| builder.count_specific(l))
                .collect::<Vec<_>>();
                let mut count = counts.pop().unwrap();
                while let Some(c) = counts.pop() {
                    count = builder.instr2(Min, c, count);
                }
                let i = builder.load_const(0.0);
                let loop_start = builder.label();
                let i_copy = builder.copy(&i);
                let count_copy = builder.copy(&count);
                let i_lt_count = builder.instr2(LessThan, i_copy, count_copy);
                let loop_jump_if_false = builder.jump_if_false(i_lt_count);
                let t = if broadcast_test {
                    let i_copy = builder.copy(&i);
                    builder.unchecked_index(&test, i_copy)
                } else {
                    builder.copy(&test)
                };
                let pw_jump_if_false = builder.jump_if_false(t);
                let result = if broadcast_consequent {
                    let i_copy = builder.copy(&i);
                    builder.unchecked_index(&consequent, i_copy)
                } else {
                    builder.copy(&consequent)
                };
                let pw_jump = builder.jump(vec![result]);
                builder.set_jump_label(pw_jump_if_false, &builder.label());
                let result = if broadcast_alternate {
                    let i_copy = builder.copy(&i);
                    builder.unchecked_index(&alternate, i_copy)
                } else {
                    builder.copy(&alternate)
                };
                builder.set_jump_label(pw_jump, &builder.label());
                builder.append(&list, result);
                let one = builder.load_const(1.0);
                builder.instr2_in_place(Add, &i, one);
                let loop_jump = builder.jump(vec![]);
                builder.set_jump_label(loop_jump, &loop_start);
                let loop_end = builder.label();
                builder.set_jump_label(loop_jump_if_false, &loop_end);
                builder.pop(i);
                builder.pop(count);
                builder.swap_pop(&mut list, vec![consequent, alternate, test]);
                Ok(list)
            } else {
                let pw_jump_if_false = builder.jump_if_false(test);
                let result = builder.copy(&consequent);
                let pw_jump = builder.jump(vec![result]);
                builder.set_jump_label(pw_jump_if_false, &builder.label());
                let mut result = builder.copy(&alternate);
                builder.set_jump_label(pw_jump, &builder.label());
                builder.swap_pop(&mut result, vec![consequent, alternate]);
                Ok(result)
            }
        }
        Expression::SumProd { .. } => todo!(),
        Expression::For { .. } => todo!(),
    }
}

pub fn compile_assignments(
    assignments: &[Assignment],
    assignment_indices: &[Option<Result<usize, String>>],
) -> Result<(Vec<Instruction>, Vec<Option<Result<(Type, usize), String>>>), String> {
    let mut names = Names::default();
    let mut builder = InstructionBuilder::default();

    for Assignment { id, value } in assignments {
        let value = compile_expression(value, &mut builder, &names)?;
        names.create(*id, value.ty())?;
        builder.store(*id, value);
    }

    let vars = builder.defined_vars();
    let program = builder.finish();
    Ok((
        program,
        assignment_indices
            .iter()
            .map(|i| match i {
                Some(Ok(index)) => Some(Ok(*vars.get(index).unwrap())),
                Some(Err(error)) => Some(Err(error.clone())),
                None => None,
            })
            .collect(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use assert_matches::assert_matches;
    use pretty_assertions::assert_eq;
    use BinaryOperator as Bo;
    use Expression as E;
    use InstructionBuilder as Ib;

    fn bx<T>(x: T) -> Box<T> {
        Box::new(x)
    }

    fn assert_type(v: Result<Value, String>, t: Type) {
        assert_matches!(v, Ok(..));
        let Ok(v) = v else {
            panic!("assertion `left == right` failed\n  left: {v:?}\n right: Ok({t:?})");
        };
        assert_eq!(v.ty(), t);
    }

    #[test]
    fn number() {
        let mut b = Ib::default();
        assert_type(
            compile_expression(&E::Number(5.0), &mut b, &mut Names::default()),
            Type::Number,
        );
        assert_type(
            compile_expression(&E::Number(3.0), &mut b, &mut Names::default()),
            Type::Number,
        );
        assert_eq!(b.finish(), [LoadConst(5.0), LoadConst(3.0)]);
    }

    fn create_empty_variable(
        names: &mut Names,
        builder: &mut InstructionBuilder,
        name: usize,
        ty: Type,
    ) {
        names.create(name, ty).unwrap();
        let v = match ty {
            Type::Number => builder.load_const(0.0),
            Type::NumberList => builder.build_list(BaseType::Number, vec![]),
            Type::Point => {
                let x = builder.load_const(0.0);
                let y = builder.load_const(0.0);
                builder.instr2(Point, x, y)
            }
            Type::PointList => builder.build_list(BaseType::Point, vec![]),
            Type::EmptyList => builder.empty_list(),
            Type::Bool | Type::BoolList => panic!("bruh"),
        };
        builder.store(name, v);
    }

    #[test]
    fn identifiers() {
        assert_eq!(
            compile_expression(&E::Identifier(0), &mut Ib::default(), &mut Names::default()),
            Err("variable '0' is not defined".to_string())
        );

        let mut names = Names::default();
        let mut b = Ib::default();
        create_empty_variable(&mut names, &mut b, 0, Type::Number);
        assert_eq!(names.get(0), Ok(Type::Number));

        assert_type(
            compile_expression(&E::Identifier(0), &mut b, &mut names),
            Type::Number,
        );

        create_empty_variable(&mut names, &mut b, 1, Type::Point);
        assert_eq!(names.get(1), Ok(Type::Point));
        assert_type(
            compile_expression(&E::Identifier(1), &mut b, &mut names),
            Type::Point,
        );

        create_empty_variable(&mut names, &mut b, 2, Type::PointList);
        assert_eq!(names.get(2), Ok(Type::PointList));
        assert_type(
            compile_expression(&E::Identifier(2), &mut b, &mut names),
            Type::PointList,
        );

        create_empty_variable(&mut names, &mut b, 3, Type::NumberList);
        assert_eq!(names.get(3), Ok(Type::NumberList));
        assert_type(
            compile_expression(&E::Identifier(3), &mut b, &mut names),
            Type::NumberList,
        );

        assert_type(
            compile_expression(&E::Identifier(1), &mut b, &mut names),
            Type::Point,
        );
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
    fn bop_point() {
        let mut b = Ib::default();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Point,
                    left: bx(E::Number(1.0)),
                    right: bx(E::Number(2.0)),
                },
                &mut b,
                &mut Names::default(),
            ),
            Type::Point,
        );
        assert_type(
            compile_expression(&E::Number(3.0), &mut b, &mut Names::default()),
            Type::Number,
        );
        assert_eq!(b.finish(), [LoadConst(1.0), LoadConst(2.0), LoadConst(3.0)]);

        let mut names = Names::default();
        let mut b = Ib::default();
        create_empty_variable(&mut names, &mut b, 0, Type::PointList);
        assert_eq!(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Point,
                    left: bx(E::Identifier(0)),
                    right: bx(E::Number(2.0))
                },
                &mut b,
                &mut names
            ),
            Err("cannot use a list of points and a number as the coordinates of a point".into())
        );

        let mut names = Names::default();
        let mut b = Ib::default();
        create_empty_variable(&mut names, &mut b, 1, Type::NumberList);
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Point,
                    left: bx(E::Identifier(1)),
                    right: bx(E::Number(5.0)),
                },
                &mut b,
                &mut names,
            ),
            Type::PointList,
        );
        let mut expected = vec![];
        expected.extend([
            BuildList(0),
            Store(0),
            Load(0),          // xs
            LoadConst(5.0),   // xs, 5
            BuildList(0),     // xs, 5, []
            CountSpecific(3), // xs, 5, [], xs.count
            LoadConst(0.0),   // xs, 5, [], xs.count, 0
        ]);
        let j = expected.len();
        expected.extend([
            Copy(1),  // xs, 5, [], xs.count, 0, 0
            Copy(3),  // xs, 5, [], xs.count, 0, 0, xs.count
            LessThan, // xs, 5, [], xs.count, 0, 0 < xs.count
        ]);
        let jif = expected.len();
        expected.extend([
            JumpIfFalse(!0),   //  xs, 5, [], xs.count, 0
            Copy(1),           //  xs, 5, [], xs.count, 0, 0
            UncheckedIndex(5), //  xs, 5, [], xs.count, 0, x
            Copy(5),           //  xs, 5, [], xs.count, 0, x, 5
            Append2(3),        //  xs, 5, [x,5], xs.count, 0
            LoadConst(1.0),    //  xs, 5, [x,5], xs.count, 0, 1
            Add,               //  xs, 5, [x,5], xs.count, 1
            Jump(j),           //  xs, 5, [x,5], xs.count, 1
        ]);
        expected[jif] = JumpIfFalse(expected.len());
        expected.extend([
            Pop(1),  //  xs, 5, [x,5,...], xs.count
            Pop(1),  //  xs, 5, [x,5,...]
            Swap(3), //  [x,5,...], xs, 5
            Pop(2),  //  [x,5,...]
        ]);

        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Point,
                    left: bx(E::Number(5.0)),
                    right: bx(E::Identifier(1)),
                },
                &mut b,
                &mut names,
            ),
            Type::PointList,
        );
        expected.extend([
            LoadConst(5.0),   // 5
            Load(0),          // 5, xs
            BuildList(0),     // 5, xs, []
            CountSpecific(2), // 5, xs, [], xs.count
            LoadConst(0.0),   // 5, xs, [], xs.count, 0
        ]);
        let j = expected.len();
        expected.extend([
            Copy(1),  // 5, xs, [], xs.count, 0, 0
            Copy(3),  // 5, xs, [], xs.count, 0, 0, xs.count
            LessThan, // 5, xs, [], xs.count, 0, 0 < xs.count
        ]);
        let jif = expected.len();
        expected.extend([
            JumpIfFalse(!0),   //  5, xs, [], xs.count, 0
            Copy(5),           //  5, xs, [], xs.count, 0, 5
            Copy(2),           //  5, xs, [], xs.count, 0, 5, 0
            UncheckedIndex(5), //  5, xs, [], xs.count, 0, 5, x
            Append2(3),        //  5, xs, [x,5], xs.count, 0
            LoadConst(1.0),    //  5, xs, [x,5], xs.count, 0, 1
            Add,               //  5, xs, [x,5], xs.count, 1
            Jump(j),           //  5, xs, [x,5], xs.count, 1
        ]);
        expected[jif] = JumpIfFalse(expected.len());
        expected.extend([
            Pop(1),  //  5, xs, [x,5,...], xs.count
            Pop(1),  //  5, xs, [x,5,...]
            Swap(3), //  [x,5,...], 5, xs
            Pop(2),  //  [x,5,...]
        ]);

        create_empty_variable(&mut names, &mut b, 2, Type::NumberList);
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Point,
                    left: bx(E::Identifier(1)),
                    right: bx(E::Identifier(2)),
                },
                &mut b,
                &mut names,
            ),
            Type::PointList,
        );
        expected.extend([
            BuildList(0),
            Store(2),
            Load(0),          // a
            Load(2),          // a, b
            BuildList(0),     // a, b, []
            CountSpecific(3), // a, b, [], a.count
            CountSpecific(3), // a, b, [], a.count, b.count
            Min,              // a, b, [], min(a.count,b.count)
            LoadConst(0.0),   // a, b, [], min(a.count,b.count), i
        ]);
        let j = expected.len();
        expected.extend([
            Copy(1),  // a, b, [], min(a.count,b.count), i, i
            Copy(3),  // a, b, [], min(a.count,b.count), i, i, min(a.count,b.count)
            LessThan, // a, b, [], min(a.count,b.count), i, i < min(a.count,b.count)
        ]);
        let jif = expected.len();
        expected.extend([
            JumpIfFalse(!0),   //  a, b, [], min(a.count,b.count), i
            Copy(1),           //  a, b, [], min(a.count,b.count), i, i
            UncheckedIndex(5), //  a, b, [], min(a.count,b.count), i, a[i]
            Copy(2),           //  a, b, [], min(a.count,b.count), i, a[i], i
            UncheckedIndex(5), //  a, b, [], min(a.count,b.count), i, a[i], b[i]
            Append2(3),        //  a, b, [a[i],b[i]], min(a.count,b.count), i
            LoadConst(1.0),    //  a, b, [a[i],b[i]], min(a.count,b.count), i, 1
            Add,               //  a, b, [a[i],b[i]], min(a.count,b.count), i+1
            Jump(j),           //  a, b, [a[i],b[i]], min(a.count,b.count), i+1
        ]);
        expected[jif] = JumpIfFalse(expected.len());
        expected.extend([
            Pop(1),  //  a, b, [a[i],b[i],...], min(a.count,b.count)
            Pop(1),  //  a, b, [a[i],b[i],...]
            Swap(3), //  [a[i],b[i],...], a, b
            Pop(2),  //  [a[i],b[i],...]
        ]);
        assert_eq!(b.finish(), expected);
    }

    #[test]
    fn bop() {
        let make_names_and_builder = || {
            let mut n = Names::default();
            let mut b = Ib::default();
            create_empty_variable(&mut n, &mut b, 0, Type::Number);
            create_empty_variable(&mut n, &mut b, 1, Type::NumberList);
            create_empty_variable(&mut n, &mut b, 2, Type::Point);
            create_empty_variable(&mut n, &mut b, 3, Type::PointList);
            create_empty_variable(&mut n, &mut b, 4, Type::EmptyList);
            (n, b)
        };
        let number = || bx(E::Identifier(0));
        let number_list = || bx(E::Identifier(1));
        let point = || bx(E::Identifier(2));
        let _point_list = || bx(E::Identifier(3));
        let empty_list = || bx(E::Identifier(4));

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Add,
                    left: number(),
                    right: number(),
                },
                &mut b,
                &mut n,
            ),
            Type::Number,
        );
        assert_eq!(b.finish()[11..], [Load(0), Load(0), Add]);

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Add,
                    left: point(),
                    right: point(),
                },
                &mut b,
                &mut n,
            ),
            Type::Point,
        );
        assert_eq!(b.finish()[11..], [Load2(4), Load2(4), Add2]);

        let (mut n, mut b) = make_names_and_builder();
        assert_eq!(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Add,
                    left: point(),
                    right: number()
                },
                &mut b,
                &mut n
            ),
            Err("cannot add a point and a number".into())
        );

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Mul,
                    left: point(),
                    right: number(),
                },
                &mut b,
                &mut n,
            ),
            Type::Point,
        );
        assert_eq!(b.finish()[11..], [Load2(4), Load(0), Mul2_1]);

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Mul,
                    left: number(),
                    right: point(),
                },
                &mut b,
                &mut n,
            ),
            Type::Point,
        );
        assert_eq!(b.finish()[11..], [Load(0), Load2(4), Mul1_2]);

        let (mut n, mut b) = make_names_and_builder();
        assert_eq!(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Mul,
                    left: point(),
                    right: point(),
                },
                &mut b,
                &mut n
            ),
            Err("cannot multiply a point by a point".into())
        );

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Div,
                    left: point(),
                    right: number(),
                },
                &mut b,
                &mut n,
            ),
            Type::Point,
        );
        assert_eq!(b.finish()[11..], [Load2(4), Load(0), Div2_1]);

        let (mut n, mut b) = make_names_and_builder();
        assert_eq!(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Div,
                    left: number(),
                    right: point(),
                },
                &mut b,
                &mut n
            ),
            Err("cannot divide a number by a point".into())
        );

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Pow,
                    left: number(),
                    right: number(),
                },
                &mut b,
                &mut n,
            ),
            Type::Number,
        );
        assert_eq!(b.finish()[11..], [Load(0), Load(0), Pow]);

        let (mut n, mut b) = make_names_and_builder();
        assert_eq!(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Pow,
                    left: number(),
                    right: point(),
                },
                &mut b,
                &mut n
            ),
            Err("cannot raise a number to a point".into())
        );

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Dot,
                    left: point(),
                    right: point(),
                },
                &mut b,
                &mut n,
            ),
            Type::Number,
        );
        assert_eq!(b.finish()[11..], [Load2(4), Load2(4), Dot]);

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Dot,
                    left: point(),
                    right: empty_list(),
                },
                &mut b,
                &mut n,
            ),
            Type::EmptyList,
        );
        assert_eq!(
            b.finish()[11..],
            [Load2(4), Load(8), Pop(1), Pop(2), BuildList(0)]
        );

        let (mut n, mut b) = make_names_and_builder();
        assert_eq!(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Index,
                    left: point(),
                    right: empty_list(),
                },
                &mut b,
                &mut n,
            ),
            Err("cannot index a point with an empty list".into()),
        );

        let (mut n, mut b) = make_names_and_builder();
        assert_eq!(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Pow,
                    left: point(),
                    right: empty_list(),
                },
                &mut b,
                &mut n,
            ),
            Err("cannot raise a point to an empty list".into()),
        );

        let (mut n, mut b) = make_names_and_builder();
        assert_type(
            compile_expression(
                &E::BinaryOperation {
                    operation: Bo::Index,
                    left: number_list(),
                    right: empty_list(),
                },
                &mut b,
                &mut n,
            ),
            Type::EmptyList,
        );
    }

    // #[test]
    // fn chained_comparison() {
    //     let mut builder = Ib::default();
    //     let mut names = Names::default();
    //     assert_type(
    //         compile_expression(
    //             &E::ChainedComparison(Cc {
    //                 operands: vec![E::Number(1.0), E::Number(2.0)],
    //                 operators: vec![Co::Less],
    //             }),
    //             &mut builder,
    //             &mut names,
    //         ),
    //         Type::Bool,
    //     );
    // }
}
