use std::iter::zip;

use ambavia::{compiler::compile_assignments, vm::Vm};
use parse::{
    ast_parser::parse_expression_list_entry,
    latex_parser::parse_latex,
    name_resolver::resolve_names,
    type_checker::{Type, type_check},
};
use rstest::rstest;

#[derive(Debug)]
enum Value {
    Number(f64),
    Point(f64, f64),
    NumberList(Vec<f64>),
    PointList(Vec<(f64, f64)>),
    EmptyList,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        let eq = |x: &f64, y: &f64| x.total_cmp(y).is_eq();
        match (self, other) {
            (Self::Number(l), Self::Number(r)) => eq(l, r),
            (Self::NumberList(l), Self::NumberList(r)) => {
                l.len() == r.len() && zip(l, r).all(|(l, r)| eq(l, r))
            }
            (Self::Point(lx, ly), Self::Point(rx, ry)) => eq(lx, rx) && eq(ly, ry),
            (Self::PointList(l), Self::PointList(r)) => {
                l.len() == r.len() && zip(l, r).all(|((lx, ly), (rx, ry))| eq(lx, rx) && eq(ly, ry))
            }
            (Self::EmptyList, Self::EmptyList) => true,
            _ => false,
        }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Number(value)
    }
}

impl From<i64> for Value {
    fn from(value: i64) -> Self {
        (value as f64).into()
    }
}

impl From<(f64, f64)> for Value {
    fn from(value: (f64, f64)) -> Self {
        Value::Point(value.0, value.1)
    }
}

impl From<(i64, i64)> for Value {
    fn from(value: (i64, i64)) -> Self {
        (value.0 as f64, value.1 as f64).into()
    }
}

impl<const N: usize> From<&[f64; N]> for Value {
    fn from(value: &[f64; N]) -> Self {
        Value::NumberList(value.into())
    }
}

impl<const N: usize> From<[i64; N]> for Value {
    fn from(value: [i64; N]) -> Self {
        (&value.map(|x| x as f64)).into()
    }
}

impl<const N: usize> From<&[(f64, f64); N]> for Value {
    fn from(value: &[(f64, f64); N]) -> Self {
        Value::PointList(value.into())
    }
}

impl<const N: usize> From<[(i64, i64); N]> for Value {
    fn from(value: [(i64, i64); N]) -> Self {
        (&value.map(|(x, y)| (x as f64, y as f64))).into()
    }
}

fn assert_expression_eq<'a>(source: &str, value: Value) {
    println!("expression: {source}");
    let tree = parse_latex(source).unwrap();
    let entry = parse_expression_list_entry(&tree).unwrap();
    let (assignments, ei_to_nr) = resolve_names([entry].as_slice().as_ref());
    let index = ei_to_nr.first().unwrap().clone().unwrap().unwrap();
    let (assignments, nr_to_tc) = type_check(assignments.as_slice().as_ref());
    let index = nr_to_tc[index].clone().unwrap();
    let ty = assignments[index].value.ty;
    let (instructions, vars) = compile_assignments(&assignments);

    println!("Compiled instructions:");
    let width = (instructions.len() - 1).to_string().len();
    for (i, instruction) in instructions.iter().enumerate() {
        println!("  {i:width$} {instruction:?}");
    }

    let index = vars[index];

    let mut vm = Vm::with_program(instructions);
    vm.run(false);
    assert_eq!(
        match ty {
            Type::Number => Value::Number(vm.vars[index].clone().number()),
            Type::NumberList => {
                let l = vm.vars[index].clone().list();
                let list = l.borrow().clone();
                Value::NumberList(list)
            }
            Type::Point => Value::Point(
                vm.vars[index].clone().number(),
                vm.vars[index + 1].clone().number()
            ),
            Type::PointList => {
                let a = vm.vars[index].clone().list();
                let list = a
                    .borrow()
                    .chunks(2)
                    .map(|p| (p[0], p[1]))
                    .collect::<Vec<_>>();
                Value::PointList(list)
            }
            Type::Bool | Type::BoolList => unreachable!(),
            Type::EmptyList => Value::EmptyList,
        },
        value
    );
}

// fn assert_ast_error(source: &str) {
//     println!("expression: {source}");
//     let tree = parse_latex(source).unwrap();
//     assert_matches::assert_matches!(parse_nodes_into_expression(&tree, Token::EndOfInput), Err(..));
// }

fn assert_type_error(source: &str, error: &str) {
    println!("expression: {source}");
    let tree = parse_latex(source).unwrap();
    let entry = parse_expression_list_entry(&tree).unwrap();
    let (assignments, ei_to_nr) = resolve_names([entry].as_slice().as_ref());
    let index = ei_to_nr.first().unwrap().clone().unwrap().unwrap();
    let errors = type_check(assignments.as_slice().as_ref()).1;
    assert_eq!(errors[index], Err(error.into()));
}

const NAN: f64 = f64::NAN;

#[rstest]
#[case(r"1 + [2,3]", [3, 4])]
#[case(r"[10,20,30] + [2,3]", [12, 23])]
#[case(r"(1, [3,4])", [(1, 3), (1, 4)])]
#[case(r"([1...2], [3...5])", [(1, 3), (2, 4)])]
#[case(r"1 + 3", 4)]
#[case(r"(3,4) \cdot (5,6)", 39)]
#[case(r"(3,4) + (5,6)", (8, 10))]
#[case(r"\{\}", 1)]
#[case(r"\{1<2\}", 1)]
#[case(r"\{2<1\}", NAN)]
#[case(r"\{1>2: 3, 4<5: 6\}", 6)]
#[case(r"\{2<1, 5\}", 5)]
#[case(r"\{1<2, 5\}", 1)]
#[case(r"\{[1...3]<=2: 5, [7...9]\}", [5, 5, 9])]
#[case(r"\{1<2: (3,4)\}", (3, 4))]
#[case(r"\{1>2: (3,4)\}", (NAN, NAN))]
#[case(r"\{1<2: (3,4), []\}", [(0, 0); 0])]
#[case(r"\{1<2: [(3,4)], []\}", [(3, 4)])]
#[case(r"\{1<2: (3,4), [(5,6),(7,8)]\}", [(3, 4), (3, 4)])]
#[case(r"[1,2,3][4=4]", [1, 2, 3])]
#[case(r"[1,2,3][4=5]", [0; 0])]
#[case(r"[(1,2), (3,4), (5,6)][[7...9]>=8]", [(3, 4), (5, 6)])]
#[case(r"[1,2][[7...9]>=8]", [2])]
#[case(r"[][[7...9]>=8] + (1,2)", [(0, 0); 0])]
fn expression_eq(#[case] expression: &str, #[case] expected: impl Into<Value>) {
    assert_expression_eq(expression, expected.into());
}

#[rstest]
#[case(
    r"\{1<2:(3,4),5\}",
    "cannot use a point and a number as the branches in a piecewise, every branch must have the same type"
)]
fn expression_type_error(#[case] expression: &str, #[case] error: &str) {
    assert_type_error(expression, error);
}
