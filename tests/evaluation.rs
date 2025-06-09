use ambavia::{
    ast_parser::parse_nodes_into_expression, compiler::{compile_expression, Names}, instruction_builder::InstructionBuilder, latex_parser::parse_latex, latex_tree_flattener::Token, resolver::resolve_names, vm::{self, Vm}
};
use rstest::rstest;

#[derive(Debug)]
enum Value {
    Number(f64),
    Point(f64, f64),
    List(Vec<f64>),
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Number(l), Self::Number(r)) => l.total_cmp(r).is_eq(),
            (Self::List(l), Self::List(r)) => {
                l.len() == r.len() && l.iter().zip(r.iter()).all(|(l, r)| l.total_cmp(r).is_eq())
            }
            (Self::Point(..), _) | (_, Self::Point(..)) => {
                panic!("point should have been turned into numbers")
            }
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
        Value::List(value.into())
    }
}

impl<const N: usize> From<[i64; N]> for Value {
    fn from(value: [i64; N]) -> Self {
        (&value.map(|x| x as f64)).into()
    }
}

fn assert_expression_eq<'a>(source: &str, value: Value) {
    println!("expression: {source}");
    let tree = parse_latex(source).unwrap();
    let ast = parse_nodes_into_expression(&tree, Token::EndOfInput).unwrap();
    let mut builder = InstructionBuilder::default();
    let mut names = Names::default();
    compile_expression(&ast, &mut builder, &mut names).unwrap();
    let instructions = builder.finish();

    println!("Compiled instructions:");
    let width = (instructions.len() - 1).to_string().len();
    for (i, instruction) in instructions.iter().enumerate() {
        println!("  {i:width$} {instruction:?}");
    }

    let mut vm = Vm::with_program(instructions);
    vm.run(false);
    let stack: &[Value] = match value.into() {
        Value::Point(x, y) => &[Value::Number(x), Value::Number(y)],
        other @ _ => &[other],
    };
    assert_eq!(
        vm.stack
            .iter()
            .map(|v| match v {
                vm::Value::Number(x) => Value::Number(*x),
                vm::Value::List(list) => Value::List(list.borrow().to_owned()),
            })
            .collect::<Vec<_>>(),
        stack
    );
}

// fn assert_ast_error(source: &str) {
//     println!("expression: {source}");
//     let tree = parse_latex(source).unwrap();
//     assert_matches::assert_matches!(parse_nodes_into_expression(&tree, Token::EndOfInput), Err(..));
// }

fn assert_compile_error(source: &str, error: &str) {
    println!("expression: {source}");
    let tree = parse_latex(source).unwrap();
    let ast = parse_nodes_into_expression(&tree, Token::EndOfInput).unwrap();
    let mut builder = InstructionBuilder::default();
    let mut names = Names::default();
    assert_eq!(
        compile_expression(&ast, &mut builder, &mut names),
        Err(error.into())
    );
}

const NAN: f64 = f64::NAN;

#[rstest]
#[case(r"1 + [2,3]", [3, 4])]
#[case(r"[10,20,30] + [2,3]", [12, 23])]
#[case(r"(1, [3,4])", [1, 3, 1, 4])]
#[case(r"([1...2], [3...5])", [1, 3, 2, 4])]
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
#[case(r"\{1<2: (3,4), []\}", [])]
#[case(r"\{1<2: [(3,4)], []\}", [3, 4])]
#[case(r"\{1<2: (3,4), [(5,6),(7,8)]\}", [3, 4, 3, 4])]
#[case(r"[1,2,3][4=4]", [1, 2, 3])]
#[case(r"[1,2,3][4=5]", [])]
#[case(r"[(1,2), (3,4), (5,6)][[7...9]>=8]", [3, 4, 5, 6])]
#[case(r"[1,2][[7...9]>=8]", [2])]
#[case(r"[][[7...9]>=8] + (1,2)", [])]
fn expression_eq(#[case] expression: &str, #[case] expected: impl Into<Value>) {
    assert_expression_eq(expression, expected.into());
}

#[rstest]
#[case(r"\{1<2:(3,4),5\}", "cannot use a point and a number as the branches in a piecewise, every branch must have the same type")]
fn expression_compile_error(#[case] expression: &str, #[case] error: &str) {
    assert_compile_error(expression, error);
}
