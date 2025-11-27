use std::iter::zip;

use eval::{compiler::compile_assignments, vm::Vm};
use parse::{
    analyze_expression_list::{AnalysisError, ExpressionResult, analyze_expression_list},
    ast_parser::parse_statement,
    latex_parser::parse_latex,
    name_resolver::{Domain, ExpressionListEntry},
    op::{OpError, OpName},
    type_checker::{Type, TypeError},
};
use rstest::rstest;

#[derive(Debug)]
enum Value {
    Number(f64),
    Point(f64, f64),
    Polygon(Vec<(f64, f64)>),
    NumberList(Vec<f64>),
    PointList(Vec<(f64, f64)>),
    PolygonList(Vec<Vec<(f64, f64)>>),
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
            (Self::PointList(l), Self::PointList(r)) | (Self::Polygon(l), Self::Polygon(r)) => {
                l.len() == r.len() && zip(l, r).all(|((lx, ly), (rx, ry))| eq(lx, rx) && eq(ly, ry))
            }
            (Self::PolygonList(l), Self::PolygonList(r)) => {
                l.len() == r.len()
                    && zip(l, r).all(|(l, r)| {
                        l.len() == r.len()
                            && zip(l, r).all(|((lx, ly), (rx, ry))| eq(lx, rx) && eq(ly, ry))
                    })
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

fn polygon<const N: usize>(points: [(i64, i64); N]) -> Value {
    Value::Polygon(points.map(|(x, y)| (x as f64, y as f64)).into())
}
#[track_caller]
fn assert_expression_eq<'a>(source: &str, value: Value) {
    println!("Expression: {source}");
    let tree = parse_latex(source).unwrap();
    let statement = parse_statement(&tree).unwrap();
    let analysis = analyze_expression_list(
        [ExpressionListEntry {
            expression: &statement,
            parametric_domain: Domain::ZERO_TO_ONE,
        }]
        .as_slice()
        .as_ref(),
        false,
    );
    let Some(ExpressionResult::Value(id, ty)) = analysis.results.first() else {
        panic!(
            "expected Value but first result was {:#?}",
            analysis.results.first()
        );
    };
    let (instructions, _, var_indices) =
        compile_assignments::<std::iter::Empty<_>, &[_]>(&analysis.assignments, []);

    println!("Compiled instructions:");
    let width = (instructions.len() - 1).to_string().len();
    for (i, instruction) in instructions.iter().enumerate() {
        println!("  {i:width$} {instruction:?}");
    }

    let v = var_indices[id];

    let mut vm = Vm::new(&instructions, Default::default());
    vm.run(false);
    assert_eq!(
        match ty {
            Type::Number => Value::Number(vm.vars[v].clone().number()),
            Type::NumberList => {
                let l = vm.vars[v].clone().list();
                let list = l.borrow().clone();
                Value::NumberList(list)
            }
            Type::Point => Value::Point(
                vm.vars[v].clone().number(),
                vm.vars[v + 1.into()].clone().number()
            ),
            Type::PointList | Type::Polygon => {
                let a = vm.vars[v].clone().list();
                let list = a
                    .borrow()
                    .chunks(2)
                    .map(|p| (p[0], p[1]))
                    .collect::<Vec<_>>();
                (if *ty == Type::PointList {
                    Value::PointList
                } else {
                    Value::Polygon
                })(list)
            }
            Type::PolygonList => {
                let a = vm.vars[v].clone().polygon_list();
                let list = a
                    .borrow()
                    .iter()
                    .map(|a| {
                        a.borrow()
                            .chunks(2)
                            .map(|p| (p[0], p[1]))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                Value::PolygonList(list)
            }
            Type::Bool | Type::BoolList => unreachable!(),
            Type::EmptyList => Value::EmptyList,
        },
        value
    );
}

fn assert_type_error(source: &str, error: TypeError) {
    println!("expression: {source}");
    let tree = parse_latex(source).unwrap();
    let statement = parse_statement(&tree).unwrap();
    let analysis = analyze_expression_list(
        [ExpressionListEntry {
            expression: &statement,
            parametric_domain: Domain::ZERO_TO_ONE,
        }]
        .as_slice()
        .as_ref(),
        false,
    );
    assert_eq!(
        analysis.results.first(),
        Some(&ExpressionResult::Err(AnalysisError::TypeError(error)))
    );
}

const NAN: f64 = f64::NAN;

#[rstest]
#[case(r"1 + [2,3]", [3, 4])]
#[case(r"[10,20,30] + [2,3]", [12, 23])]
#[case(r"[5,6,7][-4,0,0.9,1,1.9,2,3.9,4,1/0,0/0]", &[NAN, NAN, NAN, 5.0, 5.0, 6.0, 7.0, NAN, NAN, NAN])]
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
#[case(r"\min([1])", 1)]
#[case(r"\polygon()", polygon([]))]
#[case(r"\total([], [])", Value::EmptyList)]
#[case(r"\polygon(4,[5,6,7])", polygon([(4, 5), (4, 6), (4, 7)]))]
#[case(r"1\times(2,3)", (2, 3))]
#[case(r"(4,5)\times6", (24, 30))]
#[case(r"\total([])", -0.0)]
#[case(r"[][7]", f64::NAN)]
#[case(r"\polygon([])", polygon([]))]
#[case(r"\polygon([],[],[])", Value::PolygonList(vec![]))]
#[case(r"\polygon((8,9),[])", Value::PolygonList(vec![]))]
#[case(r"[\polygon()][1]", polygon([]))]
#[case(r"\unique([])+(0,0)", [(0, 0); 0])]
#[case(r"[][1,2,3]", &[f64::NAN, f64::NAN, f64::NAN])]
#[case(r"\argmin([])", 0)]
#[case(r"\argmin([3])", 1)]
#[case(r"\argmin([3,3])", 1)]
#[case(r"\argmin([4,9,2,5,2,9,1/0])", 3)]
#[case(r"\argmax([4,9,2,5,2,9,1/0])", 7)]
#[case(r"\argmax([4,9,2,5,0/0,9,1/0])", 0)]
#[case(r"\sortPerm([])", [0; 0])]
#[case(r"\sortPerm([4,9,2,3])", [2,3,0,1])]
fn expression_eq(#[case] expression: &str, #[case] expected: impl Into<Value>) {
    assert_expression_eq(expression, expected.into());
}

#[rstest]
#[case(
    r"\{1<2:(3,4),5\}",
    TypeError::PiecewiseBranchMismatch(Type::Point, Type::Number)
)]
#[case(
    r"\sort([])+(0,0)",
    TypeError::OpError(OpError::NoOverload(OpName::Add, vec![Type::NumberList, Type::Point]))
)]
fn expression_type_error(#[case] expression: &str, #[case] error: TypeError) {
    assert_type_error(expression, error);
}
