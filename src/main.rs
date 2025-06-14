use std::{collections::HashMap, hint::black_box};

mod bruh {
    use ambavia::{
        ast::ExpressionListEntry, compiler as cp, name_resolver as nr, type_checker as tc,
        vm::Instruction,
    };
    use derive_more::{From, Into};
    use typed_index_collections::{TiSlice, TiVec};

    #[derive(From, Into, Clone, Copy, Debug)]
    pub struct ExpressionListIndex(usize);
    #[derive(From, Into, Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct AssignmentIndex(usize);

    pub fn resolve_names(
        list: &TiSlice<ExpressionListIndex, ExpressionListEntry>,
    ) -> (
        TiVec<AssignmentIndex, nr::Assignment>,
        TiVec<ExpressionListIndex, Option<Result<AssignmentIndex, String>>>,
    ) {
        let (a, b) = nr::resolve_names(list.as_ref());
        (
            a.into(),
            b.into_iter()
                .map(|x| x.map(|x| x.map(|x| x.into())))
                .collect(),
        )
    }

    pub fn type_check(
        assignments: &TiSlice<AssignmentIndex, nr::Assignment>,
    ) -> TiVec<AssignmentIndex, Result<tc::Assignment, String>> {
        tc::type_check(assignments.as_ref()).into()
    }

    #[derive(From, Into, Clone, Copy, Debug)]
    pub struct FilteredAssignmentIndex(usize);

    #[derive(From, Into, Clone, Copy, Debug)]
    pub struct VmVarIndex(usize);

    pub fn compile_assignments(
        assignments: &TiSlice<FilteredAssignmentIndex, tc::Assignment>,
    ) -> (Vec<Instruction>, TiVec<FilteredAssignmentIndex, VmVarIndex>) {
        let (a, b) = cp::compile_assignments(assignments.as_ref());
        (a, b.into_iter().map(|x| x.into()).collect())
    }

    pub use ambavia::{ast_parser::parse_expression_list_entry, latex_parser::parse_latex, vm::Vm};
}

use ambavia::type_checker::Type;
use bruh::*;
use derive_more::{From, Into};
use typed_index_collections::{ti_vec, TiVec};

use crate::timer::Timer;

pub mod timer;

fn main() {
    // let graph = r"
    //     i \for i=(4,5)+[]
    // "
    // .trim();
    // let graph = r"
    //    ( i \for i = [])+(3,4)
    // "
    // .trim();
    // let graph = black_box(
    //     r"
    //     [][[]]
    // "
    //     .trim(),
    // );
    let graph = r"
        \{1=2:[(3,4)],[]\}[1]
    ";

    let mut timer = Timer::default();
    timer.start("entire");
    timer.start("parse");

    #[derive(From, Into, Clone, Copy)]
    struct OutputIndex(usize);

    let mut output: TiVec<OutputIndex, _> = ti_vec!["".to_string(); graph.lines().count()];
    let mut list_indices: TiVec<ExpressionListIndex, OutputIndex> = ti_vec![];
    let mut list: TiVec<ExpressionListIndex, _> = ti_vec![];

    for (i, line) in graph.lines().enumerate() {
        timer.start(format!("line {i}"));
        let i = OutputIndex(i);
        if line.trim().is_empty() {
            timer.stop();
            continue;
        }

        timer.start("parse_latex");
        let tree = match parse_latex(line) {
            Ok(tree) => tree,
            Err(e) => {
                output[i] = format!("LaTeX error: {e:?}");
                timer.stop();
                timer.stop();
                continue;
            }
        };

        timer.stop_start("parse_expression_list_entry");
        let ast = match parse_expression_list_entry(&tree) {
            Ok(ast) => ast,
            Err(e) => {
                output[i] = format!("AST Error: {e}");
                timer.stop();
                timer.stop();
                continue;
            }
        };
        timer.stop();
        timer.stop();
        list_indices.push(i);
        list.push(ast);
    }

    timer.stop_start("resolve_names");
    let (assignments, assignment_indices) = resolve_names(&list);
    // println!("{}", assignments.len());
    timer.stop_start("type_check");
    let assignments = type_check(&assignments);
    // dbg!(&assignments);
    timer.stop();

    let mut foo = HashMap::<AssignmentIndex, ExpressionListIndex>::new();

    for (i, a) in assignment_indices.iter_enumerated() {
        match a {
            Some(Ok(j)) => match &assignments[*j] {
                Ok(_) => {
                    foo.insert(*j, i);
                }
                Err(e) => output[list_indices[i]] = format!("Type Error: {e}"),
            },
            Some(Err(e)) => output[list_indices[i]] = format!("Reference Error: {e}"),
            _ => (),
        }
    }

    let asdf: TiVec<FilteredAssignmentIndex, _> = assignments
        .iter_enumerated()
        .filter_map(|(i, a)| a.as_ref().ok().and(Some(i)))
        .collect();

    timer.start("compile_assignments");
    let filtered_assignments = assignments
        .into_iter()
        .filter_map(|a| a.ok())
        .collect::<TiVec<_, _>>();
    let (program, vars) = compile_assignments(filtered_assignments.as_slice());
    timer.stop_start("create vm");
    let mut vm = Vm::with_program(program);
    timer.stop_start("run vm");
    vm.run(false);
    vm = black_box(vm);
    let vm_vars: TiVec<VmVarIndex, _> = vm.vars.into();
    timer.stop_start("write output");

    // dbg!(&vars);

    // println!("{:?}", Vec::from(asdf.clone()));

    // println!(
    //     "{}",
    //     vm_vars
    //         .iter()
    //         .enumerate()
    //         .map(|x| format!("{x:?}"))
    //         .collect::<Vec<_>>()
    //         .join("\n")
    // );

    for (i, v) in vars.iter_enumerated() {
        output[list_indices[foo[&asdf[i]]]] = match filtered_assignments[i].value.ty {
            Type::Number | Type::NumberList => format!("{}", vm_vars[*v]),
            Type::Point => format!(
                "({},{})",
                vm_vars[*v],
                vm_vars[VmVarIndex::from(usize::from(*v) + 1)]
            ),
            Type::PointList => {
                let a = vm_vars[*v].clone().list();
                format!(
                    "[{}]",
                    a.borrow()
                        .chunks(2)
                        .map(|p| format!("({},{})", p[0], p[1]))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            Type::Bool | Type::BoolList => unreachable!(),
            Type::EmptyList => "[]".into(),
        };
    }

    timer.stop();

    timer.stop();

    let lines = graph.lines().map(|l| l.trim()).collect::<Vec<_>>();
    let len = lines.iter().map(|l| l.len()).max().unwrap_or(0);

    println!(
        "\n{}",
        lines
            .iter()
            .zip(output.iter())
            .map(|(i, o)| format!("{i:len$}   │ {o}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    if output.len() != lines.len() {
        println!("{}", output.last().unwrap());
    }

    // println!("{}", timer.string());
    // cli_clipboard::set_contents(timer.string()).unwrap();

    println!();
}
