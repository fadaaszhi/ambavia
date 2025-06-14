use std::hint::black_box;

use ambavia::{
    ast_parser::parse_expression_list_entry,
    compiler::compile_assignments,
    latex_parser::parse_latex,
    name_resolver::{resolve_names, ExpressionIndex},
    type_checker::{type_check, Type},
    vm::Vm,
};
use derive_more::{From, Into};
use typed_index_collections::{ti_vec, TiVec};

use crate::timer::Timer;

pub mod timer;

fn main() {
    let graph = r"
        a = 1
        b = a + (2,3)
        b \with a = (4,5)
    "
    .trim();

    let mut timer = Timer::default();
    let mut t_entire = timer.start("entire");
    let mut t_parse = t_entire.start("parse");
    t_parse.disable_sections();

    #[derive(From, Into, Clone, Copy)]
    struct OutputIndex(usize);

    let mut output: TiVec<OutputIndex, _> = ti_vec!["".to_string(); graph.lines().count()];
    let mut ei_to_oi: TiVec<ExpressionIndex, OutputIndex> = ti_vec![];
    let mut list: TiVec<ExpressionIndex, _> = ti_vec![];

    for (i, line) in graph.lines().enumerate() {
        let mut t_line_i = t_parse.start(format!("line {i}"));
        let i = OutputIndex(i);
        if line.trim().is_empty() {
            continue;
        }

        let t_parse_latex = t_line_i.start("parse_latex");
        let tree = match parse_latex(line) {
            Ok(tree) => tree,
            Err(e) => {
                output[i] = format!("LaTeX error: {e:?}");
                continue;
            }
        };
        drop(t_parse_latex);
        let t_parse_expression_list_entry = t_line_i.start("parse_expression_list_entry");
        let ast = match parse_expression_list_entry(&tree) {
            Ok(ast) => ast,
            Err(e) => {
                output[i] = format!("AST Error: {e}");
                continue;
            }
        };
        drop(t_parse_expression_list_entry);
        ei_to_oi.push(i);
        list.push(ast);
    }

    drop(t_parse);
    let t_resolve_names = t_entire.start("resolve_names");
    let (assignments, ei_to_nr) = resolve_names(&list);
    drop(t_resolve_names);
    let t_type_check = t_entire.start("type_check");
    let (assignments, nr_to_tc) = type_check(&assignments);
    drop(t_type_check);
    let t_compile_assignments = t_entire.start("compile_assignments");
    let (program, vars) = compile_assignments(&assignments);
    drop(t_compile_assignments);
    let t_create_vm = t_entire.start("create vm");
    let mut vm = Vm::with_program(program);
    drop(t_create_vm);
    let t_run_vm = t_entire.start("run vm");
    vm.run(false);
    vm = black_box(vm);
    drop(t_run_vm);
    let t_write_output = t_entire.start("write output");

    for (ei, nr) in ei_to_nr.into_iter_enumerated() {
        output[ei_to_oi[ei]] = match nr {
            Some(Ok(nr)) => match nr_to_tc[nr].clone() {
                Ok(tc) => {
                    let ty = assignments[tc].value.ty;
                    let v = vars[tc];
                    format!(
                        "({ty}) {}",
                        match ty {
                            Type::Number | Type::NumberList => format!("{}", vm.vars[v]),
                            Type::Point => format!("({},{})", vm.vars[v], vm.vars[v + 1]),
                            Type::PointList => {
                                let a = vm.vars[v].clone().list();
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
                        }
                    )
                }
                Err(e) => format!("Type Error: {e}"),
            },
            Some(Err(e)) => format!("Name Error: {e}"),
            None => continue,
        }
    }

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

    drop(t_write_output);
    drop(t_entire);
    println!("{}", timer.string());
    // cli_clipboard::set_contents(timer.string()).unwrap();

    println!();
}
