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
    1 + (2 + \{2=4: 5\})

    // 1
    // 1 2
    // 1 2 2
    // 1 2 
    "
    .trim();

    let mut timer = Timer::default();
    let mut t = timer.start("entire");
    let mut t_parse = t.start("parse");
    t_parse.disable_sections();

    #[derive(From, Into, Clone, Copy)]
    struct OutputIndex(usize);

    let mut output: TiVec<OutputIndex, _> = ti_vec!["".to_string(); graph.lines().count()];
    let mut ei_to_oi: TiVec<ExpressionIndex, OutputIndex> = ti_vec![];
    let mut list: TiVec<ExpressionIndex, _> = ti_vec![];

    for (i, line) in graph.lines().enumerate() {
        let mut t = t_parse.start(format!("line {i}"));
        let i = OutputIndex(i);
        if line.trim().is_empty() {
            continue;
        }

        let tree = match t.time("parse_latex", || parse_latex(line)) {
            Ok(tree) => tree,
            Err(e) => {
                output[i] = format!("LaTeX error: {e:?}");
                continue;
            }
        };
        let ast = match t.time("parse_expression_list_entry", || {
            parse_expression_list_entry(&tree)
        }) {
            Ok(ast) => ast,
            Err(e) => {
                output[i] = format!("AST Error: {e}");
                continue;
            }
        };
        ei_to_oi.push(i);
        list.push(ast);
    }

    drop(t_parse);

    let (assignments, ei_to_nr) = t.time("resolve_names", || resolve_names(&list));
    let (assignments, nr_to_tc) = t.time("type_check", || type_check(&assignments));
    let (program, vars) = t.time("compile_assignments", || compile_assignments(&assignments));
    let mut vm = t.time("create vm", || Vm::with_program(program));
    t.time("run vm", || vm.run(false));
    vm = black_box(vm);
    t.time("write output", || {
        for (ei, nr) in ei_to_nr.into_iter_enumerated() {
            output[ei_to_oi[ei]] = match nr {
                Some(Ok(nr)) => match nr_to_tc[nr].clone() {
                    Ok(tc) => {
                        let ty = assignments[tc].value.ty;
                        let v = vars[tc];
                        format!(
                            // "({ty}) {}",
                            "{}",
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
    });

    drop(t);
    println!("{}", timer.string());
    // cli_clipboard::set_contents(timer.string()).unwrap();

    println!();
}
