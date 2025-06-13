use std::hint::black_box;

use ambavia::{
    ast_parser::parse_expression_list_entry, compiler::compile_assignments,
    latex_parser::parse_latex, name_resolver::resolve_names, vm::Vm,
};

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
    let graph = black_box(
        r"
\count(5,5)
    "
        .trim(),
    );
    // let graph = r"
    //     (x,y) \for x=[1...5],y=[1...2]
    // ";

    let mut timer = Timer::default();
    timer.start("entire");
    timer.start("parse");
    let mut output = vec!["".to_string(); graph.lines().count()];
    let mut list_indices = vec![];
    let mut list = vec![];

    for (i, line) in graph.lines().enumerate() {
        timer.start(format!("line {i}"));
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
    println!("{}", assignments.len());
    timer.stop();

    for (i, a) in assignment_indices.iter().enumerate() {
        match a {
            Some(Ok(_)) => (),
            Some(Err(e)) => output[list_indices[i]] = format!("Reference Error: {e}"),
            _ => (),
        }
    }

    timer.start("compile_assignments");
    let compiled = compile_assignments(&assignments, &assignment_indices);
    timer.stop();

    match compiled {
        Err(e) => output.push(format!("Compile Error: {e}")),
        Ok((program, vars)) => {
            timer.start("create vm");
            let mut vm = Vm::with_program(program);
            timer.stop_start("run vm");
            vm.run(false);
            vm = black_box(vm);
            timer.stop_start("write output");

            for (i, v) in vars.iter().enumerate() {
                let Some(Ok((ty, index))) = v else {
                    continue;
                };

                use ambavia::instruction_builder::Type;

                output[list_indices[i]] = match ty {
                    Type::Number | Type::NumberList => format!("{}", vm.vars[*index]),
                    Type::Point => format!("({},{})", vm.vars[*index], vm.vars[*index + 1]),
                    Type::PointList => {
                        let a = vm.vars[*index].clone().list();
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
        }
    }
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
