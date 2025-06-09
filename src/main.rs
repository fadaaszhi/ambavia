use std::{
    fmt::format,
    io::{self, Write},
};

use ambavia::{
    ast_parser::parse_expression_list_entry,
    compiler::{compile_assignments, compile_expression, Names},
    instruction_builder::InstructionBuilder,
    latex_parser::parse_latex,
    latex_tree_flattener::Token,
    resolver::resolve_names,
    vm::Vm,
};

// fn main() {
//     let mut first = true;

//     loop {
//         if !first {
//             // println!();
//         }
//         first = false;

//         print!(">>> ");
//         io::stdout().flush().unwrap();
//         let mut input = String::new();
//         io::stdin().read_line(&mut input).unwrap();

//         let input = input.trim();

//         let tree = match parse_latex(input) {
//             Ok(tree) => tree,
//             Err(e) => {
//                 println!("LaTeX error: {e:?}");
//                 continue;
//             }
//         };

//         let ast = match parse_nodes_into_expression(&tree, Token::EndOfInput) {
//             Ok(ast) => ast,
//             Err(e) => {
//                 println!("AST error: {e}");
//                 continue;
//             }
//         };

//         let mut builder = InstructionBuilder::default();
//         let mut names = Names::default();
//         if let Err(e) = compile_expression(&ast, &mut builder, &mut names) {
//             println!("Compile error: {e}");
//             continue;
//         }
//         let instructions = builder.finish();

//         // println!("Compiled instructions:");
//         // let width = (instructions.len() - 1).to_string().len();
//         // for (i, instruction) in instructions.iter().enumerate() {
//         //     println!("  {i:width$} {instruction:?}");
//         // }

//         let mut vm = Vm::with_program(instructions);
//         vm.run(false);

//         // println!("\nStack after running:");
//         // let width = (vm.stack.len() - 1).to_string().len();
//         // for (i, value) in vm.stack.iter().enumerate() {
//         //     println!("  [{i:width$}] = {value}");
//         // }
//         println!("{}", vm.stack[0]);
//     }
// }

fn main() {
    let graph = r"
        c
        a = c \with b = 3
        b = 5a
        c = 2b
        c
    "
    .trim();

    let mut output = vec!["".to_string(); graph.lines().count()];
    let mut list_indices = vec![];
    let mut list = vec![];

    for (i, line) in graph.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }

        let tree = match parse_latex(line) {
            Ok(tree) => tree,
            Err(e) => {
                output[i] = format!("LaTeX error: {e:?}");
                continue;
            }
        };

        let ast = match parse_expression_list_entry(&tree) {
            Ok(ast) => ast,
            Err(e) => {
                output[i] = format!("AST Error: {e}");
                continue;
            }
        };

        list_indices.push(i);
        list.push(ast);
    }

    let (assignments, assignment_indices) = resolve_names(&list);

    for (i, a) in assignment_indices.iter().enumerate() {
        match a {
            Some(Ok(_)) => (),
            Some(Err(e)) => output[list_indices[i]] = format!("Reference Error: {e}"),
            _ => (),
        }
    }

    match compile_assignments(&assignments, &assignment_indices) {
        Err(e) => output.push(format!("Compile Error: {e}")),
        Ok((program, vars)) => {
            let mut vm = Vm::with_program(program);
            vm.run(false);

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
                    Type::EmptyList => todo!(),
                };
            }
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

    if output.len() != lines.len() {
        println!("{}", output.last().unwrap());
    }

    println!();
}
