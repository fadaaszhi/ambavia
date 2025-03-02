use std::io::{self, Write};

use ambavia::{
    ast_parser::parse_nodes_into_expression,
    compiler::{compile_expression, Names},
    instruction_builder::InstructionBuilder,
    latex_parser::parse_latex,
    latex_tree_flattener::Token,
    vm::Vm,
};

fn main() {
    let mut first = true;

    loop {
        if !first {
            // println!();
        }
        first = false;

        print!(">>> ");
        io::stdout().flush().unwrap();
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();

        let input = input.trim();

        let tree = match parse_latex(input) {
            Ok(tree) => tree,
            Err(e) => {
                println!("LaTeX error: {e:?}");
                continue;
            }
        };

        let ast = match parse_nodes_into_expression(&tree, Token::EndOfInput) {
            Ok(ast) => ast,
            Err(e) => {
                println!("AST error: {e}");
                continue;
            }
        };

        let mut builder = InstructionBuilder::default();
        let mut names = Names::default();
        if let Err(e) = compile_expression(&ast, &mut builder, &mut names) {
            println!("Compile error: {e}");
            continue;
        }
        let instructions = builder.finish();

        // println!("Compiled instructions:");
        // let width = (instructions.len() - 1).to_string().len();
        // for (i, instruction) in instructions.iter().enumerate() {
        //     println!("  {i:width$} {instruction:?}");
        // }

        let mut vm = Vm::with_program(instructions);
        vm.run(false);

        // println!("\nStack after running:");
        // let width = (vm.stack.len() - 1).to_string().len();
        // for (i, value) in vm.stack.iter().enumerate() {
        //     println!("  [{i:width$}] = {value}");
        // }
        println!("{}", vm.stack[0]);
    }
}
