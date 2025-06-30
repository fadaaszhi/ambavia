# Ambavia

Incomplete implementation of [Desmos Graphing Calculator](https://www.desmos.com/calculator).

Currently it can parse a list of expressions/assignments, do name resolution and type checking, compile everything into bytecode and then interpret the bytecode to spit out the result of each expression.

If you do `cargo run` then you'll be greeted with the beginnings of a UI that's not hooked up to the aforementioned evaluator yet.

## Credits

- Fonts are from [KaTeX](https://github.com/KaTeX/KaTeX).
- Font atlas generated with [msdf-atlas-gen](https://github.com/Chlumsky/msdf-atlas-gen).
