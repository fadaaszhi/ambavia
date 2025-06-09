goal: evaluate expressions list with assignments, function declarations and expressions (no relations)

input: expressions list

output:




```rs
type Index = usize;
type Map = HashMap<&str, Index>;

// current_parameters: function parameters from the current line. goes away when entering variable
// globals: stack of values, including with/for
// past_parameters: function parameters from other lines
fn stuff(expr, current_parameters: Map, globals: Vec<Map>, past_parameters: Vec<Map>) {
    match expr {
        
    }
}
        
```