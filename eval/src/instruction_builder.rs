use std::{collections::HashMap, iter::zip};

use crate::vm::Instruction::{self, *};

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BaseType {
    Number,
    Point,
    Bool,
}

impl BaseType {
    fn size(&self) -> usize {
        match self {
            BaseType::Number => 1,
            BaseType::Point => 2,
            BaseType::Bool => 1,
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
enum Type {
    Number,
    NumberList,
    Point,
    PointList,
    Bool,
    BoolList,
}

impl Type {
    fn size(&self) -> usize {
        match self {
            Type::Number | Type::NumberList | Type::PointList | Type::Bool | Type::BoolList => 1,
            Type::Point => 2,
        }
    }

    fn base(&self) -> BaseType {
        match self {
            Type::Number | Type::NumberList => BaseType::Number,
            Type::Point | Type::PointList => BaseType::Point,
            Type::Bool | Type::BoolList => BaseType::Bool,
        }
    }

    fn list_of(base: BaseType) -> Type {
        match base {
            BaseType::Number => Type::NumberList,
            BaseType::Point => Type::PointList,
            BaseType::Bool => Type::BoolList,
        }
    }

    fn single(base: BaseType) -> Type {
        match base {
            BaseType::Number => Type::Number,
            BaseType::Point => Type::Point,
            BaseType::Bool => Type::Bool,
        }
    }

    fn is_list(&self) -> bool {
        matches!(self, Type::NumberList | Type::PointList | Type::BoolList)
    }
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self {
            Type::Number => "a number",
            Type::NumberList => "a list of numbers",
            Type::Point => "a point",
            Type::PointList => "a list of points",
            Type::Bool => "a true/false value",
            Type::BoolList => "a list of true/false values",
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct Value {
    name: usize,
    index: usize,
    ty: Type,
}

impl Value {
    fn clone_private(&self) -> Value {
        Value {
            name: self.name,
            index: self.index,
            ty: self.ty,
        }
    }
}

pub struct JumpHandle {
    index: usize,
    stack: Vec<Value>,
    ctx: Vec<Value>,
}

pub struct Label {
    index: usize,
    stack: Vec<Value>,
}

#[derive(Default)]
pub struct InstructionBuilder {
    instructions: Vec<Instruction>,
    value_counter: usize,
    stack: Vec<Value>,
    var_counter: usize,
    vars: HashMap<usize, (Option<Type>, usize)>,
}

impl InstructionBuilder {
    fn create_and_push_value(&mut self, ty: Type) -> Value {
        let name = self.value_counter;
        self.value_counter += 1;
        let index = self.stack.len();
        self.stack.push(Value { name, index, ty });
        Value { name, index, ty }
    }

    fn assert_exists(&self, v: &Value, ty: Type) {
        assert_eq!(v.ty, ty);
        assert_eq!(self.stack.get(v.index), Some(v));
    }

    fn assert_pop(&mut self, v: Value, ty: Type) {
        assert_eq!(v.ty, ty);
        assert_eq!(self.stack.pop(), Some(v));
    }

    fn position_from_top(&self, v: &Value) -> usize {
        self.assert_exists(v, v.ty);
        self.stack[v.index..].iter().map(|a| a.ty.size()).sum()
    }

    fn clone_stack(&self) -> Vec<Value> {
        self.stack.iter().map(|v| v.clone_private()).collect()
    }

    pub fn load_const(&mut self, x: f64) -> Value {
        self.instructions.push(LoadConst(x));
        self.create_and_push_value(Type::Number)
    }

    pub fn instr1(&mut self, instr: Instruction, a: Value) -> Value {
        let (a_type, return_type) = match instr {
            Neg | Abs | Floor | Ceil | Log2 | Sqrt | Sin | Cos | Tan | Asin | Acos | Atan => {
                (Type::Number, Type::Number)
            }
            Neg2 => (Type::Point, Type::Point),
            PointX | PointY | Hypot => (Type::Point, Type::Number),
            Count => (Type::NumberList, Type::Number),
            Count2 => (Type::PointList, Type::Number),
            Total => (Type::NumberList, Type::Number),
            Total2 => (Type::PointList, Type::Point),
            _ => panic!("instruction '{instr:?}' not unary"),
        };
        self.assert_pop(a, a_type);

        if instr != Point {
            self.instructions.push(instr);
        }

        self.create_and_push_value(return_type)
    }

    pub fn instr2(&mut self, instr: Instruction, a: Value, b: Value) -> Value {
        let (a_type, b_type, return_type) = match instr {
            Add | Sub | Mul | Div | Pow | Mod | Atan2 | Min => {
                (Type::Number, Type::Number, Type::Number)
            }
            Equal | LessThan | LessThanEqual | GreaterThan | GreaterThanEqual => {
                (Type::Number, Type::Number, Type::Bool)
            }
            Add2 | Sub2 => (Type::Point, Type::Point, Type::Point),
            Mul1_2 => (Type::Number, Type::Point, Type::Point),
            Mul2_1 | Div2_1 => (Type::Point, Type::Number, Type::Point),
            Dot | Distance => (Type::Point, Type::Point, Type::Number),
            Point => (Type::Number, Type::Number, Type::Point),
            Index => (Type::NumberList, Type::Number, Type::Number),
            Index2 => (Type::PointList, Type::Number, Type::Point),
            BuildListFromRange => (Type::Number, Type::Number, Type::NumberList),
            _ => panic!("instruction '{instr:?}' not binary"),
        };
        self.assert_pop(b, b_type);
        self.assert_pop(a, a_type);

        if instr != Point {
            self.instructions.push(instr);
        }

        self.create_and_push_value(return_type)
    }

    pub fn instr2_in_place(&mut self, instr: Instruction, a: &Value, b: Value) {
        let c = self.instr2(instr, a.clone_private(), b);
        assert_eq!(c.ty, a.ty);
        self.stack[c.index].name = a.name;
    }

    pub fn unchecked_index(&mut self, list: &Value, index: Value) -> Value {
        let base = list.ty.base();
        self.assert_exists(list, list.ty);
        assert!(list.ty.is_list());
        self.assert_pop(index, Type::Number);
        self.instructions.push(match base.size() {
            1 => UncheckedIndex,
            2 => UncheckedIndex2,
            _ => unreachable!(),
        }(self.position_from_top(list)));
        self.create_and_push_value(Type::single(base))
    }

    pub fn build_list(&mut self, base: BaseType, list: Vec<Value>) -> Value {
        let n = list.len();
        for v in list.into_iter().rev() {
            self.assert_pop(v, Type::single(base));
        }
        self.instructions.push(BuildList(n * base.size()));
        self.create_and_push_value(Type::list_of(base))
    }

    pub fn append(&mut self, list: &Value, v: Value) {
        let base = list.ty.base();
        self.assert_exists(list, list.ty);
        assert!(list.ty.is_list());
        self.assert_pop(v, Type::single(base));
        self.instructions.push(match base.size() {
            1 => Append,
            2 => Append2,
            _ => unreachable!(),
        }(self.position_from_top(list)));
    }

    pub fn store(&mut self, name: usize, v: Value) {
        let ty = v.ty;
        self.assert_pop(v, ty);
        let index = if let Some((var_ty, index)) = self.vars.get_mut(&name) {
            *var_ty = Some(ty);
            *index
        } else {
            let index = self.var_counter;
            self.var_counter += 2;
            self.vars.insert(name, (Some(ty), index));
            index
        };
        self.instructions.push(match ty.size() {
            1 => Store,
            2 => Store2,
            _ => unreachable!(),
        }(index));
    }

    pub fn load(&mut self, name: usize) -> Value {
        let (ty, index) = self
            .vars
            .get(&name)
            .unwrap_or_else(|| panic!("{name} was never stored to"));
        let ty = ty.unwrap_or_else(|| panic!("{name} was stored to but later undefined"));
        self.instructions.push(match ty.size() {
            1 => Load,
            2 => Load2,
            _ => unreachable!(),
        }(*index));
        self.create_and_push_value(ty)
    }

    pub fn load_store(&mut self, name: usize, v: Value) -> Value {
        let store_ty = v.ty;
        self.assert_pop(v, store_ty);
        let (ty, index) = self
            .vars
            .get_mut(&name)
            .unwrap_or_else(|| panic!("{name} was never stored to"));
        let load_ty = ty.unwrap_or_else(|| panic!("{name} was stored to but later undefined"));
        *ty = Some(store_ty);
        self.instructions
            .push(match (load_ty.size(), store_ty.size()) {
                (1, 1) => LoadStore,
                (1, 2) => Load1Store2,
                (2, 1) => Load2Store1,
                (2, 2) => Load2Store2,
                _ => unreachable!(),
            }(*index));
        self.create_and_push_value(load_ty)
    }

    pub fn undefine(&mut self, name: usize) {
        let (ty, _) = self
            .vars
            .get_mut(&name)
            .unwrap_or_else(|| panic!("{name} was never stored to"));
        if ty.is_none() {
            panic!("{name} was stored to but later undefined");
        }
        *ty = None;
    }

    #[cfg(test)]
    fn defined_vars_and_types(&self) -> HashMap<usize, (Type, usize)> {
        self.vars
            .iter()
            .filter_map(|(k, (t, i))| t.map(|t| (*k, (t, *i))))
            .collect()
    }

    pub fn defined_vars(&self) -> HashMap<usize, usize> {
        self.vars
            .iter()
            .filter_map(|(k, (t, i))| t.map(|_| (*k, *i)))
            .collect()
    }

    pub fn jump_if_false(&mut self, test: Value) -> JumpHandle {
        self.assert_pop(test, Type::Bool);
        let handle = JumpHandle {
            index: self.instructions.len(),
            stack: self.clone_stack(),
            ctx: vec![],
        };
        self.instructions.push(JumpIfFalse(!0));
        handle
    }

    pub fn jump(&mut self, ctx: Vec<Value>) -> JumpHandle {
        for v in ctx.iter().rev() {
            self.assert_pop(v.clone_private(), v.ty);
        }
        let handle = JumpHandle {
            index: self.instructions.len(),
            stack: self.clone_stack(),
            ctx,
        };
        self.instructions.push(Jump(!0));
        handle
    }

    pub fn label(&self) -> Label {
        Label {
            index: self.instructions.len(),
            stack: self.clone_stack(),
        }
    }

    pub fn set_jump_label(&mut self, jump: JumpHandle, label: &Label) {
        assert_eq!(
            jump.stack.iter().map(|v| v.name).collect::<Vec<_>>(),
            label.stack[0..jump.stack.len()]
                .iter()
                .map(|v| v.name)
                .collect::<Vec<_>>()
        );
        assert_eq!(jump.ctx.len(), label.stack.len() - jump.stack.len());
        for (j, l) in zip(&jump.ctx, &label.stack[jump.stack.len()..]) {
            assert_eq!((j.index, j.ty), (l.index, l.ty));
        }
        match &mut self.instructions[jump.index] {
            Jump(i) | JumpIfFalse(i) => *i = label.index,
            other => panic!("expected jump handle to point to jump instruction, found {other:?}"),
        }
    }

    pub fn copy(&mut self, v: &Value) -> Value {
        self.assert_exists(v, v.ty);
        let position = self.position_from_top(v);
        for _ in 0..v.ty.size() {
            self.instructions.push(Copy(position));
        }
        self.create_and_push_value(v.ty)
    }

    pub fn pop(&mut self, v: Value) {
        let ty = v.ty;
        self.instructions.push(Pop(ty.size()));
        self.assert_pop(v, ty);
    }

    pub fn count_specific(&mut self, list: &Value) -> Value {
        self.assert_exists(list, list.ty);
        assert!(list.ty.is_list());
        self.instructions.push(match list.ty.base().size() {
            1 => CountSpecific,
            2 => CountSpecific2,
            _ => unreachable!(),
        }(self.position_from_top(list)));
        self.create_and_push_value(Type::Number)
    }

    pub fn swap(&mut self, a: &mut Value, b: &mut Value) {
        assert_eq!(a.ty.size(), 1);
        assert_eq!(b.ty.size(), 1);
        self.assert_exists(a, a.ty);
        self.assert_exists(b, b.ty);
        assert_eq!(a.index, self.stack.len() - 1);
        self.instructions.push(Swap(self.position_from_top(b)));
        std::mem::swap(&mut a.index, &mut b.index);
        self.stack[a.index] = a.clone_private();
        self.stack[b.index] = b.clone_private();
    }

    pub fn swap_pop(&mut self, s: &mut Value, p: Vec<Value>) {
        let p_size = p.iter().map(|v| v.ty.size()).sum();
        self.instructions.push(match s.ty.size() {
            1 => Swap,
            2 => Swap2,
            _ => unreachable!(),
        }(s.ty.size() + p_size));
        self.instructions.push(Pop(p_size));

        self.assert_pop(s.clone_private(), s.ty);
        for v in p.into_iter().rev() {
            let ty = v.ty;
            self.assert_pop(v, ty);
        }

        s.index = self.stack.len();
        self.stack.push(s.clone_private());
    }

    pub fn finish(self) -> Vec<Instruction> {
        for instruction in &self.instructions {
            match instruction {
                Jump(i) | JumpIfFalse(i) => {
                    assert!(*i <= self.instructions.len(), "unset jump label")
                }
                _ => {}
            }
        }
        self.instructions
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn load_const() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        assert_eq!(
            a,
            Value {
                name: 0,
                index: 0,
                ty: Type::Number
            }
        );
        assert_eq!(ib.finish(), [LoadConst(5.0)]);
    }

    #[test]
    fn add() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        assert_eq!(
            a,
            Value {
                name: 0,
                index: 0,
                ty: Type::Number
            }
        );
        let b = ib.load_const(7.0);
        assert_eq!(
            b,
            Value {
                name: 1,
                index: 1,
                ty: Type::Number
            }
        );
        let c = ib.instr2(Add, a, b);
        assert_eq!(
            c,
            Value {
                name: 2,
                index: 0,
                ty: Type::Number
            }
        );
        assert_eq!(ib.stack, [c]);
        assert_eq!(ib.finish(), [LoadConst(5.0), LoadConst(7.0), Add]);
    }

    fn assert_panics<F, R>(f: F)
    where
        F: FnOnce() -> R + std::panic::UnwindSafe,
        R: std::fmt::Debug,
    {
        let r = std::panic::catch_unwind(f);

        if let Ok(x) = r {
            panic!("assertion failed: function didn't panic, returned {x:?}");
        }
    }

    #[test]
    fn error_add_wrong_order() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        let b = ib.load_const(7.0);
        assert_panics(move || ib.instr2(Add, b, a));
    }

    #[test]
    fn error_add_wrong_value() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        let b = ib.load_const(7.0);
        let _ = ib.load_const(7.0);
        assert_panics(move || ib.instr2(Add, a, b));
    }

    #[test]
    fn point() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        let b = ib.load_const(7.0);
        let c = ib.instr2(Point, a, b);
        assert_eq!(ib.stack, [c]);
        assert_eq!(ib.finish(), [LoadConst(5.0), LoadConst(7.0)]);
    }

    #[test]
    fn error_point_wrong_type() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        let b = ib.load_const(7.0);
        let c = ib.instr2(Point, a, b);
        let d = ib.load_const(7.0);
        assert_panics(move || ib.instr2(Point, c, d));
    }

    #[test]
    fn error_add_point_number() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        let b = ib.load_const(7.0);
        let c = ib.instr2(Point, a, b);
        let d = ib.load_const(7.0);
        assert_panics(move || ib.instr2(Add, c, d));
    }

    #[test]
    fn add2() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.instr2(Point, a, b);
        let d = ib.load_const(3.0);
        let e = ib.load_const(4.0);
        let f = ib.instr2(Point, d, e);
        let g = ib.instr2(Add2, c, f);
        assert_eq!(ib.stack, [g]);
        assert_eq!(
            ib.finish(),
            [
                LoadConst(1.0),
                LoadConst(2.0),
                LoadConst(3.0),
                LoadConst(4.0),
                Add2,
            ]
        );
    }

    #[test]
    fn index() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.instr2(BuildListFromRange, a, b);
        assert_eq!(
            c,
            Value {
                name: 2,
                index: 0,
                ty: Type::NumberList
            }
        );
        let d = ib.load_const(3.0);
        let e = ib.instr2(Index, c, d);
        assert_eq!(
            e,
            Value {
                name: 4,
                index: 0,
                ty: Type::Number
            }
        );
        assert_eq!(ib.stack, [e]);
        assert_eq!(
            ib.finish(),
            [
                LoadConst(1.0),
                LoadConst(2.0),
                BuildListFromRange,
                LoadConst(3.0),
                Index,
            ]
        );
    }

    #[test]
    fn error_index() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.instr2(BuildListFromRange, a, b);
        let d = ib.load_const(3.0);
        let _ = ib.load_const(5.0);
        assert_panics(move || ib.instr2(Index, c, d));
    }

    #[test]
    fn unchecked_index() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.instr2(BuildListFromRange, a, b);
        let d = ib.load_const(3.0);
        let e = ib.load_const(4.0);
        let de = ib.instr2(Point, d, e);
        let f = ib.load_const(5.0);
        let g = ib.unchecked_index(&c, f);
        assert_eq!(
            g,
            Value {
                name: 7,
                index: 2,
                ty: Type::Number,
            }
        );
        assert_eq!(ib.stack, [c, de, g]);
        assert_eq!(
            ib.finish(),
            [
                LoadConst(1.0),
                LoadConst(2.0),
                BuildListFromRange,
                LoadConst(3.0),
                LoadConst(4.0),
                LoadConst(5.0),
                UncheckedIndex(3)
            ]
        );
    }

    #[test]
    fn error_unchecked_index() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let _c = ib.instr2(BuildListFromRange, a, b);
        let d = ib.load_const(3.0);
        let e = ib.load_const(4.0);
        let de = ib.instr2(Point, d, e);
        let f = ib.load_const(5.0);
        assert_panics(move || ib.unchecked_index(&de, f));
    }

    #[test]
    fn build_number_list() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let d = ib.build_list(BaseType::Number, vec![a, b, c]);
        assert_eq!(
            d,
            Value {
                name: 3,
                index: 0,
                ty: Type::NumberList
            }
        );
        assert_eq!(
            ib.finish(),
            [LoadConst(1.0), LoadConst(2.0), LoadConst(3.0), BuildList(3)]
        );
    }

    #[test]
    fn build_point_list() {
        let mut ib = InstructionBuilder::default();
        let _a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let d = ib.instr2(Point, b, c);
        let e = ib.load_const(4.0);
        let f = ib.load_const(5.0);
        let g = ib.instr2(Point, e, f);
        let h = ib.build_list(BaseType::Point, vec![d, g]);
        assert_eq!(
            h,
            Value {
                name: 7,
                index: 1,
                ty: Type::PointList
            }
        );
        assert_eq!(
            ib.finish(),
            [
                LoadConst(1.0),
                LoadConst(2.0),
                LoadConst(3.0),
                LoadConst(4.0),
                LoadConst(5.0),
                BuildList(4)
            ]
        );
    }

    #[test]
    fn error_build_list() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let _b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        assert_panics(move || ib.build_list(BaseType::Number, vec![a, c]));

        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let d = ib.instr2(Point, b, c);
        assert_panics(move || ib.build_list(BaseType::Number, vec![a, d]));
    }

    #[test]
    fn append() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.instr2(Point, a, b);
        let d = ib.build_list(BaseType::Number, vec![]);
        let e = ib.build_list(BaseType::Point, vec![]);
        let f = ib.load_const(3.0);
        let g = ib.load_const(4.0);
        let h = ib.instr2(Point, f, g);
        let i = ib.load_const(5.0);
        let j = ib.load_const(6.0);
        ib.append(&d, j);
        ib.append(&d, i);
        ib.append(&e, h);
        assert_eq!(ib.stack, [c, d, e]);
        assert_eq!(
            ib.finish(),
            [
                LoadConst(1.0),
                LoadConst(2.0),
                BuildList(0),
                BuildList(0),
                LoadConst(3.0),
                LoadConst(4.0),
                LoadConst(5.0),
                LoadConst(6.0),
                Append(5),
                Append(4),
                Append2(1),
            ]
        );
    }

    #[test]
    fn instr2_in_place() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        assert_eq!(
            a,
            Value {
                name: 0,
                index: 0,
                ty: Type::Number
            }
        );
        let b = ib.load_const(7.0);
        assert_eq!(
            b,
            Value {
                name: 1,
                index: 1,
                ty: Type::Number
            }
        );
        ib.instr2_in_place(Add, &a, b);
        assert_eq!(ib.stack, [a]);
        assert_eq!(ib.finish(), [LoadConst(5.0), LoadConst(7.0), Add]);
    }

    #[test]
    fn load_store() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        ib.store(0, a);
        assert_eq!(
            ib.defined_vars_and_types(),
            HashMap::from([(0, (Type::Number, 0))])
        );
        let b = ib.load_const(1.0);
        let c = ib.load_const(2.0);
        let d = ib.instr2(Point, b, c);
        ib.store(1, d);
        assert_eq!(
            ib.defined_vars_and_types(),
            HashMap::from([(0, (Type::Number, 0)), (1, (Type::Point, 2))])
        );
        assert_eq!(ib.stack, []);
        let a = ib.load(0);
        assert_eq!(
            a,
            Value {
                name: 4,
                index: 0,
                ty: Type::Number
            }
        );
        let d = ib.load(1);
        assert_eq!(
            d,
            Value {
                name: 5,
                index: 1,
                ty: Type::Point
            }
        );
        let old_a = ib.load_store(0, d);
        assert_eq!(
            ib.defined_vars_and_types(),
            HashMap::from([(0, (Type::Point, 0)), (1, (Type::Point, 2))])
        );
        ib.undefine(0);
        assert_eq!(
            ib.defined_vars_and_types(),
            HashMap::from([(1, (Type::Point, 2))])
        );
        assert_eq!(ib.stack, [a, old_a]);
        let l = ib.build_list(BaseType::Point, vec![]);
        ib.store(0, l);
        assert_eq!(
            ib.defined_vars_and_types(),
            HashMap::from([(0, (Type::PointList, 0)), (1, (Type::Point, 2))])
        );
        assert_eq!(
            ib.finish(),
            [
                LoadConst(5.0),
                Store(0),
                LoadConst(1.0),
                LoadConst(2.0),
                Store2(2),
                Load(0),
                Load2(2),
                Load1Store2(0),
                BuildList(0),
                Store(0),
            ]
        );
    }

    #[test]
    fn error_load_store() {
        let mut ib = InstructionBuilder::default();
        assert_panics(move || ib.load(0));

        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(5.0);
        ib.store(0, a);
        ib.undefine(0);
        assert_panics(move || ib.load(0));
    }

    #[test]
    fn if_else() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let b_less_than_c = ib.instr2(LessThan, b, c);
        let jif = ib.jump_if_false(b_less_than_c);
        let d = ib.load_const(5.0);
        let e = ib.load_const(6.0);
        let j = ib.jump(vec![d, e]);
        ib.set_jump_label(jif, &ib.label());
        let d = ib.load_const(7.0);
        let e = ib.load_const(8.0);
        ib.set_jump_label(j, &ib.label());
        assert_eq!(ib.stack, [a, d, e]);
        assert_eq!(
            ib.finish(),
            [
                LoadConst(1.0),
                LoadConst(2.0),
                LoadConst(3.0),
                LessThan,
                JumpIfFalse(8),
                LoadConst(5.0),
                LoadConst(6.0),
                Jump(10),
                LoadConst(7.0),
                LoadConst(8.0)
            ]
        );
    }

    #[test]
    fn error_if_else() {
        let mut ib = InstructionBuilder::default();
        let _a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let b_less_than_c = ib.instr2(LessThan, b, c);
        let jif = ib.jump_if_false(b_less_than_c);
        let d = ib.load_const(5.0);
        let e = ib.load_const(6.0);
        let f = ib.instr2(Point, d, e);
        let j = ib.jump(vec![f]);
        ib.set_jump_label(jif, &ib.label());
        let _d = ib.load_const(7.0);
        let _e = ib.load_const(8.0);
        assert_panics(move || ib.set_jump_label(j, &ib.label()));

        let mut ib = InstructionBuilder::default();
        let _a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let b_less_than_c = ib.instr2(LessThan, b, c);
        let jif = ib.jump_if_false(b_less_than_c);
        let _d = ib.load_const(5.0);
        let _e = ib.load_const(6.0);
        let _j = ib.jump(vec![]);
        assert_panics(move || ib.set_jump_label(jif, &ib.label()));

        let mut ib = InstructionBuilder::default();
        let _a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let b_less_than_c = ib.instr2(LessThan, b, c);
        let jif = ib.jump_if_false(b_less_than_c);
        let d = ib.load_const(5.0);
        let j = ib.jump(vec![d]);
        ib.set_jump_label(jif, &ib.label());
        let _d = ib.load_const(7.0);
        let _e = ib.load_const(8.0);
        assert_panics(move || ib.set_jump_label(j, &ib.label()));

        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let a_less_than_b = ib.instr2(LessThan, a, b);
        let jif = ib.jump_if_false(a_less_than_b);
        let c = ib.load_const(2.0);
        let _j = ib.jump(vec![c]);
        ib.set_jump_label(jif, &ib.label());
        let _c = ib.load_const(2.0);
        assert_panics(move || ib.finish());
    }

    #[test]
    fn copy() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let d = ib.instr2(Point, b, c);
        let a_copy = ib.copy(&a);
        let d_copy = ib.copy(&d);
        let d_copy2 = ib.copy(&d);
        assert_eq!(ib.stack, [a, d, a_copy, d_copy, d_copy2]);
        assert_eq!(
            ib.finish(),
            [
                LoadConst(1.0),
                LoadConst(2.0),
                LoadConst(3.0),
                Copy(3),
                Copy(3),
                Copy(3),
                Copy(5),
                Copy(5)
            ]
        );
    }

    #[test]
    fn pop() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let d = ib.instr2(Point, b, c);
        ib.pop(d);
        ib.pop(a);
        assert_eq!(ib.stack, []);
        assert_eq!(
            ib.finish(),
            [
                LoadConst(1.0),
                LoadConst(2.0),
                LoadConst(3.0),
                Pop(2),
                Pop(1)
            ]
        );
    }

    #[test]
    fn error_pop() {
        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(2.0);
        let c = ib.load_const(3.0);
        let _d = ib.instr2(Point, b, c);
        assert_panics(move || ib.pop(a));
    }

    #[test]
    fn loop_() {
        let mut ib = InstructionBuilder::default();
        let list = ib.build_list(BaseType::Number, vec![]);
        let n = ib.load_const(5.0);
        let i = ib.load_const(0.0);
        let start = ib.label();
        let i_copy = ib.copy(&i);
        let n_copy = ib.copy(&n);
        let i_lt_n = ib.instr2(LessThan, i_copy, n_copy);
        let jif = ib.jump_if_false(i_lt_n);
        let i_copy = ib.copy(&i);
        let two = ib.load_const(2.0);
        let i_squared = ib.instr2(Pow, i_copy, two);
        ib.append(&list, i_squared);
        let jump = ib.jump(vec![]);
        ib.set_jump_label(jump, &start);
        ib.set_jump_label(jif, &ib.label());
        ib.pop(i);
        ib.pop(n);
        assert_eq!(ib.stack, [list]);
        assert_eq!(
            ib.finish(),
            [
                BuildList(0),
                LoadConst(5.0),
                LoadConst(0.0),
                Copy(1),
                Copy(3),
                LessThan,
                JumpIfFalse(12),
                Copy(1),
                LoadConst(2.0),
                Pow,
                Append(3),
                Jump(3),
                Pop(1),
                Pop(1),
            ]
        );
    }

    #[test]
    fn error_loop() {
        let mut ib = InstructionBuilder::default();
        let list = ib.build_list(BaseType::Number, vec![]);
        let n = ib.load_const(5.0);
        let i = ib.load_const(0.0);
        let start = ib.label();
        let i_copy = ib.copy(&i);
        let n_copy = ib.copy(&n);
        let i_lt_n = ib.instr2(LessThan, i_copy, n_copy);
        let _jif = ib.jump_if_false(i_lt_n);
        let _oops = ib.load_const(9.0);
        let i_copy = ib.copy(&i);
        let two = ib.load_const(2.0);
        let i_squared = ib.instr2(Pow, i_copy, two);
        ib.append(&list, i_squared);
        let jump = ib.jump(vec![]);
        assert_panics(move || ib.set_jump_label(jump, &start));
    }

    #[test]
    fn swap() {
        let mut ib = InstructionBuilder::default();
        let mut a = ib.build_list(BaseType::Point, vec![]);
        let b = ib.load_const(1.0);
        let c = ib.load_const(2.0);
        let d = ib.instr2(Point, b, c);
        let mut e = ib.load_const(3.0);
        ib.swap(&mut e, &mut a);
        assert_eq!(
            e,
            Value {
                name: 4,
                index: 0,
                ty: Type::Number
            }
        );
        assert_eq!(
            a,
            Value {
                name: 0,
                index: 2,
                ty: Type::PointList
            }
        );
        assert_eq!(ib.stack, [e, d, a]);
        assert_eq!(
            ib.finish(),
            [
                BuildList(0),
                LoadConst(1.0),
                LoadConst(2.0),
                LoadConst(3.0),
                Swap(4)
            ]
        );
    }

    #[test]
    fn error_swap() {
        let mut ib = InstructionBuilder::default();
        let mut a = ib.build_list(BaseType::Point, vec![]);
        let b = ib.load_const(1.0);
        let c = ib.load_const(2.0);
        let mut d = ib.instr2(Point, b, c);
        let _e = ib.load_const(3.0);
        assert_panics(move || ib.swap(&mut d, &mut a));

        let mut ib = InstructionBuilder::default();
        let a = ib.load_const(1.0);
        let b = ib.load_const(1.0);
        let mut c = ib.instr2(Point, a, b);
        let d = ib.load_const(1.0);
        let e = ib.load_const(1.0);
        let mut f = ib.instr2(Point, d, e);
        assert_panics(move || ib.swap(&mut c, &mut f));
    }

    #[test]
    fn swap_pop() {
        let mut ib = InstructionBuilder::default();
        let a = ib.build_list(BaseType::Point, vec![]);
        let b = ib.load_const(1.0);
        let c = ib.load_const(2.0);
        let d = ib.instr2(Point, b, c);
        let e = ib.load_const(3.0);
        let f = ib.load_const(4.0);
        let g = ib.load_const(5.0);
        let h = ib.instr2(Point, f, g);
        let mut i = ib.build_list(BaseType::Number, vec![]);
        ib.swap_pop(&mut i, vec![d, e, h]);
        assert_eq!(
            i,
            Value {
                name: 8,
                index: 1,
                ty: Type::NumberList
            }
        );
        assert_eq!(ib.stack, [a, i]);
        assert_eq!(
            ib.finish(),
            [
                BuildList(0),
                LoadConst(1.0),
                LoadConst(2.0),
                LoadConst(3.0),
                LoadConst(4.0),
                LoadConst(5.0),
                BuildList(0),
                Swap(6),
                Pop(5)
            ]
        )
    }

    #[test]
    fn error_swap_pop() {
        let mut ib = InstructionBuilder::default();
        let _a = ib.build_list(BaseType::Point, vec![]);
        let b = ib.load_const(1.0);
        let c = ib.load_const(2.0);
        let d = ib.instr2(Point, b, c);
        let _e = ib.load_const(3.0);
        let f = ib.load_const(4.0);
        let g = ib.load_const(5.0);
        let h = ib.instr2(Point, f, g);
        let mut i = ib.build_list(BaseType::Number, vec![]);
        assert_panics(move || ib.swap_pop(&mut i, vec![d, h]));

        let mut ib = InstructionBuilder::default();
        let _a = ib.build_list(BaseType::Point, vec![]);
        let b = ib.load_const(1.0);
        let c = ib.load_const(2.0);
        let _d = ib.instr2(Point, b, c);
        let e = ib.load_const(3.0);
        let mut f = ib.load_const(4.0);
        let _g = ib.load_const(5.0);
        assert_panics(move || ib.swap_pop(&mut f, vec![e]));
    }
}
