use core::f64;
use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Write,
    hash::{DefaultHasher, Hash, Hasher},
    iter::zip,
    rc::Rc,
};

use derive_more::{Add, From, Into};
use ordered_float::OrderedFloat;
use strum::{Display, EnumCount, EnumDiscriminants, FromRepr};
use typed_index_collections::{TiSlice, TiVec};

use crate::math;

#[derive(Debug, Clone, Copy, PartialEq, EnumCount, EnumDiscriminants)]
#[strum_discriminants(derive(FromRepr, Display))]
pub enum Instruction {
    Start,
    Halt,
    Unreachable,

    LoadConst(f64),
    Load(VarIndex),
    Load2(VarIndex),
    Load3(VarIndex),
    Store(VarIndex),
    Store2(VarIndex),
    Store3(VarIndex),
    LoadStore(VarIndex),
    Load1Store2(VarIndex),
    Load2Store1(VarIndex),
    Load2Store2(VarIndex),
    Copy(usize),
    Swap(usize),
    Swap2(usize),
    Swap3(usize),
    Pop(usize),

    Neg,
    Neg2,
    Neg3,
    Add,
    Add2,
    Add3,
    Sub,
    Sub2,
    Sub3,
    Mul,
    Mul1_2,
    Mul1_3,
    Div,
    Div2_1,
    Div3_1,
    Pow,
    Dot2,
    Dot3,
    Cross,
    Point2,
    Point3,

    Equal,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,

    Point2X,
    Point2Y,
    Point3X,
    Point3Y,
    Point3Z,
    Hypot2,
    Hypot3,
    Sqrt,

    Ln,
    Exp,
    Erf,
    Sin,
    Cos,
    Tan,
    Sec,
    Csc,
    Cot,
    Sinh,
    Cosh,
    Tanh,
    Sech,
    Csch,
    Coth,
    Asin,
    Acos,
    Atan,
    Atan2,
    Asec,
    Acsc,
    Acot,
    Asinh,
    Acosh,
    Atanh,
    Asech,
    Acsch,
    Acoth,
    Abs,
    Sgn,
    Round,
    RoundWithPrecision,
    Floor,
    Ceil,
    Mod,
    Midpoint2,
    Midpoint3,
    Distance2,
    Distance3,
    Min,
    Max,
    Median,
    Argmin,
    Argmax,
    Total,
    Total2,
    Total3,
    Mean,
    Mean2,
    Mean3,
    Count,
    Count2,
    Count3,
    CountPolygonList,
    Repeat,
    Repeat2,
    Repeat3,
    RepeatPolygon,
    RepeatList,
    Repeat2List,
    Repeat3List,
    RepeatPolygonList,
    Unique,
    Unique2,
    Unique3,
    UniquePolygon,
    UniquePerm,
    UniquePerm2,
    UniquePerm3,
    UniquePermPolygon,
    Sort,
    SortKey,
    SortKey2,
    SortKey3,
    SortKeyPolygon,
    SortPerm,
    Polygon,
    Vertices,
    Push,
    Push2,
    Push3,
    PushPolygon,
    Concat,
    Concat2,
    Concat3,
    ConcatPolygon,

    And,
    MinInternal,
    Index,
    Index2,
    Index3,
    IndexPolygonList,
    UncheckedIndex(usize),
    UncheckedIndex2(usize),
    UncheckedIndex3(usize),
    UncheckedIndexPolygonList(usize),
    BuildList(usize),
    BuildPolygonList(usize),
    BuildListFromRange,
    Append(usize),
    Append2(usize),
    Append3(usize),
    AppendPolygonList(usize),
    CountSpecific(usize),
    CountSpecific2(usize),
    CountSpecific3(usize),
    CountSpecificPolygonList(usize),
    Slider,

    StartArgs,
    EndArgs(usize),
    Jump(usize),
    JumpIfFalse(usize),
    Return1,
    Return2,
}

type RcVec<T> = Rc<RefCell<Vec<T>>>;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    List(RcVec<f64>),
    PolygonList(RcVec<RcVec<f64>>),
}

impl Value {
    pub fn number(self) -> f64 {
        match self {
            Value::Number(v) => v,
            _ => panic!("value is not a number: {self:?}"),
        }
    }

    pub fn list(self) -> RcVec<f64> {
        match self {
            Value::List(v) => v,
            _ => panic!("value is not a list: {self:?}"),
        }
    }

    pub fn polygon_list(self) -> RcVec<RcVec<f64>> {
        match self {
            Value::PolygonList(v) => v,
            _ => panic!("value is not a polygon list: {self:?}"),
        }
    }
}

impl From<f64> for Value {
    fn from(value: f64) -> Self {
        Value::Number(value)
    }
}

impl From<Rc<RefCell<Vec<f64>>>> for Value {
    fn from(value: Rc<RefCell<Vec<f64>>>) -> Self {
        Value::List(value)
    }
}

impl From<Rc<RefCell<Vec<Rc<RefCell<Vec<f64>>>>>>> for Value {
    fn from(value: Rc<RefCell<Vec<Rc<RefCell<Vec<f64>>>>>>) -> Self {
        Value::PolygonList(value)
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Number(x) => write!(f, "{x}"),
            Value::List(list) => {
                write!(
                    f,
                    "[{}]",
                    list.borrow()
                        .iter()
                        .map(|x| format!("{x}"))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
            Value::PolygonList(list) => {
                write!(
                    f,
                    "[{}]",
                    list.borrow()
                        .iter()
                        .map(|list| format!(
                            "[{}]",
                            list.borrow()
                                .iter()
                                .map(|x| format!("{x}"))
                                .collect::<Vec<_>>()
                                .join(",")
                        ))
                        .collect::<Vec<_>>()
                        .join(",")
                )
            }
        }
    }
}

#[derive(Debug, Copy, Clone, From, Into, PartialEq, Add)]
pub struct VarIndex(pub usize);
pub type Vars = TiVec<VarIndex, Value>;

#[derive(Debug, Default, Clone)]
pub struct Vm<'a, 'i> {
    pub program: &'i [Instruction],
    pub pc: usize,
    pub stack: Vec<Value>,
    pub vars: Vars,
    pub names: Option<&'a TiSlice<VarIndex, String>>,
}

/// The bits of a quiet NaN with a randomly-generated payload
pub const UNINITIALIZED_BITS: u64 = 0x7ff90a1a42c77dd3;

impl<'a, 'i> Vm<'a, 'i> {
    pub fn new(
        program: &'i [Instruction],
        mut vars: Vars,
        builtin_constant_indices: impl IntoIterator<Item = VarIndex>,
    ) -> Vm<'a, 'i> {
        let n_vars = program
            .iter()
            .map(|i| match i {
                Instruction::Load(j) | Instruction::Store(j) => j.0 + 1,
                Instruction::Load2(j) | Instruction::Store2(j) => j.0 + 2,
                Instruction::Load3(j) | Instruction::Store3(j) => j.0 + 3,
                _ => 0,
            })
            .chain(builtin_constant_indices.into_iter().map(|i| i.0 + 3))
            .max()
            .unwrap_or(0);
        if vars.len() < n_vars {
            vars.resize(n_vars, Value::Number(f64::from_bits(UNINITIALIZED_BITS)));
        }
        Vm {
            program,
            vars,
            ..Default::default()
        }
    }

    pub fn set_names(&mut self, names: &'a TiSlice<VarIndex, String>) {
        self.names = Some(names);
    }

    fn push(&mut self, value: impl Into<Value>) {
        self.stack.push(value.into());
    }

    fn pop(&mut self) -> Value {
        self.stack.pop().unwrap()
    }

    fn peek(&mut self, index: usize) -> Value {
        self.stack[self.stack.len() - index].clone()
    }

    fn name(&self, index: VarIndex) -> Option<&str> {
        self.names.as_ref().map(|n| n[index].as_ref())
    }

    fn load(&self, index: VarIndex) -> Value {
        let value = self.vars[index].clone();

        if let Value::Number(x) = value
            && x.to_bits() == UNINITIALIZED_BITS
        {
            let mut msg: String = "".into();
            write!(msg, "variable is uninitialized: {index:?}").unwrap();

            if let Some(name) = self.name(index) {
                write!(msg, " ({name})").unwrap();
            }

            panic!("{}", msg);
        }

        value
    }

    pub fn run(&mut self, print_trace: bool) {
        if let Some(start) = self
            .program
            .iter()
            .position(|i| matches!(i, Instruction::Start))
        {
            self.pc = start + 1;
        } else {
            self.pc = 0;
        }

        const COUNT_INSTRUCTIONS: bool = false;
        let mut instruction_counts = [0; Instruction::COUNT];

        while self.pc < self.program.len() {
            let instruction = self.program[self.pc];

            if COUNT_INSTRUCTIONS {
                instruction_counts[InstructionDiscriminants::from(instruction) as usize] += 1;
            }

            if print_trace {
                print!("{} {:?}", self.pc, instruction);

                match instruction {
                    Instruction::Load(index)
                    | Instruction::Load2(index)
                    | Instruction::Store(index)
                    | Instruction::Store2(index) => {
                        if let Some(name) = self.name(index) {
                            print!(" ({name})");
                        }
                    }
                    _ => {}
                }

                println!();
            }

            self.pc += 1;

            match instruction {
                Instruction::Start => unreachable!(),
                Instruction::Halt => self.pc = self.program.len(),
                Instruction::Unreachable => unreachable!(),

                Instruction::LoadConst(value) => self.push(value),
                Instruction::Load(index) => {
                    let value = self.load(index);
                    self.push(value);
                }
                Instruction::Load2(index) => {
                    self.push(self.load(index));
                    self.push(self.load(index + 1.into()));
                }
                Instruction::Load3(index) => {
                    self.push(self.load(index));
                    self.push(self.load(index + 1.into()));
                    self.push(self.load(index + 2.into()));
                }
                Instruction::Store(index) => self.vars[index] = self.pop(),
                Instruction::Store2(index) => {
                    self.vars[index + 1.into()] = self.pop();
                    self.vars[index] = self.pop();
                }
                Instruction::Store3(index) => {
                    self.vars[index + 2.into()] = self.pop();
                    self.vars[index + 1.into()] = self.pop();
                    self.vars[index] = self.pop();
                }
                Instruction::LoadStore(index) => {
                    std::mem::swap(&mut self.vars[index], self.stack.last_mut().unwrap());
                }
                Instruction::Load1Store2(index) => {
                    self.vars[index + 1.into()] = self.pop();
                    std::mem::swap(&mut self.vars[index], self.stack.last_mut().unwrap());
                }
                Instruction::Load2Store1(index) => {
                    std::mem::swap(&mut self.vars[index], self.stack.last_mut().unwrap());
                    self.push(self.vars[index + 1.into()].clone());
                }
                Instruction::Load2Store2(index) => {
                    let len = self.stack.len();
                    std::mem::swap(&mut self.vars[index], &mut self.stack[len - 2]);
                    std::mem::swap(&mut self.vars[index + 1.into()], &mut self.stack[len - 1]);
                }
                Instruction::Copy(index) => {
                    self.push(self.stack[self.stack.len() - index].clone());
                }
                Instruction::Swap(index) => {
                    let a = self.stack.len();
                    self.stack.swap(a - 1, a - index);
                }
                Instruction::Swap2(index) => {
                    let a = self.stack.len();
                    self.stack.swap(a - 2, a - index);
                    self.stack.swap(a - 1, a - index + 1);
                }
                Instruction::Swap3(index) => {
                    let a = self.stack.len();
                    self.stack.swap(a - 3, a - index);
                    self.stack.swap(a - 2, a - index + 1);
                    self.stack.swap(a - 1, a - index + 2);
                }
                Instruction::Pop(count) => {
                    for _ in 0..count {
                        self.stack.pop();
                    }
                }

                Instruction::Neg => {
                    let a = self.pop().number();
                    self.push(-a);
                }
                Instruction::Neg2 => {
                    let y = self.pop().number();
                    let x = self.pop().number();
                    self.push(-x);
                    self.push(-y);
                }
                Instruction::Neg3 => {
                    let z = self.pop().number();
                    let y = self.pop().number();
                    let x = self.pop().number();
                    self.push(-x);
                    self.push(-y);
                    self.push(-z);
                }
                Instruction::Add => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a + b);
                }
                Instruction::Add2 => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax + bx);
                    self.push(ay + by);
                }
                Instruction::Add3 => {
                    let bz = self.pop().number();
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let az = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax + bx);
                    self.push(ay + by);
                    self.push(az + bz);
                }
                Instruction::Sub => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a - b);
                }
                Instruction::Sub2 => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax - bx);
                    self.push(ay - by);
                }
                Instruction::Sub3 => {
                    let bz = self.pop().number();
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let az = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax - bx);
                    self.push(ay - by);
                    self.push(az - bz);
                }
                Instruction::Mul => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a * b);
                }
                Instruction::Mul1_2 => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let a = self.pop().number();
                    self.push(a * bx);
                    self.push(a * by);
                }
                Instruction::Mul1_3 => {
                    let bz = self.pop().number();
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let a = self.pop().number();
                    self.push(a * bx);
                    self.push(a * by);
                    self.push(a * bz);
                }
                Instruction::Div => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a / b);
                }
                Instruction::Div2_1 => {
                    let b = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax / b);
                    self.push(ay / b);
                }
                Instruction::Div3_1 => {
                    let b = self.pop().number();
                    let az = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax / b);
                    self.push(ay / b);
                    self.push(az / b);
                }
                Instruction::Pow => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a.powf(b));
                }
                Instruction::Dot2 => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax * bx + ay * by);
                }
                Instruction::Dot3 => {
                    let bz = self.pop().number();
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let az = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax * bx + ay * by + az * bz);
                }
                Instruction::Cross => {
                    let bz = self.pop().number();
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let az = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ay * bz - az * by);
                    self.push(az * bx - ax * bz);
                    self.push(ax * by - ay * bx);
                }
                Instruction::Point2 => {
                    // noop
                }
                Instruction::Point3 => {
                    // noop
                }

                Instruction::Equal => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(if a == b { 1.0 } else { 0.0 });
                }
                Instruction::LessThan => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(if a < b { 1.0 } else { 0.0 });
                }
                Instruction::LessThanEqual => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(if a <= b { 1.0 } else { 0.0 });
                }
                Instruction::GreaterThan => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(if a > b { 1.0 } else { 0.0 });
                }
                Instruction::GreaterThanEqual => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(if a >= b { 1.0 } else { 0.0 });
                }

                Instruction::Point2X => {
                    let _y = self.pop().number();
                }
                Instruction::Point2Y => {
                    let y = self.pop().number();
                    let _x = self.pop().number();
                    self.push(y);
                }
                Instruction::Point3X => {
                    let _z = self.pop().number();
                    let _y = self.pop().number();
                }
                Instruction::Point3Y => {
                    let _z = self.pop().number();
                    let y = self.pop().number();
                    let _x = self.pop().number();
                    self.push(y);
                }
                Instruction::Point3Z => {
                    let z = self.pop().number();
                    let _y = self.pop().number();
                    let _x = self.pop().number();
                    self.push(z);
                }
                Instruction::Hypot2 => {
                    let y = self.pop().number();
                    let x = self.pop().number();
                    self.push(if !x.is_nan() && !y.is_nan() {
                        x.hypot(y)
                    } else {
                        f64::NAN
                    });
                }
                Instruction::Hypot3 => {
                    let z = self.pop().number();
                    let y = self.pop().number();
                    let x = self.pop().number();
                    self.push(math::hypot3(x, y, z));
                }
                Instruction::Sqrt => {
                    let a = self.pop().number();
                    self.push(a.sqrt());
                }

                Instruction::Ln => {
                    let a = self.pop().number();
                    self.push(a.ln());
                }
                Instruction::Exp => {
                    let a = self.pop().number();
                    self.push(a.exp());
                }
                Instruction::Erf => todo!(),
                Instruction::Sin => {
                    let a = self.pop().number();
                    self.push(a.sin());
                }
                Instruction::Cos => {
                    let a = self.pop().number();
                    self.push(a.cos());
                }
                Instruction::Tan => {
                    let a = self.pop().number();
                    self.push(a.tan());
                }
                Instruction::Sec => {
                    let a = self.pop().number();
                    self.push(1.0 / a.cos());
                }
                Instruction::Csc => {
                    let a = self.pop().number();
                    self.push(1.0 / a.sin());
                }
                Instruction::Cot => {
                    let a = self.pop().number();
                    self.push(1.0 / a.tan());
                }
                Instruction::Sinh => {
                    let a = self.pop().number();
                    self.push(a.sinh());
                }
                Instruction::Cosh => {
                    let a = self.pop().number();
                    self.push(a.cosh());
                }
                Instruction::Tanh => {
                    let a = self.pop().number();
                    self.push(a.tanh());
                }
                Instruction::Sech => {
                    let a = self.pop().number();
                    self.push(1.0 / a.cosh());
                }
                Instruction::Csch => {
                    let a = self.pop().number();
                    self.push(1.0 / a.sinh());
                }
                Instruction::Coth => {
                    let a = self.pop().number();
                    self.push(1.0 / a.tanh());
                }
                Instruction::Asin => {
                    let a = self.pop().number();
                    self.push(a.asin());
                }
                Instruction::Acos => {
                    let a = self.pop().number();
                    self.push(a.acos());
                }
                Instruction::Atan => {
                    let a = self.pop().number();
                    self.push(a.atan());
                }
                Instruction::Atan2 => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a.atan2(b));
                }
                Instruction::Asec => {
                    let a = self.pop().number();
                    self.push((1.0 / a).acos());
                }
                Instruction::Acsc => {
                    let a = self.pop().number();
                    self.push((1.0 / a).asin());
                }
                Instruction::Acot => {
                    let a = self.pop().number();
                    self.push((1.0 / a).atan());
                }
                Instruction::Asinh => {
                    let a = self.pop().number();
                    self.push(a.asinh());
                }
                Instruction::Acosh => {
                    let a = self.pop().number();
                    self.push(a.acosh());
                }
                Instruction::Atanh => {
                    let a = self.pop().number();
                    self.push(a.atanh());
                }
                Instruction::Asech => {
                    let a = self.pop().number();
                    self.push((1.0 / a).acosh());
                }
                Instruction::Acsch => {
                    let a = self.pop().number();
                    self.push((1.0 / a).asinh());
                }
                Instruction::Acoth => {
                    let a = self.pop().number();
                    self.push((1.0 / a).atanh());
                }
                Instruction::Abs => {
                    let a = self.pop().number();
                    self.push(a.abs());
                }
                Instruction::Sgn => {
                    let a = self.pop().number();
                    self.push(if a < 0.0 {
                        -1.0
                    } else if a > 0.0 {
                        1.0
                    } else if a == 0.0 {
                        0.0
                    } else {
                        f64::NAN
                    });
                }
                Instruction::Round => {
                    let a = self.pop().number();
                    self.push(a.round());
                }
                Instruction::RoundWithPrecision => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    let p = 10f64.powi(b.round().clamp(-1e3, 1e3) as i32);
                    self.push((a * p).round() / p);
                }
                Instruction::Floor => {
                    let a = self.pop().number();
                    self.push(a.floor());
                }
                Instruction::Ceil => {
                    let a = self.pop().number();
                    self.push(a.ceil());
                }
                Instruction::Mod => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a - (a / b).floor() * b);
                }
                Instruction::Midpoint2 => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax.midpoint(bx));
                    self.push(ay.midpoint(by));
                }
                Instruction::Midpoint3 => {
                    let bz = self.pop().number();
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let az = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax.midpoint(bx));
                    self.push(ay.midpoint(by));
                    self.push(az.midpoint(bz));
                }
                Instruction::Distance2 => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push((bx - ax).hypot(by - ay));
                }
                Instruction::Distance3 => {
                    let bz = self.pop().number();
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let az = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(math::hypot3(bx - ax, by - ay, bz - az));
                }
                Instruction::Min => {
                    let a = self.pop().list();
                    let a = a.borrow();
                    if a.is_empty() {
                        self.push(f64::NAN);
                    } else {
                        let mut result = f64::INFINITY;
                        for x in a.iter() {
                            if x.is_nan() {
                                result = f64::NAN;
                                break;
                            }
                            result = x.min(result);
                        }
                        self.push(result);
                    }
                }
                Instruction::Max => {
                    let a = self.pop().list();
                    let a = a.borrow();
                    if a.is_empty() {
                        self.push(f64::NAN);
                    } else {
                        let mut result = -f64::INFINITY;
                        for x in a.iter() {
                            if x.is_nan() {
                                result = f64::NAN;
                                break;
                            }
                            result = x.max(result);
                        }
                        self.push(result);
                    }
                }
                Instruction::Median => {
                    let a = self.pop().list();
                    let a = a.borrow();
                    if a.is_empty() || a.contains(&f64::NAN) {
                        self.push(f64::NAN);
                    } else {
                        self.push(medians::Medianf64::medf_unchecked(a.as_slice()));
                    }
                }
                Instruction::Argmin => {
                    let a = self.pop().list();
                    let a = a.borrow();
                    let mut result = 0.0;
                    let mut index = 0;
                    for (i, &x) in a.iter().enumerate() {
                        if x.is_nan() {
                            index = 0;
                            break;
                        }
                        if i == 0 || x < result {
                            result = x;
                            index = i + 1;
                        }
                    }
                    self.push(index as f64);
                }
                Instruction::Argmax => {
                    let a = self.pop().list();
                    let a = a.borrow();
                    let mut result = 0.0;
                    let mut index = 0;
                    for (i, &x) in a.iter().enumerate() {
                        if x.is_nan() {
                            index = 0;
                            break;
                        }
                        if i == 0 || x > result {
                            result = x;
                            index = i + 1;
                        }
                    }
                    self.push(index as f64);
                }
                Instruction::Total => {
                    let a = self.pop().list();
                    self.push(a.borrow().iter().sum::<f64>());
                }
                Instruction::Total2 => {
                    let a = self.pop().list();
                    let (x, y) = a
                        .borrow()
                        .as_chunks()
                        .0
                        .iter()
                        .fold((0.0, 0.0), |(x, y), [u, v]| (x + u, y + v));
                    self.push(x);
                    self.push(y);
                }
                Instruction::Total3 => {
                    let a = self.pop().list();
                    let (x, y, z) = a
                        .borrow()
                        .as_chunks()
                        .0
                        .iter()
                        .fold((0.0, 0.0, 0.0), |(x, y, z), [u, v, w]| {
                            (x + u, y + v, z + w)
                        });
                    self.push(x);
                    self.push(y);
                    self.push(z);
                }
                Instruction::Mean => {
                    let a = self.pop().list();
                    let a = a.borrow();
                    self.push(a.iter().sum::<f64>() / a.len() as f64);
                }
                Instruction::Mean2 => {
                    let a = self.pop().list();
                    let a = a.borrow();
                    let (x, y) = a
                        .as_chunks()
                        .0
                        .iter()
                        .fold((0.0, 0.0), |(x, y), [u, v]| (x + u, y + v));
                    self.push(x / a.len() as f64);
                    self.push(y / a.len() as f64);
                }
                Instruction::Mean3 => {
                    let a = self.pop().list();
                    let a = a.borrow();
                    let (x, y, z) = a
                        .as_chunks()
                        .0
                        .iter()
                        .fold((0.0, 0.0, 0.0), |(x, y, z), [u, v, w]| {
                            (x + u, y + v, z + w)
                        });
                    self.push(x / a.len() as f64);
                    self.push(y / a.len() as f64);
                    self.push(z / a.len() as f64);
                }
                Instruction::Count => {
                    let a = self.pop().list();
                    self.push(a.borrow().len() as f64);
                }
                Instruction::Count2 => {
                    let a = self.pop().list();
                    self.push(a.borrow().len() as f64 / 2.0);
                }
                Instruction::Count3 => {
                    let a = self.pop().list();
                    self.push(a.borrow().len() as f64 / 3.0);
                }
                Instruction::CountPolygonList => {
                    let a = self.pop().polygon_list();
                    self.push(a.borrow().len() as f64);
                }
                Instruction::Repeat => {
                    let count = self.pop().number().round().max(0.0) as usize;
                    let value = self.pop().number();
                    self.push(Rc::new(RefCell::new(vec![value; count])));
                }
                Instruction::Repeat2 => {
                    let count = self.pop().number().round().max(0.0) as usize;
                    let y = self.pop().number();
                    let x = self.pop().number();
                    self.push(Rc::new(RefCell::new([x, y].repeat(count))));
                }
                Instruction::Repeat3 => {
                    let count = self.pop().number().round().max(0.0) as usize;
                    let z = self.pop().number();
                    let y = self.pop().number();
                    let x = self.pop().number();
                    self.push(Rc::new(RefCell::new([x, y, z].repeat(count))));
                }
                Instruction::RepeatPolygon => {
                    let count = self.pop().number().round().max(0.0) as usize;
                    let value = self.pop().list();
                    self.push(Rc::new(RefCell::new(vec![value; count])));
                }
                Instruction::RepeatList => {
                    let counts = self.pop().list();
                    let counts = counts.borrow();
                    let values = self.pop().list();
                    let values = values.borrow();
                    let mut list = vec![];

                    // TODO "When the arguments of 'repeat' are lists, they must have the same length."
                    for (&value, &count) in zip(values.iter(), counts.iter()) {
                        let count = count.round().max(0.0) as usize;
                        list.resize(list.len() + count, value);
                    }

                    self.push(Rc::new(RefCell::new(list)));
                }
                Instruction::Repeat2List => {
                    let counts = self.pop().list();
                    let counts = counts.borrow();
                    let values = self.pop().list();
                    let values = values.borrow();
                    let mut list = vec![];

                    for (&[x, y], &count) in zip(values.as_chunks().0, counts.iter()) {
                        let count = count.round().max(0.0) as usize;
                        list.reserve(count * 2);
                        for _ in 0..count {
                            list.push(x);
                            list.push(y);
                        }
                    }

                    self.push(Rc::new(RefCell::new(list)));
                }
                Instruction::Repeat3List => {
                    let counts = self.pop().list();
                    let counts = counts.borrow();
                    let values = self.pop().list();
                    let values = values.borrow();
                    let mut list = vec![];

                    for (&[x, y, z], &count) in zip(values.as_chunks().0, counts.iter()) {
                        let count = count.round().max(0.0) as usize;
                        list.reserve(count * 3);
                        for _ in 0..count {
                            list.push(x);
                            list.push(y);
                            list.push(z);
                        }
                    }

                    self.push(Rc::new(RefCell::new(list)));
                }
                Instruction::RepeatPolygonList => {
                    let counts = self.pop().list();
                    let counts = counts.borrow();
                    let values = self.pop().polygon_list();
                    let values = values.borrow();
                    let mut list = vec![];

                    for (value, &count) in zip(values.iter(), counts.iter()) {
                        let count = count.round().max(0.0) as usize;
                        list.resize(list.len() + count, Rc::clone(value));
                    }

                    self.push(Rc::new(RefCell::new(list)));
                }
                Instruction::Unique => {
                    let a = self.pop().list();
                    let mut seen = HashSet::new();
                    self.push(Rc::new(RefCell::new(
                        a.borrow()
                            .iter()
                            .cloned()
                            .filter(|&x| seen.insert(OrderedFloat(x)))
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::Unique2 => {
                    let a = self.pop().list();
                    let mut seen = HashSet::new();
                    self.push(Rc::new(RefCell::new(
                        a.borrow()
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .cloned()
                            .filter(|&p| seen.insert(p.map(OrderedFloat)))
                            .flatten()
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::Unique3 => {
                    let a = self.pop().list();
                    let mut seen = HashSet::new();
                    self.push(Rc::new(RefCell::new(
                        a.borrow()
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .cloned()
                            .filter(|&p| seen.insert(p.map(OrderedFloat)))
                            .flatten()
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::UniquePolygon => {
                    let a = self.pop().polygon_list();
                    let mut seen = HashSet::new();
                    self.push(Rc::new(RefCell::new(
                        a.borrow()
                            .iter()
                            .filter(|p| {
                                let p = p.borrow();
                                let mut h = DefaultHasher::new();
                                for &x in p.as_slice() {
                                    OrderedFloat(x).hash(&mut h);
                                }
                                seen.insert(h.finish())
                            })
                            .cloned()
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::UniquePerm => {
                    let a = self.pop().list();
                    let mut seen = HashSet::new();
                    self.push(Rc::new(RefCell::new(
                        a.borrow()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &x)| seen.insert(OrderedFloat(x)).then_some(i as f64))
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::UniquePerm2 => {
                    let a = self.pop().list();
                    let mut seen = HashSet::new();
                    self.push(Rc::new(RefCell::new(
                        a.borrow()
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &p)| {
                                seen.insert(p.map(OrderedFloat)).then_some(i as f64)
                            })
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::UniquePerm3 => {
                    let a = self.pop().list();
                    let mut seen = HashSet::new();
                    self.push(Rc::new(RefCell::new(
                        a.borrow()
                            .as_chunks::<3>()
                            .0
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &p)| {
                                seen.insert(p.map(OrderedFloat)).then_some(i as f64)
                            })
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::UniquePermPolygon => {
                    let a = self.pop().polygon_list();
                    let mut seen = HashSet::new();
                    self.push(Rc::new(RefCell::new(
                        a.borrow()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, p)| {
                                let p = p.borrow();
                                let mut h = DefaultHasher::new();
                                for &x in p.as_slice() {
                                    OrderedFloat(x).hash(&mut h);
                                }
                                seen.insert(h.finish()).then_some(i as f64)
                            })
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::Sort => {
                    let mut a = Rc::unwrap_or_clone(self.pop().list()).take();
                    a.sort_unstable_by(f64::total_cmp);
                    self.push(Rc::new(RefCell::new(a)));
                }
                Instruction::SortKey => {
                    let key = self.pop().list();
                    let key = key.borrow();
                    let list = self.pop().list();
                    let list = list.borrow();
                    self.push(Rc::new(RefCell::new(
                        math::sort_perm(&key[..key.len().min(list.len())])
                            .iter()
                            .map(|&i| list[i])
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::SortKey2 => {
                    let key = self.pop().list();
                    let key = key.borrow();
                    let list = self.pop().list();
                    let list = list.borrow();
                    self.push(Rc::new(RefCell::new(
                        math::sort_perm(&key[..key.len().min(list.len() / 2)])
                            .iter()
                            .flat_map(|&i| [list[2 * i], list[2 * i + 1]])
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::SortKey3 => {
                    let key = self.pop().list();
                    let key = key.borrow();
                    let list = self.pop().list();
                    let list = list.borrow();
                    self.push(Rc::new(RefCell::new(
                        math::sort_perm(&key[..key.len().min(list.len() / 3)])
                            .iter()
                            .flat_map(|&i| [list[3 * i], list[3 * i + 1], list[3 * i + 2]])
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::SortKeyPolygon => {
                    let key = self.pop().list();
                    let key = key.borrow();
                    let list = self.pop().polygon_list();
                    let list = list.borrow();
                    self.push(Rc::new(RefCell::new(
                        math::sort_perm(&key[..key.len().min(list.len())])
                            .iter()
                            .map(|&i| Rc::clone(&list[i]))
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::SortPerm => {
                    let key = self.pop().list();
                    let key = key.borrow();
                    self.push(Rc::new(RefCell::new(
                        math::sort_perm(&key)
                            .iter()
                            .map(|i| *i as f64)
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::Polygon => {
                    // noop
                }
                Instruction::Vertices => {
                    // noop
                }
                Instruction::Push => {
                    let b = self.pop().number();
                    let a = Rc::unwrap_or_clone(self.pop().list());
                    a.borrow_mut().push(b);
                    self.push(Rc::new(a));
                }
                Instruction::Push2 => {
                    let y = self.pop().number();
                    let x = self.pop().number();
                    let a = Rc::unwrap_or_clone(self.pop().list());
                    a.borrow_mut().extend([x, y]);
                    self.push(Rc::new(a));
                }
                Instruction::Push3 => {
                    let z = self.pop().number();
                    let y = self.pop().number();
                    let x = self.pop().number();
                    let a = Rc::unwrap_or_clone(self.pop().list());
                    a.borrow_mut().extend([x, y, z]);
                    self.push(Rc::new(a));
                }
                Instruction::PushPolygon => {
                    let b = self.pop().list();
                    let a = Rc::unwrap_or_clone(self.pop().polygon_list());
                    a.borrow_mut().push(b);
                    self.push(Rc::new(a));
                }
                Instruction::Concat | Instruction::Concat2 | Instruction::Concat3 => {
                    let b = self.pop().list();
                    let a = Rc::unwrap_or_clone(self.pop().list());
                    a.borrow_mut().extend_from_slice(&b.borrow());
                    self.push(Rc::new(a));
                }
                Instruction::ConcatPolygon => {
                    let b = self.pop().polygon_list();
                    let a = Rc::unwrap_or_clone(self.pop().polygon_list());
                    a.borrow_mut().extend_from_slice(&b.borrow());
                    self.push(Rc::new(a));
                }

                Instruction::And => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a * b);
                }
                Instruction::MinInternal => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a.min(b));
                }
                Instruction::Index => {
                    let b = self.pop().number().floor() - 1.0;
                    let a = self.pop().list();
                    let a = a.borrow();

                    self.push(if 0.0 <= b && b < a.len() as f64 {
                        a[b as usize]
                    } else {
                        f64::NAN
                    });
                }
                Instruction::Index2 => {
                    let b = (self.pop().number().floor() - 1.0) * 2.0;
                    let a = self.pop().list();
                    let a = a.borrow();

                    if 0.0 <= b && b < a.len() as f64 {
                        self.push(a[b as usize]);
                        self.push(a[b as usize + 1]);
                    } else {
                        self.push(f64::NAN);
                        self.push(f64::NAN);
                    }
                }
                Instruction::Index3 => {
                    let b = (self.pop().number().floor() - 1.0) * 3.0;
                    let a = self.pop().list();
                    let a = a.borrow();

                    if 0.0 <= b && b < a.len() as f64 {
                        self.push(a[b as usize]);
                        self.push(a[b as usize + 1]);
                        self.push(a[b as usize + 2]);
                    } else {
                        self.push(f64::NAN);
                        self.push(f64::NAN);
                        self.push(f64::NAN);
                    }
                }
                Instruction::IndexPolygonList => {
                    let b = self.pop().number().floor() - 1.0;
                    let a = self.pop().polygon_list();
                    let a = a.borrow();

                    self.push(if 0.0 <= b && b < a.len() as f64 {
                        Rc::clone(&a[b as usize])
                    } else {
                        Rc::new(RefCell::new(vec![]))
                    });
                }
                Instruction::UncheckedIndex(index) => {
                    let b = self.pop().number() as usize;
                    let a = self.peek(index).list();
                    self.push(a.borrow()[b]);
                }
                Instruction::UncheckedIndex2(index) => {
                    let b = self.pop().number() as usize * 2;
                    let a = self.peek(index).list();
                    let a = a.borrow();
                    self.push(a[b]);
                    self.push(a[b + 1]);
                }
                Instruction::UncheckedIndex3(index) => {
                    let b = self.pop().number() as usize * 3;
                    let a = self.peek(index).list();
                    let a = a.borrow();
                    self.push(a[b]);
                    self.push(a[b + 1]);
                    self.push(a[b + 2]);
                }
                Instruction::UncheckedIndexPolygonList(index) => {
                    let b = self.pop().number() as usize;
                    let a = self.peek(index).polygon_list();
                    self.push(Rc::clone(&a.borrow()[b]));
                }
                Instruction::BuildList(count) => {
                    let mut list = vec![0.0; count];

                    for v in list.iter_mut().rev() {
                        *v = self.pop().number();
                    }

                    self.push(Rc::new(RefCell::new(list)));
                }
                Instruction::BuildPolygonList(count) => {
                    let mut list = vec![];

                    for _ in 0..count {
                        list.push(self.pop().list());
                    }

                    list.reverse();
                    self.push(Rc::new(RefCell::new(list)));
                }
                Instruction::BuildListFromRange => {
                    let b = self.pop().number().round() as i64;
                    let a = self.pop().number().round() as i64;

                    self.push(Rc::new(RefCell::new(if a <= b {
                        (a..=b).map(|i| i as f64).collect::<Vec<_>>()
                    } else {
                        (b..=a).rev().map(|i| i as f64).collect()
                    })));
                }
                Instruction::Append(index) => {
                    let a = self.pop().number();
                    self.peek(index).clone().list().borrow_mut().push(a);
                }
                Instruction::Append2(index) => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let a = self.peek(index).clone().list();
                    let mut a = a.borrow_mut();
                    a.push(bx);
                    a.push(by);
                }
                Instruction::Append3(index) => {
                    let bz = self.pop().number();
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let a = self.peek(index).clone().list();
                    let mut a = a.borrow_mut();
                    a.push(bx);
                    a.push(by);
                    a.push(bz);
                }
                Instruction::AppendPolygonList(index) => {
                    let a = self.pop().list();
                    self.peek(index).clone().polygon_list().borrow_mut().push(a);
                }
                Instruction::CountSpecific(index) => {
                    let a = self.peek(index).list();
                    self.push(a.borrow().len() as f64);
                }
                Instruction::CountSpecific2(index) => {
                    let a = self.peek(index).list();
                    self.push(a.borrow().len() as f64 / 2.0);
                }
                Instruction::CountSpecific3(index) => {
                    let a = self.peek(index).list();
                    self.push(a.borrow().len() as f64 / 3.0);
                }
                Instruction::CountSpecificPolygonList(index) => {
                    let a = self.peek(index).polygon_list();
                    self.push(a.borrow().len() as f64);
                }
                Instruction::Slider => {
                    let step = self.pop().number();
                    let max = self.pop().number();
                    let min = self.pop().number();
                    let value = self.pop().number();
                    self.push(math::apply_slider(value, min, max, step));
                }

                Instruction::StartArgs => {
                    self.push(0.0);
                }
                Instruction::EndArgs(n_args) => {
                    let index = self.stack.len() - 1 - n_args;
                    self.stack[index] = Value::Number(self.pc as f64 + 1.0);
                }
                Instruction::Jump(pc) => {
                    self.pc = pc;
                }
                Instruction::JumpIfFalse(pc) => {
                    let a = self.pop().number();

                    if a == 0.0 {
                        self.pc = pc;
                    }
                }

                Instruction::Return1 => {
                    self.pc = self.stack.remove(self.stack.len() - 2).number() as usize;
                }
                Instruction::Return2 => {
                    self.pc = self.stack.remove(self.stack.len() - 3).number() as usize;
                }
            }
        }

        if COUNT_INSTRUCTIONS {
            let mut counts = vec![];
            for (i, c) in instruction_counts.iter().enumerate() {
                if *c > 0 {
                    counts.push((
                        InstructionDiscriminants::from_repr(i).unwrap().to_string(),
                        *c,
                    ));
                }
            }
            counts.push((
                "Total Instruction Count".into(),
                counts.iter().map(|(_, c)| c).sum(),
            ));
            counts.sort_by_key(|(_, c)| -*c);
            let l = counts
                .iter()
                .map(|(_, c)| c.to_string().len())
                .max()
                .unwrap_or(0);

            for (n, c) in counts {
                println!("{c: >l$} {n}");
            }
            println!();
        }
    }
}
