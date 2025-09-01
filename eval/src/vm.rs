use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Write,
    hash::{DefaultHasher, Hash, Hasher},
    rc::Rc,
};

use ordered_float::OrderedFloat;
use strum::{Display, EnumCount, EnumDiscriminants, FromRepr};

#[derive(Debug, Clone, Copy, PartialEq, EnumCount, EnumDiscriminants)]
#[strum_discriminants(derive(FromRepr, Display))]
pub enum Instruction {
    Start,
    Halt,
    Unreachable,

    LoadConst(f64),
    Load(usize),
    Load2(usize),
    Store(usize),
    Store2(usize),
    LoadStore(usize),
    Load1Store2(usize),
    Load2Store1(usize),
    Load2Store2(usize),
    Copy(usize),
    Swap(usize),
    Swap2(usize),
    Pop(usize),

    Neg,
    Neg2,
    Add,
    Add2,
    Sub,
    Sub2,
    Mul,
    Mul1_2,
    Div,
    Div2_1,
    Pow,
    Dot,
    Point,

    Equal,
    LessThan,
    LessThanEqual,
    GreaterThan,
    GreaterThanEqual,

    PointX,
    PointY,
    Hypot,
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
    Midpoint,
    Distance,
    Min,
    Max,
    Median,
    Argmin,
    Argmax,
    Total,
    Total2,
    Mean,
    Mean2,
    Count,
    Count2,
    CountPolygonList,
    Unique,
    Unique2,
    UniquePolygon,
    UniquePerm,
    UniquePerm2,
    UniquePermPolygon,
    Sort,
    SortKey,
    SortKey2,
    SortKeyPolygon,
    SortPerm,
    Polygon,
    Push,
    Push2,
    PushPolygon,
    Concat,
    Concat2,
    ConcatPolygon,

    MinInternal,
    Index,
    Index2,
    IndexPolygonList,
    UncheckedIndex(usize),
    UncheckedIndex2(usize),
    UncheckedIndexPolygonList(usize),
    BuildList(usize),
    BuildPolygonList(usize),
    BuildListFromRange,
    Append(usize),
    Append2(usize),
    AppendPolygonList(usize),
    CountSpecific(usize),
    CountSpecific2(usize),
    CountSpecificPolygonList(usize),

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

fn sort_perm(list: &[f64]) -> Vec<usize> {
    let mut indices = (0..list.len()).collect::<Vec<_>>();
    indices.sort_by(|a, b| list[*a].total_cmp(&list[*b]));
    indices
}

#[derive(Debug, Default)]
pub struct Vm<'a> {
    pub program: Vec<Instruction>,
    pub pc: usize,
    pub stack: Vec<Value>,
    pub vars: Vec<Value>,
    pub names: Option<&'a [String]>,
}

pub const UNINITIALIZED: f64 = -9.405704329145218e-291;

impl<'a> Vm<'a> {
    pub fn with_program(program: Vec<Instruction>) -> Vm<'a> {
        let n_vars = program
            .iter()
            .map(|i| match i {
                Instruction::Load(j) | Instruction::Store(j) => j + 1,
                Instruction::Load2(j) | Instruction::Store2(j) => j + 2,
                _ => 0,
            })
            .max()
            .unwrap_or(0);
        Vm {
            program,
            vars: vec![Value::Number(UNINITIALIZED); n_vars],
            ..Default::default()
        }
    }

    pub fn set_names(&mut self, names: &'a [String]) {
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

    fn name(&self, index: usize) -> Option<&str> {
        self.names.as_ref().map(|n| n[index].as_ref())
    }

    fn load(&self, index: usize) -> Value {
        let value = self.vars[index].clone();

        if matches!(value, Value::Number(UNINITIALIZED)) {
            let mut msg: String = "".into();
            write!(msg, "variable is uninitialized: {index}").unwrap();

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
                    self.push(self.load(index + 1));
                }
                Instruction::Store(index) => self.vars[index] = self.pop(),
                Instruction::Store2(index) => {
                    self.vars[index + 1] = self.pop();
                    self.vars[index] = self.pop();
                }
                Instruction::LoadStore(index) => {
                    std::mem::swap(&mut self.vars[index], self.stack.last_mut().unwrap());
                }
                Instruction::Load1Store2(index) => {
                    self.vars[index + 1] = self.pop();
                    std::mem::swap(&mut self.vars[index], self.stack.last_mut().unwrap());
                }
                Instruction::Load2Store1(index) => {
                    std::mem::swap(&mut self.vars[index], self.stack.last_mut().unwrap());
                    self.push(self.vars[index + 1].clone());
                }
                Instruction::Load2Store2(index) => {
                    let len = self.stack.len();
                    std::mem::swap(&mut self.vars[index], &mut self.stack[len - 2]);
                    std::mem::swap(&mut self.vars[index + 1], &mut self.stack[len - 1]);
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
                Instruction::Pow => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a.powf(b));
                }
                Instruction::Dot => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax * bx + ay * by);
                }
                Instruction::Point => {
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

                Instruction::PointX => {
                    self.pop().number();
                }
                Instruction::PointY => {
                    let y = self.pop().number();
                    self.pop().number();
                    self.push(y);
                }
                Instruction::Hypot => {
                    let y = self.pop().number();
                    let x = self.pop().number();
                    self.push(x.hypot(y));
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
                Instruction::Midpoint => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax.midpoint(bx));
                    self.push(ay.midpoint(by));
                }
                Instruction::Distance => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push((bx - ax).hypot(by - ay));
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
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
                    self.push(x);
                    self.push(y);
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
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .fold((0.0, 0.0), |(x, y), p| (x + p[0], y + p[1]));
                    self.push(x / a.len() as f64);
                    self.push(y / a.len() as f64);
                }
                Instruction::Count => {
                    let a = self.pop().list();
                    self.push(a.borrow().len() as f64);
                }
                Instruction::Count2 => {
                    let a = self.pop().list();
                    self.push(a.borrow().len() as f64 / 2.0);
                }
                Instruction::CountPolygonList => {
                    let a = self.pop().polygon_list();
                    self.push(a.borrow().len() as f64);
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
                            .filter(|&[x, y]| seen.insert((OrderedFloat(x), OrderedFloat(y))))
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
                            .filter_map(|(i, &[x, y])| {
                                seen.insert((OrderedFloat(x), OrderedFloat(y)))
                                    .then_some(i as f64)
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
                        sort_perm(&key[..key.len().min(list.len())])
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
                        sort_perm(&key[..key.len().min(list.len())])
                            .iter()
                            .flat_map(|&i| [list[2 * i], list[2 * i + 1]])
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::SortKeyPolygon => {
                    let key = self.pop().list();
                    let key = key.borrow();
                    let list = self.pop().polygon_list();
                    let list = list.borrow();
                    self.push(Rc::new(RefCell::new(
                        sort_perm(&key[..key.len().min(list.len())])
                            .iter()
                            .map(|&i| Rc::clone(&list[i]))
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::SortPerm => {
                    let key = self.pop().list();
                    let key = key.borrow();
                    self.push(Rc::new(RefCell::new(
                        sort_perm(&key)
                            .iter()
                            .map(|i| *i as f64)
                            .collect::<Vec<_>>(),
                    )));
                }
                Instruction::Polygon => {
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
                Instruction::PushPolygon => {
                    let b = self.pop().list();
                    let a = Rc::unwrap_or_clone(self.pop().polygon_list());
                    a.borrow_mut().push(b);
                    self.push(Rc::new(a));
                }
                Instruction::Concat | Instruction::Concat2 => {
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

                Instruction::MinInternal => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a.min(b));
                }
                Instruction::Index => {
                    let b = self.pop().number().floor() - 1.0;
                    let a = self.pop().list();
                    let a = a.borrow();

                    self.push(if b < 0.0 || b >= a.len() as f64 {
                        f64::NAN
                    } else {
                        a[b as usize]
                    });
                }
                Instruction::Index2 => {
                    let b = (self.pop().number().floor() - 1.0) * 2.0;
                    let a = self.pop().list();
                    let a = a.borrow();

                    if b < 0.0 || b >= a.len() as f64 {
                        self.push(f64::NAN);
                        self.push(f64::NAN);
                    } else {
                        self.push(a[b as usize]);
                        self.push(a[b as usize + 1]);
                    }
                }
                Instruction::IndexPolygonList => {
                    let b = self.pop().number().floor() - 1.0;
                    let a = self.pop().polygon_list();
                    let a = a.borrow();

                    self.push(if b < 0.0 || b >= a.len() as f64 {
                        Rc::new(RefCell::new(vec![]))
                    } else {
                        Rc::clone(&a[b as usize])
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
                Instruction::CountSpecificPolygonList(index) => {
                    let a = self.peek(index).polygon_list();
                    self.push(a.borrow().len() as f64);
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
