use std::{cell::RefCell, fmt::Write, rc::Rc};

#[derive(Debug, Clone, Copy, PartialEq)]
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
    Mul2_1,
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
    Abs,
    Floor,
    Ceil,
    Mod,
    Log2,
    Sqrt,
    Sin,
    Cos,
    Tan,
    Asin,
    Acos,
    Atan,
    Atan2,
    Distance,
    Min,
    Hypot,

    Index,
    Index2,
    UncheckedIndex(usize),
    UncheckedIndex2(usize),
    BuildList(usize),
    BuildListFromRange,
    Append(usize),
    Append2(usize),
    Count,
    Count2,
    CountSpecific(usize),
    CountSpecific2(usize),

    StartArgs,
    EndArgs(usize),
    Jump(usize),
    JumpIfFalse(usize),
    Return1,
    Return2,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    List(Rc<RefCell<Vec<f64>>>),
}

impl Value {
    pub fn number(self) -> f64 {
        match self {
            Value::Number(v) => v,
            _ => panic!("value is not a number: {self:?}"),
        }
    }

    pub fn list(self) -> Rc<RefCell<Vec<f64>>> {
        match self {
            Value::List(v) => v,
            _ => panic!("value is not a list: {self:?}"),
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
        }
    }
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
            write!(msg, "variable is uninitialized: {}", index).unwrap();

            if let Some(name) = self.name(index) {
                write!(msg, " ({})", name).unwrap();
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

        while self.pc < self.program.len() {
            let instruction = self.program[self.pc];

            if print_trace {
                print!("{} {:?}", self.pc, instruction);

                match instruction {
                    Instruction::Load(index)
                    | Instruction::Load2(index)
                    | Instruction::Store(index)
                    | Instruction::Store2(index) => {
                        if let Some(name) = self.name(index) {
                            print!(" ({})", name);
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
                Instruction::Mul2_1 => {
                    let b = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push(ax * b);
                    self.push(ay * b);
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
                Instruction::Abs => {
                    let a = self.pop().number();
                    self.push(a.abs());
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
                Instruction::Log2 => {
                    let a = self.pop().number();
                    self.push(a.log2());
                }
                Instruction::Sqrt => {
                    let a = self.pop().number();
                    self.push(a.sqrt());
                }
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
                    let x = self.pop().number();
                    let y = self.pop().number();
                    self.push(y.atan2(x));
                }
                Instruction::Distance => {
                    let by = self.pop().number();
                    let bx = self.pop().number();
                    let ay = self.pop().number();
                    let ax = self.pop().number();
                    self.push((bx - ax).hypot(by - ay));
                }
                Instruction::Min => {
                    let b = self.pop().number();
                    let a = self.pop().number();
                    self.push(a.min(b));
                }
                Instruction::Hypot => {
                    let y = self.pop().number();
                    let x = self.pop().number();
                    self.push(x.hypot(y));
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
                Instruction::BuildList(count) => {
                    let mut list = vec![0.0; count];

                    for v in list.iter_mut().rev() {
                        *v = self.pop().number();
                    }

                    self.push(Rc::new(RefCell::new(list)));
                }
                Instruction::BuildListFromRange => {
                    let b = self.pop().number().round() as i64;
                    let a = self.pop().number().round() as i64;

                    self.push(Rc::new(RefCell::new(if a <= b {
                        (a..=b).map(|i| i as f64).collect()
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
                Instruction::Count => {
                    let a = self.pop().list();
                    self.push(a.borrow().len() as f64);
                }
                Instruction::Count2 => {
                    let a = self.pop().list();
                    self.push(a.borrow().len() as f64 / 2.0);
                }
                Instruction::CountSpecific(index) => {
                    let a = self.peek(index).list();
                    self.push(a.borrow().len() as f64);
                }
                Instruction::CountSpecific2(index) => {
                    let a = self.peek(index).list();
                    self.push(a.borrow().len() as f64 / 2.0);
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
    }
}
