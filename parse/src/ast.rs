#[derive(Debug, PartialEq, Clone, Copy)]
pub enum UnaryOperator {
    Neg,
    Fac,
    Sqrt,
    Norm,
    PointX,
    PointY,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Dot,
    Cross,
    Point,
    Index,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum ComparisonOperator {
    Equal,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[derive(Debug, PartialEq)]
pub struct ChainedComparison {
    pub operands: Vec<Expression>,
    pub operators: Vec<ComparisonOperator>,
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub enum SumProdKind {
    Sum,
    Prod,
}

#[derive(Debug, PartialEq)]
pub enum Expression {
    Number(f64),
    Identifier(String),
    List(Vec<Expression>),
    ListRange {
        before_ellipsis: Vec<Expression>,
        after_ellipsis: Vec<Expression>,
    },
    UnaryOperation {
        operation: UnaryOperator,
        arg: Box<Expression>,
    },
    BinaryOperation {
        operation: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
    CallOrMultiply {
        callee: String,
        args: Vec<Expression>,
    },
    Call {
        callee: String,
        args: Vec<Expression>,
    },
    ChainedComparison(ChainedComparison),
    Piecewise {
        test: Box<Expression>,
        consequent: Box<Expression>,
        alternate: Option<Box<Expression>>,
    },
    SumProd {
        kind: SumProdKind,
        variable: String,
        lower_bound: Box<Expression>,
        upper_bound: Box<Expression>,
        body: Box<Expression>,
    },
    With {
        body: Box<Expression>,
        substitutions: Vec<(String, Expression)>,
    },
    For {
        body: Box<Expression>,
        lists: Vec<(String, Expression)>,
    },
}

#[derive(Debug, PartialEq)]
pub enum ExpressionListEntry {
    Assignment {
        name: String,
        value: Expression,
    },
    FunctionDeclaration {
        name: String,
        parameters: Vec<String>,
        body: Expression,
    },
    Relation(ChainedComparison),
    Expression(Expression),
}
