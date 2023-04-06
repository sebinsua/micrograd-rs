use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;
use std::iter::Sum;
use std::ops::{Add, Sub, Mul, Div, Neg, Deref};
use std::fmt;


#[derive(Copy, Clone, Debug)]
pub enum Operation {
    Input,
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Negate,
    ReLU,
}


#[derive(Clone)]
pub struct InternalValueData {
    pub data: f64,
    pub gradient: f64,
    pub operation: Operation,
    _previous: Vec<Value>,
    _backward: Rc<dyn Fn()>,
}

impl InternalValueData {
    fn new(
        data: f64,
        operation: Operation,
        previous: Vec<Value>,
        backward: Rc<dyn Fn()>,
    ) -> InternalValueData {
        InternalValueData {
            data,
            gradient: 0.0,
            operation: operation,
            _previous: previous,
            _backward: backward,
        }
    }
}

impl Default for InternalValueData {
    fn default() -> InternalValueData {
        InternalValueData {
            data: 0.0,
            gradient: 0.0,
            operation: Operation::Input,
            _previous: vec![],
            _backward: Rc::new(move || {}),
        }
    }
}


#[derive(Clone)]
pub struct Value(pub Rc<RefCell<InternalValueData>>);

impl Value {
    pub fn from<T>(t: T) -> Value
    where T: Into<Value> {
        t.into()
    }

    pub fn new(value: InternalValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    pub fn operation(&self) -> Operation {
        self.borrow().operation
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn gradient(&self) -> f64 {
        self.borrow().gradient
    }

    pub fn previous(&self) -> Vec<Value> {
        self.borrow()._previous.clone()
    }    

    pub fn set_data(&self, data: f64) {
        self.borrow_mut().data = data;
    }

    pub fn decrement_data(&self, data: f64) {
        self.borrow_mut().data -= data;
    }

    pub fn set_gradient(&self, gradient: f64) {
        self.borrow_mut().gradient = gradient;
    }

    pub fn increment_gradient(&self, gradient: f64) {
        self.borrow_mut().gradient += gradient;
    }

    fn set_backward(&self, _backward: Rc<dyn Fn()>) {
        self.borrow_mut()._backward = _backward;
    }

    pub fn backward(&mut self) {
        // Topological order all of the children in the graph
        let mut topo = Vec::new();
        let mut visited = HashSet::new();
        fn build_topo(v: Value, visited: &mut HashSet<usize>, topo: &mut Vec<Value>) {
            let id = v.as_ptr() as usize;
            if !visited.contains(&id) {
                visited.insert(id);

                for child in v.previous() {
                    build_topo(child, visited, topo);
                }

                topo.push(v);
            }
        }

        build_topo(self.clone(), &mut visited, &mut topo);

        // Go one variable at a time and apply the chain rule to get its gradient.
        self.set_gradient(1.0);
        for v in topo.iter().rev() {
            let backward = &v.borrow()._backward;
            backward();
        }
    }

    pub fn pow(self, other: Value) -> Value {
        power(self, other)
    }

    pub fn powi(self, otheri: i32) -> Value {
        power(
            self, 
            Value::from(
                otheri as f64,
            )
        )
    }

    pub fn powf(self, otherf: f64) -> Value {
        power(
            self, 
            Value::from(
                otherf,
            )
        )
    }

    pub fn relu(self) -> Value {
        let a = self.data();
        let c = if a > 0.0 { a } else { 0.0 };

        let out = Value::new(
            InternalValueData::new(
                c,
                Operation::ReLU,
                vec![self.clone()],
                Rc::new(move || {})
            )
        );

        let cloned_out = out.clone();
        out.set_backward(
            Rc::new(move || {
                self.increment_gradient(
                    if cloned_out.data() > 0.0 { cloned_out.gradient() } else { 0.0 }
                );
            })
        );

        out
    }
}

impl Deref for Value {
    type Target = Rc<RefCell<InternalValueData>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<f64>> From<T> for Value {
    fn from(t: T) -> Value {
        Value::new(
            InternalValueData {
                data: t.into(),
                ..Default::default()
            }
        )
    }
}

impl Sum for Value {
    fn sum<I: Iterator<Item=Self>>(mut iter: I) -> Self {
        let initial = match iter.next() {
            Some(x) => x,
            None => panic!("Cannot sum an empty iterator"),
        };
        let tail = iter;

        tail.fold(initial, |acc, x| acc + x)
    }
}

fn add(s: Value, other: Value, operation: Operation) -> Value {
    let a = s.data();
    let b = other.data();
    let c = a + b;

    let out = Value::new(
        InternalValueData::new(
            c,
            operation,
            vec![s.clone(), other.clone()],
            Rc::new(move || {})
        )
    );

    let cloned_out = out.clone();
    out.set_backward(
        Rc::new(move || {
            s.increment_gradient(1.0 * cloned_out.gradient());
            other.increment_gradient(1.0 * cloned_out.gradient());
        })
    );

    out
}

fn subtract(s: Value, other: Value) -> Value {
    add(s, negate(other), Operation::Subtract)
}

fn multiply(s: Value, other: Value, operation: Operation) -> Value {
    let a = s.data();
    let b = other.data();
    let c = a * b;

    let out = Value::new(
        InternalValueData::new(
            c,
            operation,
            vec![s.clone(), other.clone()],
            Rc::new(move || {})
        )
    );

    let cloned_out = out.clone();
    out.set_backward(
        Rc::new(move || {
            s.increment_gradient(other.data() * cloned_out.gradient());
            other.increment_gradient(s.data() * cloned_out.gradient());
        })
    );

    out
}

fn divide(s: Value, other: Value) -> Value {
    multiply(s, other.powf(-1.0), Operation::Divide)
}

fn power(s: Value, other: Value) -> Value {
    let a = s.data();
    let b = other.data();
    let c = a.powf(b);

    let out = Value::new(
        InternalValueData::new(
            c,
            Operation::Power,
            vec![s.clone(), other.clone()],
            Rc::new(move || {})
        )
    );

    let cloned_out = out.clone();
    out.set_backward(
        Rc::new(move || {
            s.increment_gradient(other.data() * s.data().powf(other.data() - 1.0) * cloned_out.gradient())
        })
    );

    out
}

fn negate(s: Value) -> Value {
    multiply(
        s,
        Value::from(
            -1.0,
        ),
        Operation::Negate
    )
}


impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(self, other, Operation::Add)
    }
}

impl Add<Value> for f64 {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(
            Value::from(self),
            other,
            Operation::Add
        )
    }
}

impl Add<f64> for Value {
    type Output = Value;

    fn add(self, b: f64) -> Self::Output {
        add(
            self,
            Value::from(
                b,
            ),
            Operation::Add
        )
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        subtract(self, other)
    }
}

impl Sub<Value> for f64 {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        subtract(
            Value::from(self),
            other
        )
    }
}

impl Sub<f64> for Value {
    type Output = Value;

    fn sub(self, b: f64) -> Self::Output {
        subtract(
            self,
            Value::from(
                b,
            )
        )
    }
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        multiply(self, other, Operation::Multiply)
    }
}

impl Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        multiply(Value::from(self), other, Operation::Multiply)
    }
}

impl Mul<f64> for Value {
    type Output = Value;

    fn mul(self, b: f64) -> Self::Output {
        multiply(
            self,
            Value::from(
                b,
            ),
            Operation::Multiply
        )
    }
}

impl Div<Value> for Value {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        divide(self, other)
    }
}

impl Div<Value> for f64 {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        divide(Value::from(self), other)
    }
}

impl Div<f64> for Value {
    type Output = Value;

    fn div(self, b: f64) -> Self::Output {
        divide(
            self,
            Value::from(
                b,
            )
        )
    }
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        negate(self)
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data())
            .field("gradient", &self.gradient())
            .field("operation", &self.operation())
            .field("_backward", &"<closure>")
            .field("_previous", &self.previous().iter().map(|x| x.data()).collect::<Vec<f64>>())
            .finish()
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Value(operation={:?}, data={}, gradient={})", self.operation(), self.data(), self.gradient())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_add_value() {
        let a = Value::from(
            1.0,
        );
        let b = Value::from(
            2.0,
        );
        
        let c = a + b;
        assert_eq!(c.data(), 3.0);
    }

    #[test]
    fn test_value_add_f64() {
        let a = Value::from(
            1.0,
        );
        let b = 10.0;
        
        let c = a + b;
        assert_eq!(c.data(), 11.0);
    }

    #[test]
    fn test_value_sub_value() {
        let a = Value::from(
            15.0,
        );
        let b = Value::from(
            12.0,
        );
        
        let c = a - b;
        assert_eq!(c.data(), 3.0);
    }

    #[test]
    fn test_value_sub_f64() {
        let a = Value::from(
            16.0,
        );
        let b = 10.0;
        
        let c = a - b;
        assert_eq!(c.data(), 6.0);
    }

    #[test]
    fn test_value_multiply_value() {
        let a = Value::from(
            33.0,
        );
        let b = Value::from(
            3.0,
        );
        
        let c = a * b;
        assert_eq!(c.data(), 99.0);
    }

    #[test]
    fn test_value_multiply_f64() {
        let a = Value::from(
            33.0,
        );
        let b = 3.0;
        
        let c = a * b;
        assert_eq!(c.data(), 99.0);
    }

    #[test]
    fn test_value_divide_value() {
        let a = Value::from(
            50.0,
        );
        let b = Value::from(
            2.0,
        );
        
        let c = a / b;
        assert_eq!(c.data(), 25.0);
    }

    #[test]
    fn test_value_divide_f64() {
        let a = Value::from(
            20.0,
        );
        let b = 2.0;
        
        let c = a / b;
        assert_eq!(c.data(), 10.0);
    }

    #[test]
    fn test_value_pow_value() {
        let a = Value::from(
            2.0,
        );
        let b = Value::from(
            2.0,
        );
        
        let c = a.pow(b);
        assert_eq!(c.data(), 4.0);
    }

    #[test]
    fn test_value_pow_f64() {
        let a = Value::from(
            2.0,
        );
        let b = 2.0;
        
        let c = a.powf(b);
        assert_eq!(c.data(), 4.0);
    }

    #[test]
    fn test_value_negate() {
        let a = Value::from(
            3.0,
        );
        
        let c = -a;

        assert_eq!(c.data(), -3.0);
    }

    #[test]
    fn test_value_relu() {
        let a = Value::from(-3.0);
        let b = Value::from(4.0);

        let relu_a = a.relu();
        let relu_b = b.relu();

        assert_eq!(relu_a.data(), 0.0);
        assert_eq!(relu_b.data(), 4.0);
    }

    #[test]
    fn test_gradient_descent() {
        let a = Value::from(1.0);
        let b = Value::from(2.0);
        let c = Value::from(3.0);

        let mut y = a.clone() * b.clone() + c.clone();

        y.backward();

        assert_eq!(a.gradient(), 2.0);
        assert_eq!(b.gradient(), 1.0);
        assert_eq!(c.gradient(), 1.0);
    }

    #[test]
    fn test_gradient_descent_from_readme() {
        let a = Value::from(-4.0);
        let b = Value::from(2.0);
        let mut c = a.clone() + b.clone();
        let mut d = a.clone() * b.clone() + b.clone().powf(3.0);
        c = c.clone() + (c.clone() + 1.0);
        c = c.clone() + (1.0 + c.clone() + (-a.clone()));
        d = d.clone() + (d.clone() * 2.0 + (b.clone() + a.clone()).relu());
        d = d.clone() + (3.0 * d.clone() + (b.clone() - a.clone()).relu());
        let e = c.clone() - d.clone();
        let f = e.powf(2.0);
        let mut g = f.clone() / 2.0;
        g = g + (10.0 / f.clone());

        assert_eq!(g.data(), 24.70408163265306); // i.e. 24.7041: the outcome of this forward pass
        g.backward();
        assert_eq!(a.gradient(), 138.83381924198252); // i.e. 138.8338: the numerical value of dg/da
        assert_eq!(b.gradient(), 645.5772594752186); // i.e. 645.5773: the numerical value of dg/db       
    }
}