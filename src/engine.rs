use std::sync::atomic::{AtomicUsize, Ordering};
use std::hash::Hash;
use std::cell::{Ref, RefCell};
use std::collections::HashSet;
use std::rc::Rc;
use std::iter::Sum;
use std::ops::{Add, Sub, Mul, Div, Neg, Deref};
use std::fmt;


static NEXT_VALUE_ID: AtomicUsize = AtomicUsize::new(0);


#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
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

type BackwardsPropagationFn = fn(internal_value_data: &Ref<InternalValueData>);

#[derive(Clone)]
pub struct InternalValueData {
    _id: usize,
    pub data: f64,
    pub gradient: f64,
    pub name: Option<String>,
    pub operation: Operation,
    _previous: Vec<Value>,
    _backward: Option<BackwardsPropagationFn>,
}

impl InternalValueData {
    fn new(
        data: f64,
        operation: Operation,
        previous: Vec<Value>,
        backward: Option<BackwardsPropagationFn>,
    ) -> InternalValueData {
        let id = NEXT_VALUE_ID.fetch_add(1, Ordering::Relaxed);
        InternalValueData {
            _id: id,
            data,
            gradient: 0.0,
            name: None,
            operation: operation,
            _previous: previous,
            _backward: backward,
        }
    }
}

impl PartialEq for InternalValueData {
    fn eq(&self, other: &Self) -> bool {
            self._id == other._id
            && self.name == other.name
            && self.operation == other.operation
            && self.data == other.data
            && self.gradient == other.gradient
            // Comparing `_previous` is too expensive and so we added
            // `_id` to faciliate cheaper comparisons.
            // && self._previous == other._previous
    }
}

impl Eq for InternalValueData {}

impl Hash for InternalValueData {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self._id.hash(state);
        self.name.hash(state);
        self.operation.hash(state);
        self.data.to_bits().hash(state);
        self.gradient.to_bits().hash(state);
        // Hashing `_previous` is too expensive and so we added
        // `_id` to faciliate cheaper hashes.
        // self._previous.hash(state);
    }
}

impl Default for InternalValueData {
    fn default() -> InternalValueData {
        let id = NEXT_VALUE_ID.fetch_add(1, Ordering::Relaxed);
        InternalValueData {
            _id: id,
            data: 0.0,
            gradient: 0.0,
            name: None,
            operation: Operation::Input,
            _previous: vec![],
            _backward: None
        }
    }
}


#[derive(Clone, Eq, PartialEq)]
pub struct Value(pub Rc<RefCell<InternalValueData>>);

impl Deref for Value {
    type Target = Rc<RefCell<InternalValueData>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().hash(state);
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

impl Value {
    pub fn from<T>(t: T) -> Value
    where T: Into<Value> {
        t.into()
    }

    pub fn new(value: InternalValueData) -> Value {
        Value(Rc::new(RefCell::new(value)))
    }

    pub fn id(&self) -> usize {
        self.borrow()._id
    }

    pub fn name(&self) -> Option<String> {
        self.borrow().name.clone()
    }

    pub fn set_name(&self, name: &str) {
        self.borrow_mut().name = Some(name.to_string());
    }

    pub fn with_name(&self, name: &str) -> Value {
        self.set_name(name);
        self.clone()
    }

    pub fn operation(&self) -> Operation {
        self.borrow().operation
    }

    pub fn data(&self) -> f64 {
        self.borrow().data
    }

    pub fn set_data(&self, data: f64) {
        self.borrow_mut().data = data;
    }

    pub fn decrement_data(&self, data: f64) {
        self.borrow_mut().data -= data;
    }

    pub fn gradient(&self) -> f64 {
        self.borrow().gradient
    }

    pub fn set_gradient(&self, gradient: f64) {
        self.borrow_mut().gradient = gradient;
    }

    pub fn previous(&self) -> Vec<Value> {
        self.borrow()._previous.clone()
    }

    pub fn backward(&mut self) {
        // Topological order all of the children in the graph.
        let mut topo = Vec::new();
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

        build_topo(
            self.clone(),
            &mut HashSet::new(),
            &mut topo
        );

        // Go one variable at a time and apply the chain rule to get its gradient.
        self.set_gradient(1.0);
        for v in topo.iter().rev() {
            let internal_value_data = v.borrow();

            if let Some(backward) = &internal_value_data._backward {
                backward(&internal_value_data);
            }
        }
    }

    pub fn pow(&self, other: &Value) -> Value {
        power(self, other)
    }

    pub fn powi(&self, otheri: i32) -> Value {
        power(
            self, 
            &Value::from(
                otheri as f64,
            )
        )
    }

    pub fn powf(&self, otherf: f64) -> Value {
        power(
            self, 
            &Value::from(
                otherf,
            )
        )
    }

    pub fn relu(&self) -> Value {
        let a = self.data();
        let c = if a > 0.0 { a } else { 0.0 };

        Value::new(
            InternalValueData::new(
                c,
                Operation::ReLU,
                vec![self.clone()],
                Some(|v| {
                    let mut s = v._previous[0].borrow_mut();
                    s.gradient += if v.data > 0.0 { v.gradient } else { 0.0 };
                })
            )
        )
    }
}

fn power(s: &Value, other: &Value) -> Value {
    let a = s.data();
    let b = other.data();
    let c = a.powf(b);

    Value::new(
        InternalValueData::new(
            c,
            Operation::Power,
            vec![s.clone(), other.clone()],
            Some(|v| {
                let mut s = v._previous[0].borrow_mut();
                let other = v._previous[1].borrow();
    
                s.gradient += other.data * s.data.powf(other.data - 1.0) * v.gradient;
            })
        )
    )
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

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(&self, &other, Operation::Add)
    }
}

impl<'b> Add<&'b Value> for Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Self::Output {
        add(&self, other, Operation::Add)
    }
}

impl<'a> Add<Value> for &'a Value {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(self, &other, Operation::Add)
    }
}

impl<'a, 'b> Add<&'b Value> for &'a Value {
    type Output = Value;

    fn add(self, other: &'b Value) -> Self::Output {
        add(self, other, Operation::Add)
    }
}

impl Add<Value> for f64 {
    type Output = Value;

    fn add(self, other: Value) -> Self::Output {
        add(
            &Value::from(self),
            &other,
            Operation::Add
        )
    }
}

impl Add<&Value> for f64 {
    type Output = Value;

    fn add(self, other: &Value) -> Self::Output {
        add(
            &Value::from(self),
            other,
            Operation::Add
        )
    }
}

impl Add<f64> for Value {
    type Output = Value;

    fn add(self, b: f64) -> Self::Output {
        add(
            &self,
            &Value::from(
                b,
            ),
            Operation::Add
        )
    }
}

impl Add<f64> for &Value {
    type Output = Value;

    fn add(self, b: f64) -> Self::Output {
        add(
            self,
            &Value::from(
                b,
            ),
            Operation::Add
        )
    }
}

fn add(s: &Value, other: &Value, operation: Operation) -> Value {
    let a = s.data();
    let b = other.data();
    let c = a + b;

    Value::new(
        InternalValueData::new(
            c,
            operation,
            vec![s.clone(), other.clone()],
            Some(|v| {
                let mut s = v._previous[0].borrow_mut();
                let mut other = v._previous[1].borrow_mut();
    
                s.gradient += 1.0 * v.gradient;
                other.gradient += 1.0 * v.gradient;
            })
        )
    )
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        subtract(&self, &other)
    }
}

impl<'b> Sub<&'b Value> for Value {
    type Output = Value;

    fn sub(self, other: &'b Value) -> Self::Output {
        subtract(&self, other)
    }
}

impl<'a> Sub<Value> for &'a Value {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        subtract(self, &other)
    }
}

impl<'a, 'b> Sub<&'b Value> for &'a Value {
    type Output = Value;

    fn sub(self, other: &'b Value) -> Self::Output {
        subtract(self, other)
    }
}

impl Sub<Value> for f64 {
    type Output = Value;

    fn sub(self, other: Value) -> Self::Output {
        subtract(
            &Value::from(self),
            &other
        )
    }
}

impl Sub<&Value> for f64 {
    type Output = Value;

    fn sub(self, other: &Value) -> Self::Output {
        subtract(
            &Value::from(self),
            other
        )
    }
}

impl Sub<f64> for Value {
    type Output = Value;

    fn sub(self, b: f64) -> Self::Output {
        subtract(
            &self,
            &Value::from(
                b,
            )
        )
    }
}

impl Sub<f64> for &Value {
    type Output = Value;

    fn sub(self, b: f64) -> Self::Output {
        subtract(
            self,
            &Value::from(
                b,
            )
        )
    }
}

fn subtract(s: &Value, other: &Value) -> Value {
    add(s, &negate(other), Operation::Subtract)
}

impl Mul<Value> for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        multiply(&self, &other, Operation::Multiply)
    }
}

impl<'b> Mul<&'b Value> for Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Self::Output {
        multiply(&self, other, Operation::Multiply)
    }
}

impl<'a> Mul<Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        multiply(self, &other, Operation::Multiply)
    }
}

impl<'a, 'b> Mul<&'b Value> for &'a Value {
    type Output = Value;

    fn mul(self, other: &'b Value) -> Self::Output {
        multiply(self, other, Operation::Multiply)
    }
}

impl Mul<Value> for f64 {
    type Output = Value;

    fn mul(self, other: Value) -> Self::Output {
        multiply(&Value::from(self), &other, Operation::Multiply)
    }
}

impl Mul<&Value> for f64 {
    type Output = Value;

    fn mul(self, other: &Value) -> Self::Output {
        multiply(&Value::from(self), other, Operation::Multiply)
    }
}

impl Mul<f64> for Value {
    type Output = Value;

    fn mul(self, b: f64) -> Self::Output {
        multiply(
            &self,
            &Value::from(
                b,
            ),
            Operation::Multiply
        )
    }
}

impl Mul<f64> for &Value {
    type Output = Value;

    fn mul(self, b: f64) -> Self::Output {
        multiply(
            self,
            &Value::from(
                b,
            ),
            Operation::Multiply
        )
    }
}

fn multiply(s: &Value, other: &Value, operation: Operation) -> Value {
    let a = s.data();
    let b = other.data();
    let c = a * b;

    Value::new(
        InternalValueData::new(
            c,
            operation,
            vec![s.clone(), other.clone()],
            Some(|v| {
                let mut s = v._previous[0].borrow_mut();
                let mut other = v._previous[1].borrow_mut();
    
                s.gradient += other.data * v.gradient;
                other.gradient += s.data * v.gradient;
            })
        )
    )
}

impl Div<Value> for Value {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        divide(&self, &other)
    }
}

impl<'b> Div<&'b Value> for Value {
    type Output = Value;

    fn div(self, other: &'b Value) -> Self::Output {
        divide(&self, other)
    }
}

impl<'a> Div<Value> for &'a Value {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        divide(self, &other)
    }
}

impl<'a, 'b> Div<&'b Value> for &'a Value {
    type Output = Value;

    fn div(self, other: &'b Value) -> Self::Output {
        divide(self, other)
    }
}

impl Div<Value> for f64 {
    type Output = Value;

    fn div(self, other: Value) -> Self::Output {
        divide(&Value::from(self), &other)
    }
}

impl Div<&Value> for f64 {
    type Output = Value;

    fn div(self, other: &Value) -> Self::Output {
        divide(&Value::from(self), other)
    }
}

impl Div<f64> for Value {
    type Output = Value;

    fn div(self, b: f64) -> Self::Output {
        divide(
            &self,
            &Value::from(
                b,
            )
        )
    }
}

impl Div<f64> for &Value {
    type Output = Value;

    fn div(self, b: f64) -> Self::Output {
        divide(
            self,
            &Value::from(
                b,
            )
        )
    }
}

fn divide(s: &Value, other: &Value) -> Value {
    multiply(s, &other.powf(-1.0), Operation::Divide)
}

impl Neg for Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        negate(&self)
    }
}

impl<'a> Neg for &'a Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        negate(self)
    }
}

fn negate(s: &Value) -> Value {
    multiply(
        s,
        &Value::from(
            -1.0,
        ),
        Operation::Negate
    )
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Value")
            .field("data", &self.data())
            .field("gradient", &self.gradient())
            .field("operation", &self.operation())
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
    fn test_backward_for_add() {
        let a = Value::from(
            1.0,
        );
        let b = Value::from(
            2.0,
        );
        
        let mut c = &a + &b;
        c.backward();

        assert_eq!(c.gradient(), 1.0);
        assert_eq!(a.gradient(), 1.0);
        assert_eq!(b.gradient(), 1.0);
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
    fn test_backward_for_sub() {
        let a = Value::from(
            15.0,
        );
        let b = Value::from(
            12.0,
        );
        
        let mut c = &a - &b;
        c.backward();

        assert_eq!(c.gradient(), 1.0);
        assert_eq!(a.gradient(), 1.0);
        assert_eq!(b.gradient(), -1.0);
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
    fn test_backward_for_multiply() {
        let a = Value::from(
            33.0,
        );
        let b = Value::from(
            3.0,
        );
        
        let mut c = &a * &b;
        c.backward();

        assert_eq!(c.gradient(), 1.0);
        assert_eq!(a.gradient(), 3.0);
        assert_eq!(b.gradient(), 33.0);
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
    fn test_backward_for_divide() {
        let a = Value::from(
            50.0,
        );
        let b = Value::from(
            2.0,
        );
        
        let mut c = &a / &b;
        c.backward();

        assert_eq!(c.gradient(), 1.0);
        assert_eq!(a.gradient(), 0.5);
        assert_eq!(b.gradient(), -12.5);
    }

    #[test]
    fn test_value_pow_value() {
        let a = Value::from(
            2.0,
        );
        let b = Value::from(
            2.0,
        );
        
        let c = a.pow(&b);
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
    fn test_backward_for_pow() {
        let a = Value::from(
            2.0,
        );
        let b = Value::from(
            2.0,
        );
        
        let mut c = a.pow(&b);
        c.backward();

        assert_eq!(c.gradient(), 1.0);
        assert_eq!(a.gradient(), 4.0);
        assert_eq!(b.gradient(), 0.0);
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
    fn test_backward_for_negate() {
        let a = Value::from(
            3.0,
        );
        
        let mut c = -(&a);
        c.backward();

        assert_eq!(c.gradient(), 1.0);
        assert_eq!(a.gradient(), -1.0);
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
    fn test_backward_for_relu() {
        let a = Value::from(-3.0);
        let b = Value::from(4.0);
        
        let mut relu_a = a.relu();
        let mut relu_b = b.relu();

        relu_a.backward();
        relu_b.backward();

        assert_eq!(relu_a.gradient(), 1.0);
        assert_eq!(a.gradient(), 0.0);
        assert_eq!(relu_b.gradient(), 1.0);
        assert_eq!(b.gradient(), 1.0);
    }

    #[test]
    fn test_gradient_descent() {
        let a = Value::from(1.0);
        let b = Value::from(2.0);
        let c = Value::from(3.0);

        let mut y = &(&a * &b) + &c;

        y.backward();

        assert_eq!(y.gradient(), 1.0);
        assert_eq!(a.gradient(), 2.0);
        assert_eq!(b.gradient(), 1.0);
        assert_eq!(c.gradient(), 1.0);
    }

    #[test]
    fn test_gradient_descent_from_readme() {
        let a = Value::from(-4.0);
        let b = Value::from(2.0);
        let mut c = &a + &b;
        let mut d = &a * &b + &b.powf(3.0);
        c = &c + (&c + 1.0);
        c = &c + (1.0 + &c + -&a);
        d = &d + (&d * 2.0 + (&b + &a).relu());
        d = &d + (3.0 * &d + (&b - &a).relu());
        let e = &c - &d;
        let f = e.powf(2.0);
        let mut g = &f / 2.0;
        g = &g + (10.0 / &f);

        assert_eq!(g.data(), 24.70408163265306); // i.e. 24.7041: the outcome of this forward pass
        g.backward();
        assert_eq!(a.gradient(), 138.83381924198252); // i.e. 138.8338: the numerical value of dg/da
        assert_eq!(b.gradient(), 645.5772594752186); // i.e. 645.5773: the numerical value of dg/db       
    }
}