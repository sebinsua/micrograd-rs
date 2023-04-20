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
    Multiply,
    Power,
    ReLU,
    // The `Negate`, `Divide`, and `Subtract` operations are implemented using the operations defined above,
    // which means that when visualizing the computation graph the inputs to these operations will not make
    // sense unless you check the implementation of the operation.
    Negate,
    Divide,
    Subtract,
}

type BackwardsPropagationFn = fn(internal_value_data: &Ref<InternalValueData>);

#[derive(Clone)]
pub struct InternalValueData {
    // `_id`, `_name`, and `_operation` are used to facilitate debugging, to generate unique names for each value
    // when visualizing the computation graph, and to facilitate cheaper comparisons and hashing.
    _id: usize,
    _name: Option<String>,
    _operation: Operation,
    pub data: f64,
    pub gradient: f64,
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
            _name: None,
            _operation: operation,
            data,
            gradient: 0.0,
            _previous: previous,
            _backward: backward,
        }
    }
}

impl PartialEq for InternalValueData {
    fn eq(&self, other: &Self) -> bool {
            self._id == other._id
            && self._name == other._name
            && self._operation == other._operation
            && self.data == other.data
            && self.gradient == other.gradient
            // Comparing `_previous` is too expensive and so we added
            // `_id` to facilitate cheaper comparisons.
            // && self._previous == other._previous
    }
}

impl Eq for InternalValueData {}

impl Hash for InternalValueData {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self._id.hash(state);
        self._name.hash(state);
        self._operation.hash(state);
        self.data.to_bits().hash(state);
        self.gradient.to_bits().hash(state);
        // Hashing `_previous` is too expensive and so we added
        // `_id` to facilitate cheaper hashes.
        // self._previous.hash(state);
    }
}

impl Default for InternalValueData {
    fn default() -> InternalValueData {
        let id = NEXT_VALUE_ID.fetch_add(1, Ordering::Relaxed);
        InternalValueData {
            _id: id,
            _name: None,
            _operation: Operation::Input,
            data: 0.0,
            gradient: 0.0,
            _previous: vec![],
            _backward: None,
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
        self.borrow()._name.clone()
    }

    pub fn set_name(&self, name: &str) {
        self.borrow_mut()._name = Some(name.to_string());
    }

    pub fn with_name(&self, name: &str) -> Value {
        self.set_name(name);
        self.clone()
    }

    pub fn operation(&self) -> Operation {
        self.borrow()._operation
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

    // The `backward` method performs the backward pass of the backpropagation algorithm on a
    // computational graph, computing gradients for each `Value`. Backpropagation
    // is commonly used in training machine learning models, particularly neural networks,
    // and relies on the chain rule of calculus for efficient gradient computation.
    pub fn backward(&mut self) {
        // The `sort_topologically` function is a recursive helper function used for performing 
        // a topological sort on the graph. It traverses the graph in a depth-first manner, adding
        // values to the `topologically_sorted_values` vector after each all their `_previous` values
        // have been visited.
        fn sort_topologically(
            value: Value,
            visited: &mut HashSet<usize>,
            topologically_sorted_values: &mut Vec<Value>
        ) {
            let id = value.as_ptr() as usize;
            if !visited.contains(&id) {
                visited.insert(id);

                for child in value.previous() {
                    sort_topologically(
                        child,
                        visited,
                        topologically_sorted_values
                    );
                }

                topologically_sorted_values.push(value);
            }
        }

        // Perform a topological sort of all of the `_previous` values in the graph so that the
        // output values are after their respective input values.
        let mut topologically_sorted_values = Vec::new();
        sort_topologically(
            self.clone(),
            &mut HashSet::new(),
            &mut topologically_sorted_values
        );

        // Reverse the topologically sorted values so that we can start at the output value and
        // propagate the gradients backwards through the graph towards the input values, ensuring
        // the correct order for the backpropagation process.
        topologically_sorted_values = topologically_sorted_values.into_iter().rev().collect();

        // The derivative of a value with respect to itself is always 1.0 so we set the gradient
        // of the output value to this to begin with before beginning backwards propagation.
        topologically_sorted_values[0].set_gradient(1.0);

        // Given the reversed topologically ordered values, we will be starting at the output value
        // and applying the chain rule on each iteration to update the gradients of the current value's 
        // previous values.
        // 
        // The gradient of each value is a scalar float representing the partial derivative of the output
        // with respect to the input.
        for value in topologically_sorted_values.iter() {
            let internal_value_data = value.borrow();

            if let Some(update_gradients_of_previous_inputs) = &internal_value_data._backward {
                update_gradients_of_previous_inputs(&internal_value_data);
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
        relu(self)
    }
}

// The `relu` function applies the Rectified Linear Unit (ReLU) activation function on an input `Value`.
// ReLU is defined as: f(x) = max(0, x), where x is the input.
//
// The derivative of ReLU with respect to its input (`input_a`) is:
//
// ∂current_value/∂input_a (the local gradient) = 1.0, if input_a > 0
//                                              = 0.0, otherwise
//
// The gradient accumulation update rule for the input value (`input_a.gradient`) is:
//
// ∂L/∂input_a += (∂current_value/∂input_a) * (∂L/∂current_value)
//
// Where L is the final output value (e.g. the loss function).
fn relu(input: &Value) -> Value {
    let a = input.data();
    let c = if a > 0.0 { a } else { 0.0 };

    Value::new(
        InternalValueData::new(
            c,
            Operation::ReLU,
            vec![input.clone()],
            Some(|current_value| {
                let mut input_a = current_value._previous[0].borrow_mut();
                input_a.gradient += if current_value.data > 0.0 { 1.0 * current_value.gradient } else { 0.0 * current_value.gradient };
            })
        )
    )
}

// The `power` function computes the exponentiation of `input_a` raised to the power of `input_b`.
//
// The derivative of the output (`current_value`) with respect to its input (`input_a`) is:
//
// ∂current_value/∂input_a (the local gradient) = input_b * input_a^(input_b - 1)
// 
// The gradient accumulation update rule for the input value (`input_a.gradient`) is:
//
// (∂L/input_a) += (∂current_value/∂input_a) * (∂L/∂current_value)
//
// Where L is the final output value (e.g. the loss function).
fn power(input_a: &Value, input_b: &Value) -> Value {
    let a = input_a.data();
    let b = input_b.data();
    let c = a.powf(b);

    Value::new(
        InternalValueData::new(
            c,
            Operation::Power,
            vec![input_a.clone(), input_b.clone()],
            Some(|current_value| {
                let mut input_a = current_value._previous[0].borrow_mut();
                let input_b = current_value._previous[1].borrow();
    
                input_a.gradient += input_b.data * input_a.data.powf(input_b.data - 1.0) * current_value.gradient;
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

    fn add(self, otherf: f64) -> Self::Output {
        add(
            &self,
            &Value::from(
                otherf,
            ),
            Operation::Add
        )
    }
}

impl Add<f64> for &Value {
    type Output = Value;

    fn add(self, otherf: f64) -> Self::Output {
        add(
            self,
            &Value::from(
                otherf,
            ),
            Operation::Add
        )
    }
}

// The `add` function computes the addition of `input_a` and `input_b`.
//
// The derivatives of the output (`current_value`) with respect to its inputs (`input_a` and `input_b`) are:
//
// ∂current_value/∂input_a (the local gradient) = 1
// ∂current_value/∂input_b (the local gradient) = 1
//
// The gradient accumulation rules for the input values (`input_a.gradient` and `input_b.gradient`) are:
//
// (∂L/input_a) += (∂current_value/∂input_a) * (∂L/∂current_value)
// (∂L/input_b) += (∂current_value/∂input_b) * (∂L/∂current_value)
//
// Where L is the final output value (e.g. the loss function).
fn add(input_a: &Value, input_b: &Value, operation: Operation) -> Value {
    let a = input_a.data();
    let b = input_b.data();
    let c = a + b;

    Value::new(
        InternalValueData::new(
            c,
            operation,
            vec![input_a.clone(), input_b.clone()],
            Some(|current_value| {
                let mut input_a = current_value._previous[0].borrow_mut();
                let mut input_b = current_value._previous[1].borrow_mut();
    
                input_a.gradient += 1.0 * current_value.gradient;
                input_b.gradient += 1.0 * current_value.gradient;
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

    fn sub(self, otherf: f64) -> Self::Output {
        subtract(
            &self,
            &Value::from(
                otherf,
            )
        )
    }
}

impl Sub<f64> for &Value {
    type Output = Value;

    fn sub(self, otherf: f64) -> Self::Output {
        subtract(
            self,
            &Value::from(
                otherf,
            )
        )
    }
}

// The `subtract` function computes the subtraction of `input_a` and `input_b`
// but does not implement its own gradient accumulation update rules as it relies on granular
// `add` and `negate` operations (the latter itself relying on `multiply`).
fn subtract(input_a: &Value, input_b: &Value) -> Value {
    add(input_a, &negate(input_b), Operation::Subtract)
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

    fn mul(self, otherf: f64) -> Self::Output {
        multiply(
            &self,
            &Value::from(
                otherf,
            ),
            Operation::Multiply
        )
    }
}

impl Mul<f64> for &Value {
    type Output = Value;

    fn mul(self, otherf: f64) -> Self::Output {
        multiply(
            self,
            &Value::from(
                otherf,
            ),
            Operation::Multiply
        )
    }
}

// The `multiply` function computes the multiplication of `input_a` and `input_b`.
//
// The derivatives of the output (`current_value`) with respect to its inputs (`input_a` and `input_b`) are:
//
// ∂current_value/∂input_a (the local gradient) = input_b
// ∂current_value/∂input_b (the local gradient) = input_a
//
// The gradient accumulation rules for the input values (`input_a.gradient` and `input_b.gradient`) are:
//
// (∂L/input_a) += (∂current_value/∂input_a) * (∂L/∂current_value)
// (∂L/input_b) += (∂current_value/∂input_b) * (∂L/∂current_value)
//
// Where L is the final output value (e.g. the loss function).
fn multiply(input_a: &Value, input_b: &Value, operation: Operation) -> Value {
    let a = input_a.data();
    let b = input_b.data();
    let c = a * b;

    Value::new(
        InternalValueData::new(
            c,
            operation,
            vec![input_a.clone(), input_b.clone()],
            Some(|current_value| {
                let mut input_a = current_value._previous[0].borrow_mut();
                let mut input_b = current_value._previous[1].borrow_mut();
    
                input_a.gradient += input_b.data * current_value.gradient;
                input_b.gradient += input_a.data * current_value.gradient;
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

// The `divide` function computes the division of `input_a` and `input_b`
// but does not implement its own gradient accumulation update rules as it is implemented
// using the `multiply` and `powf` operations (the latter itself relying on `power`).
fn divide(input_a: &Value, input_b: &Value) -> Value {
    multiply(input_a, &input_b.powf(-1.0), Operation::Divide)
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

// The `negate` function computes the negation of `input` but does not implement
// its own gradient accumulation update rule as it is implemented using the `multiply` operation.
fn negate(input: &Value) -> Value {
    multiply(
        input,
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
            .field("_name", &self.name())
            .field("_operation", &self.operation())
            .field("_previous", &self.previous().iter().map(|x| format!("data={}, gradient={}", x.data(), x.gradient())).collect::<Vec<String>>())
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
        assert_eq!(a.gradient(), 138.83381924198252); // i.e. 138.8338: the numerical value of ∂g/∂a
        assert_eq!(b.gradient(), 645.5772594752186); // i.e. 645.5773: the numerical value of ∂g/∂b       
    }
}