use rand::Rng;
use crate::engine::Value;

pub struct Neuron {
    weight: Vec<Value>,
    bias: Value,
    nonlinear: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlinear: bool) -> Self {
        let mut rng = rand::thread_rng();
        let weight = (0..nin).map(|_| Value::with_data(rng.gen_range(-1.0..=1.0))).collect();
        let bias = Value::with_data(0.0);
        
        Self {
            weight,
            bias, 
            nonlinear
        }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        let act = self.weight.iter().zip(x.iter())
            .map(|(wi, xi)| wi.clone() * xi.clone())
            .sum::<Value>() + self.bias.clone();
            
        if self.nonlinear { act.relu() } else { act }
    }

    pub fn parameters(&self) -> Vec<&Value> {
        let mut params = Vec::with_capacity(self.weight.len() + 1);
        params.extend(self.weight.iter());
        params.push(&self.bias);
        params
    }

    pub fn zero_out_gradients(&mut self) {
        for p in self.parameters() {
            p.set_gradient(0.0);
        }
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlinear: bool) -> Self {
        let neurons = (0..nout).map(|_| Neuron::new(nin, nonlinear)).collect();
        
        Self {
            neurons
        }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(&x)).collect()
    }

    pub fn parameters(&self) -> Vec<&Value> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }

    pub fn zero_out_gradients(&mut self) {
        for p in self.parameters() {
            p.set_gradient(0.0);
        }
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> Self {
        let sz = [nin].iter().chain(nouts.iter()).copied().collect::<Vec<_>>();
        let layers = sz.windows(2).map(|w| Layer::new(w[0], w[1], w[1] != *nouts.last().unwrap())).collect();
        
        Self {
            layers
        }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut output = x.to_vec();
        for layer in &self.layers {
            let forwarded_output = layer.forward(&output);
            output = forwarded_output.into_iter().collect::<Vec<Value>>();
        }
        
        output
    }

    pub fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_out_gradients(&mut self) {
        for p in self.parameters() {
            p.set_gradient(0.0);
        }
    }
}