use rand::Rng;
use crate::engine::Value;

pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
    nonlinear: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlinear: bool, neuron_name: &str) -> Neuron {
        let mut rng = rand::thread_rng();

        let weights = (0..nin)
            .enumerate()
            .map(|(i, _)|
                Value::from(rng.gen_range(-1.0..1.0)).with_name(&format!("{}_weight_{}", neuron_name, i))
            )
            .collect();
        
        // It is possible and common to initialize the biases to be zero, since the asymmetry breaking
        // is provided by the small random numbers in the weights. For ReLU non-linearities, some people
        // like to use small constant value such as 0.01 for all biases because this ensures that all ReLU
        // units fire in the beginning and therefore obtain and propagate some gradient. However, it is not
        // clear if this provides a consistent improvement (in fact some results seem to indicate that this
        // performs worse) and it is more common to simply use 0 bias initialization.
        //
        // See: http://cs231n.github.io/neural-networks-2/
        let bias = Value::from(0.00).with_name(&format!("{}_bias", neuron_name));
        
        Self {
            weights,
            bias, 
            nonlinear
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Value {
        let act = &self.weights.iter().zip(x.iter())
            .map(|(wi, xi)| wi * xi)
            .sum::<Value>() + &self.bias;
            
        if self.nonlinear { act.relu() } else { act }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let parameters = [self.bias.clone()]
            .into_iter()
            .chain(self.weights.clone())
            .collect::<Vec<Value>>();

        parameters
    }

    pub fn zero_gradients(&mut self) {
        for p in self.parameters() {
            p.set_gradient(0.0);
        }
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlinear: bool, layer_index: usize) -> Layer {
        let neurons = (0..nout)
            .map(|neuron_index|
                Neuron::new(
                    nin,
                    nonlinear,
                    &format!("layer_{}_neuron_{}", layer_index, neuron_index)
                )
            )
            .collect();
        
        Layer {
            neurons
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        let neuron_outputs = self.neurons
            .iter()
            .map(|n| n.forward(&x))
            .collect();

        neuron_outputs
    }

    pub fn parameters(&self) -> Vec<Value> {
        let parameters = self.neurons
            .iter()
            .flat_map(|n| n.parameters())
            .collect();
    
        parameters
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: Vec<usize>) -> MLP {
        let sizes = [nin]
            .iter()
            .chain(nouts.iter())
            .copied()
            .collect::<Vec<_>>();

        let layers = sizes
            .windows(2)
            .enumerate()
            .map(|(layer_index, w)|
                Layer::new(
                    w[0],
                    w[1],
                    w[1] != *nouts.last().unwrap(),
                    layer_index
                )
            )
            .collect();
        
        MLP {
            layers
        }
    }

    pub fn forward(&self, x: &Vec<Value>) -> Vec<Value> {
        let mut output = x.to_vec();
        for layer in &self.layers {
            output = layer.forward(&output)
                .into_iter()
                .collect::<Vec<Value>>();
        }
        
        output
    }

    pub fn parameters(&self) -> Vec<Value> {
        let parameters = self.layers
            .iter()
            .flat_map(|l| l.parameters())
            .collect();

        parameters
    }

    pub fn zero_gradients(&mut self) {
        for p in self.parameters() {
            p.set_gradient(0.0);
        }
    }
}