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

        // If the neurons in a layer have the same weights, then during backpropagation they will produce the
        // same outputs and gradients. In a neural network, you want the neurons to explore different aspects
        // of the data during training and to learn distinct features, therefore they must not behave identically.
        //
        // Therefore, we randomly intitialize `weight`s with values between -1.0 and 1.0, in order to break symmetry
        // and ensure that each neuron computes a distinct output. This ultimately makes the neural network more
        // effective at learning, and also helping to avoid local minima.
        let weights = (0..nin)
            .enumerate()
            .map(|(i, _)|
                // There are other forms of random weight initialization that can be used, such as Xavier, He, or
                // LeCun initialization. These take into account the activation function and network architecture
                // which can lead to faster convergence and improved performance. However, for the purposes of this
                // example, we will use a simple random initialization.
                Value::from(rng.gen_range(-1.00..1.00)).with_name(&format!("{}_weight_{}", neuron_name, i))
            )
            .collect();
        
        // The `bias` is initialized to zero.
        //
        // > It is possible and common to initialize the biases to be zero, since the asymmetry breaking
        // > is provided by the small random numbers in the weights. For ReLU non-linearities, some people
        // > like to use small constant value such as 0.01 for all biases because this ensures that all ReLU
        // > units fire in the beginning and therefore obtain and propagate some gradient. However, it is not
        // > clear if this provides a consistent improvement (in fact some results seem to indicate that this
        // > performs worse) and it is more common to simply use 0 bias initialization.
        //
        // See: http://cs231n.github.io/neural-networks-2/
        let bias = Value::from(0.00).with_name(&format!("{}_bias", neuron_name));
        
        Self {
            weights,
            bias, 
            nonlinear
        }
    }

    // In a multilayer perceptron (MLP), the output of each neuron is computed as a weighted sum of the inputs plus a bias.
    // This value is then passed through a non-linear activation function, in our case a rectified linear unit (ReLU).
    //
    // Note: The `forward` method below does not return a single float value as the output of a `Neuron` but instead a `Value`.
    //       This is due to how this particular implementation of a neural network implements backwards propagation.
    //       It is highly generic and the mathematical expressions below create a computational graph, which is basically a 
    //       tree structure representing (a) the granular operations that are performed on its inputs, weights and biases
    //       and the output of each of these, and (b) any metadata required to later on compute the gradients during the 
    //       backwards pass.
    pub fn forward(&self, x: &Vec<Value>) -> Value {
        // Every value in `x` is matched up with a corresponding `weight` and multiplied together before being summed.
        //
        // The intuition behind using a weighted sum is inspired by the structure and function of biological neurons. 
        // In a biological neuron, multiple inputs are received through its dendrites, and these inputs are combined,
        // weighted by the strength of their connections (synapses), to determine if the neuron should fire an output
        // signal through its axon.
        // 
        // Our artificial neuron is mimicing this behaviour as the weights that are learnt while training represent the
        // importance of each input.
        let weighted_sum = self.weights
            .iter()
            .zip(x.iter())
            .map(|(wi, xi)| wi * xi)
            .sum::<Value>();
        
        // The addition of a `bias` term allows a neuron to learn a function that does not always pass through the origin (0, 0).
        // It is similar to the intercept term in linear regression, and allows the regression line to be shifted up or down along
        // the y-axis. This allows the neuron to fire even when all of its inputs are zero, and also helps to break symmetry improving
        // the performance of the network.
        let preactivation = &weighted_sum + &self.bias;
        
        // ReLU transforms any negative inputs into zeroes and this ensures that our network can learn non-linear relationships.
        // Without this non-linearity, our network would only be able to learn linear relationships, no matter how many layers
        // or neurons we add.
        if self.nonlinear { preactivation.relu() } else { preactivation }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let parameters = [self.bias.clone()]
            .into_iter()
            .chain(self.weights.clone())
            .collect::<Vec<Value>>();

        parameters
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
    // A multi-layer perceptron (MLP) is a feed-forward neural network, in which
    // each neuron in a layer is connected to every neuron in the previous and next
    // layers.
    pub fn new(nin: usize, nouts: Vec<usize>) -> MLP {
        let sizes = [nin]
            .iter()
            .chain(nouts.iter())
            .copied()
            .collect::<Vec<_>>();

        // We take the `sizes` of the input, hidden and output layers and create a list of window
        // tuples that represent the input and output sizes of each layer.
        // 
        // e.g. [5, 10, 15, 10, 20] -> [(5, 10), (10, 15), (15, 10), (10, 20)]
        let layers = sizes
            .windows(2)
            .enumerate()
            .map(|(layer_index, layer_size_tuple)|
                Layer::new(
                    layer_size_tuple[0],
                    layer_size_tuple[1],
                    // We do not apply any non-linearity producing activation functions to the
                    // output layer, because the problem we are solving is a regression problem
                    // and we want our network to be able to predict continuous values less than zero.
                    // 
                    // A different choice might have been made if we were solving a classification
                    // problem with multiple classes, as then we might have used a softmax activation
                    // function on the output layer. Similarly, if we were solving a binary classification
                    // problem, we might have used a sigmoid activation function on the output layer. 
                    layer_size_tuple[1] != *nouts.last().unwrap(),
                    layer_index
                )
            )
            .collect();
        
        MLP {
            layers
        }
    }

    // The `forward` method in a neural network computes predictions and therefore has dual uses.
    // It is used (a) during training to generate predictions that are used by the loss function
    // during backpropagation to help update the weights and biases, and then (b) after training 
    // to make predictions.
    //
    // It implements forward propagation by passing the input vector `x` through each successive layer 
    // in the network, with each layer taking the output of the previous layer as its input and computing 
    // each of its neuron's outputs from the same input vector.
    // 
    // Eventually, the output vector of the final hidden layer is passed into the output layer and the output
    // of this is the network's prediction.
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

    // We must reset the gradients of all weights and biases in the network before each training iteration,
    // because otherwise the gradients would accumulate from previous iterations, leading to erratic updates
    // and poor performance.
    pub fn zero_gradients(&mut self) {
        for p in self.parameters() {
            p.set_gradient(0.0);
        }
    }
}