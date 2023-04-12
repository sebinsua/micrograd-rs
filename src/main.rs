use std::{env, vec, fs};
use rand::Rng;
use itertools::izip;
use micrograd_rs::{Value, MLP, create_graph};


fn get_graphviz_output_arg() -> bool {
    let mut graphviz_output_arg = false;

    let mut args = env::args().skip(1).collect::<Vec<_>>();
    while let Some(arg) = args.pop() {
        match arg.as_str() {
            "--graphviz-output" => {
                graphviz_output_arg = true;
            },
            _ => {
                panic!("Unknown argument: {}", arg);
            }
        }
    }

    graphviz_output_arg
}

// Generate `x_data` for a given range and size.
// Returns a vector of one-element vectors, as the `forward` method of `MLP` expects a `&Vec<Value>` input.
fn generate_x_data(start: f64, end: f64, size: usize) -> Vec<Vec<Value>> {
    (0..size)
        .map(|i| vec![Value::from(start + (i as f64) * (end - start) / ((size - 1) as f64))])
        .collect()
}

// Generate output values based on the linear function `y = 2x + 3` and optionally
// add some noise to the output.
fn generate_y_value(x: &Value, rng: &mut rand::rngs::ThreadRng, with_noise: bool) -> Value {
    let noise = if with_noise { rng.gen_range(-0.1..=0.1) } else { 0.0 };
    Value::from(2.0 * x.data() + 3.0 + noise)
}

// Compute the mean squared error (MSE) between predicted and actual values,
// by zipping them together and squaring the difference.
//
// The benefits of this approach are:
//
// - Positivity: Ensures all error values are positive.
// - Penalizes large errors: Emphasizes reducing larger errors.
// - Differentiable: Suitable for optimization algorithms.
// - Convexity: Guarantees a single global minimum with no local minima, simplifying optimization and ensuring convergence
//              to the global minimum for certain models (e.g., single-layer models). Note that the convexity of the 
//              optimization problem depends on the model's architecture itself.
fn mse(predictions: &Vec<Vec<Value>>, actuals: &Vec<Value>) -> Value {
    predictions
        .iter()
        .zip(actuals.iter())
        .map(|(predicted, actual)|
            (&predicted[0] - actual).powi(2)
        )
        .sum::<Value>() / Value::from(actuals.len() as f64)
}

fn main() {
    let should_output_graphviz = get_graphviz_output_arg();

    ///////////////////////
    // Training          //
    ///////////////////////

    let mut rng = rand::thread_rng();

    // Generate synthetic data for training.
    let x_training_data = generate_x_data(-2.0, 2.0, 20 as usize)
        .iter()
        .enumerate()
        .map(|(i, x)|
            vec![x[0].with_name(&format!("x_training_{}", i))]
        )
        .collect::<Vec<Vec<Value>>>();

    let y_training_data: Vec<Value> = x_training_data
        .iter()
        .enumerate()
        .map(|(i, x)|
            generate_y_value(&x[0], &mut rng, true).with_name(&format!("y_training_{}", i))
        )
        .collect();

    // Print training data.
    for (x, y) in x_training_data.iter().zip(y_training_data.iter()) {
        println!("x: {:.2}, y: {:.2}", x[0].data(), y.data());
    }

    // Define a mean squared error (MSE) loss function.
    fn loss(x_data: &Vec<Vec<Value>>, y_data: &Vec<Value>, model: &MLP) -> Value {
        // The "forward pass" computes predicted `y` values for the given `x` values.
        //
        // While doing so, it extends the computation graph of the `model` with new `Value` nodes.
        // Note that, the connection between the result of the `loss` function andn the `model` is one-way;
        // the `model` is not connected to the `loss` function and `loss` does not participate in the computation
        // when `forward` is called on the `model`.
        let y_predictions = x_data
            .iter()
            .enumerate()
            .map(|(i, x)|
                model.forward(&x)
                    .iter()
                    .enumerate()
                    .map(|(j, y)|
                        y.with_name(&format!("y_prediction_{}_{}", i, j))
                    )
                    .collect::<Vec<_>>()
            )
            .collect::<Vec<_>>();
        
        mse(&y_predictions, &y_data).with_name("mse_loss")
    }

    // These are the hyperparameters for training:
    //
    // `epochs` is the number of training iterations.
    // `learning_rate` is the step size for the gradient descent algorithm.
    //
    // The function we are trying to learn is `y = 2x + 3` which is a simple linear function.
    // Therefore, we do not need a large number of epochs or a particularly small learning rate
    // in order to achieve good results. However, if we were attempting to learn a more complex
    // function, we might need to increase the number of epochs, decrease the learning rate,
    // or add more neurons or hidden layers to the model.
    //
    // There are also more advanced techniques that we can use to optimize the training process
    // such as mini-batch gradient descent, adaptive learning rate algorithms (e.g. Adam, RMSprop),
    // regularization (e.g. L1 or L2 also known as "weight decay"), momentum, early stopping and
    // learning rate scheduling, among others.
    let epochs = 1000;
    let learning_rate = 0.01;

    // Create a multi-layer perceptron (MLP) model with one input neuron in the input layer,
    // two hidden layers with five neurons each, and one output neuron in the output layer.
    let mut model = MLP::new(1, vec![5, 5, 1]);

    // A neural network is proven to be a universal function approximator, which is is a system 
    // capable of mimicking any continuous function's behavior within a specified domain, given
    // appropriate configuration and capacity.
    //
    // As a programmer, we are used to constructing functions by arranging syntaxes to express our
    // logical intent -- e.g. `if n <= 1 { n } else { fn(n-1) + fn(n-2) }`. This naturally leads
    // to functions that are easy for humans to understand and interpret, because (1) they are written
    // in a way that is similar to how we would explain the function to another human, (2) related logic
    // is generally factored so as to be colocated and DRY, and finally (3) we have a full range of
    // high-level programming syntaxes and library calls available to us instead of only a limited set
    // of mathematical operations.
    //
    // Unfortunately the functions represented by neural networks are not naturally human interpretable
    // because of how neural networks are constructed. A neural network is basically a gigantic mathematical
    // expression made up of layers of weighted sums of inputs that are each passed through non-linear
    // activation functions (e.g. `activation(sum(weights * inputs) + bias)`) before being passed
    // through to the same expressions in another layer.
    //
    // The reasons why modern neural networks are constructed as giant mathematical expressions, isn't merely
    // because this makes it computationally efficient to train large expressive models, but is substantially 
    // due to training them requiring a process called "backpropagation" (also known as reverse-mode automatic
    // differentiation). 
    //
    // In order for backpropagation to be possible, it requires the following:
    //
    // 1. All operations/functions used by a neural network to produce its output *must be* differentiable. The
    //    reason for this is that backpropagation uses the chain rule of differentiation to compute each input 
    //    `gradient` (the derivative of a loss function with respect to an input weight or bias) by multiplying
    //    the local derivative of the "current weight or bias with respect to its input weight or bias" by the partial 
    //    derivative of "the loss function with respect to the current weight or bias" before accumulating this.
    //
    //      e.g.
    //      
    //        ∂L/∂input_weight_or_bias += ∂current_weight_or_bias/∂input_weight_or_bias * ∂L/∂current_weight_or_bias
    //
    //      Note 1: The `∂` symbol is the partial derivative operator.
    //      Note 2: In the example above `∂L/∂current_weight_or_bias` would have been computed by a prior iteration in the
    //              backpropagation algorithm and therefore can be substituted with the `gradient` of the current weight or
    //              bias.
    //      Note 3: On the other hand, `∂current_weight_or_bias/∂input_weight_or_bias` is the local derivative and must be 
    //              computed based on the type of operation/function and its input values. Within our neural network, this
    //              is done within the `_backward` method of each `Value` struct.
    //      Note 4: Mathematical operators like `*` and `+` are trivially differentiable. A function is differentiable if it
    //              is continuous and has a derivative at every point in its domain. Discontinuities in a function can make 
    //              it non-differentiable at those specific points. For example, `ReLU(x) = max(0, x)` is discontinuous at
    //              `x = 0` and therefore is not differentiable at that point, however, it is still differentiable otherwise 
    //              (e.g. `ReLU'(x) = 0, x < 0` or `ReLU'(x) = 1, x > 0`). In practice, `x = 0` is very rare and we can 
    //              safely set the subderivative to 0 at that point.
    //      Note 5: We accumulate the result of multiplying these two partial derivatives into `∂L/∂input_weight_or_bias` 
    //              which means that multiple output values of the network could contribute to the gradient of a single input 
    //              weight or bias. Only once every function/operation that an input weight or bias is involved in has been 
    //              processed will the `∂L/∂input_weight_or_bias` have been computed and be ready for use as a 
    //              `∂L/∂current_weight_or_bias` in a future iteration of the backpropagation algorithm. Within our neural 
    //              network, a topological sort is used to ensure that this is the case.
    // 
    // 2. Any logic or understanding learnt will be generically represented by the network's weights and biases (its parameters)
    //    and therefore because these participate in the calculations of derivatives they *must be* real-valued numeric values.
    //    Basically, if you want a universal function approximator that can learn any function, you must first start with a
    //    generic function and then learn the parameters that make it behave like the function you'd like to approximate.
    //
    // 3. Mathematical expressions and parameter initializations are carefully designed to avoid issues such as
    //    "symmetry" (neurons that produce the same outputs), "dead neurons" (neurons that always output zero),
    //    "exploding gradients" (gradients that grow exponentially in magnitude), and "vanishing gradients" 
    //    (gradients that shrink exponentially in magnitude), as well as other numerical stability issues.
    //
    // The key idea is that as long as there is a way to compute or approximate the local derivative of every function/operation,
    // we can use this to help compute the derivative of the loss function with respect to an input weight or bias and then to
    // store a `gradient` for each weight and bias in the network. This would be incredibly useful to us as it would allow us
    // to determine the impact of each weight and bias on the overall model outputs.
    // 
    // That brings us to the second key idea of neural networks. It's not enough to merely have a function that can be used to 
    // "predict" values by multipling weights and biases by inputs while passing their results through activation functions. 
    // Even if we had a way to compute gradients of these outputs with respect to their weights and biases, it would still tell
    // us nothing about how to improve the performance of the network. What we need is a way to measure how well the network is
    // performing and a method of using this information to update weights and biases. 
    //
    // That is where the "loss" function comes in. The loss function (sometimes known as a cost function or error function) is a
    // function that compares the predicted value produced by the model with the actual value that we want the model to produce.
    // It provides both a performance metric and an optimization objective, with the goal of minimizing the loss function during
    // training to improve the network's performance -- the lower the loss, the less information the model loses and the better 
    // it performs; the higher the loss, the worse the model performs. Once the gigantic mathematical expression that is your 
    // neural network is producing this value, backpropagation can be used to compute the derivative of the loss function with 
    // respect to each weight or bias in the network (the `gradient`). It's important to note that the `gradient` of a weight or
    // bias is not the same as the weight or bias itself. The `gradient` is the name given to the derivative ("rate of change") 
    // of the loss function with respect to the weight or bias and represents the impact of a small change in the weight or bias 
    // on the loss function. This `gradient` can then be used in a process called "gradient descent" to update the weight or bias
    // in a way that reduces the total loss of the network -- e.g. if the `gradient` of a weight is positive, then the weight 
    // should be decreased, while if the `gradient` of a weight is negative, then the weight should be increased; similarly if the
    // `gradient` is large, then the weight should be updated by a large amount, while if the `gradient` is small, then the weight
    // should be updated by a small amount.
    //
    // The process described above is repeated for each "epoch" (iteration) of the training loop, and the magnitude of these updates
    // to the weights and biases are also controlled by a "learning rate". Both the learning rate and the number of epochs are
    // hyperparameters that can be tuned to improve the performance of the network, alongside other aspects of the network such as
    // the number of layers, the number of neurons in each layer, the activation function used in each layer, amongst other things.
    // 
    // ...
    //
    // Training loop.
    for epoch in 0..epochs {
        // Forward pass: compute the loss for the current model's parameters (weights and biases).
        let mut total_loss = loss(&x_training_data, &y_training_data, &model);

        // Zero out all gradients in the computation graph before starting backpropagation to avoid
        // accumulating gradients from previous iterations, which would result in erratic parameter updates.
        model.zero_gradients();

        // Backward pass: compute the gradients of the loss with respect to each of the the model's parameters using backpropagation.
        total_loss.backward();

        // Update model parameters using stochastic gradient descent (SGD).
        // 
        // Gradient descent is an optimization algorithm that is used to find the minimum of a function. It does this by taking steps
        // in the opposite direction of the gradient (derivative) of the function at any given point. For non-convex functions, it is
        // important to note that we cannot guarantee that we will find the global minimum of a function or that we will find a minimum
        // at all. It is possible that we might get stuck in a saddle point, plateau, or local minima, or that we might oscillate around
        // the minimum or even diverge away from the it entirely instead of converging towards it.
        for p in model.parameters() {
            // We are attempting to minimize the loss, so we subtract the gradient from the parameter.
            // However, a negative gradient would result in an increase to a parameter.
            p.decrement_data(learning_rate * p.gradient());
        }

        if epoch % 10 == 0 {
            println!("Epoch: {} / Loss: {}", epoch, total_loss.data());
        }
    }

    ///////////////////////
    // Testing           //
    ///////////////////////

    // Generate synthetic data for testing of the model.
    let x_testing_data = generate_x_data(-3.0, 3.0, 10 as usize)
        .iter()
        .enumerate()
        .map(|(i, x)|
            vec![x[0].with_name(&format!("x_testing_{}", i))]
        )
        .collect::<Vec<Vec<Value>>>();

    let y_testing_data = x_testing_data
        .iter()
        .enumerate()
        .map(|(i, x)|
            generate_y_value(&x[0], &mut rng, false).with_name(&format!("y_testing_{}", i))
        )
        .collect::<Vec<_>>();

    // Generate predictions for the test data using the trained model.
    let y_predictions = x_testing_data
        .iter()
        .enumerate()
        .map(|(i, x)|
            model.forward(&x)
                .iter()
                .enumerate()
                .map(|(j, y)|
                    y.with_name(&format!("y_prediction_{}_{}", i, j))
                )
                .collect::<Vec<_>>()
        )
        .collect::<Vec<_>>();

    // Compare predicted `y` values to actual `y` values.
    for (x_provided, y_predicted, y_actual) in izip!(&x_testing_data, &y_predictions, &y_testing_data) {
        println!("x: {:.2}, y: {:.2} vs {:.2}", x_provided[0].data(), y_predicted[0].data(), y_actual.data());
    }

    // Calculate mean squared error (MSE) for test data.
    let mse_test = mse(&y_predictions, &y_testing_data).with_name("mse_test");

    println!("Test set mean squared error: {}", mse_test.data());

    ///////////////////////
    // Visualizing       //
    ///////////////////////

    if should_output_graphviz {
        fs::write(
            "./graph.dot",
            format!("{}", create_graph(&mse_test))
        ).expect("Unable to write file: ./graph.dot");
    }
}
