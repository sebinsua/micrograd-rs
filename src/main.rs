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
    // regularization (e.g. L1 or L2 also known as "weight decay"), early stopping and learning
    // rate scheduling, among others.
    let epochs = 1000;
    let learning_rate = 0.01;

    // Create a multi-layer perceptron (MLP) model with one input neuron in the input layer,
    // two hidden layers with five neurons each, and one output neuron in the output layer.
    let mut model = MLP::new(1, vec![5, 5, 1]);

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
        for p in model.parameters() {
            // We are attempting to minimize the loss, so we subtract the gradient from the parameter value.
            // However, a negative gradient would result in an increase to the parameter value.
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
