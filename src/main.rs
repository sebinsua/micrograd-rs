use std::{env, vec, fs};
use rand::Rng;
use itertools::izip;
use micrograd_rs::{Value, MLP, create_graph};


fn generate_x_data(start: f64, end: f64, size: usize) -> Vec<Vec<Value>> {
    (0..size)
        .map(|i| vec![Value::from(start + (i as f64) * (end - start) / ((size - 1) as f64))])
        .collect()
}

fn generate_y_value(x: &Value, rng: &mut rand::rngs::ThreadRng, with_noise: bool) -> Value {
    let noise = if with_noise == true { rng.gen_range(-0.1..=0.1) } else { 0.0 };
    Value::from(2.0 * x.data() + 3.0 + noise)
}

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

fn main() {
    let should_output_graphviz = get_graphviz_output_arg();

    let mut rng = rand::thread_rng();

    // Generate synthetic data for training
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

    for (x, y) in x_training_data.iter().zip(y_training_data.iter()) {
        println!("x: {:.2}, y: {:.2}", x[0].data(), y.data());
    }

    // Creating a model
    let mut model = MLP::new(1, vec![5, 5, 1]);

    fn loss(x_data: &[Vec<Value>], y_data: &Vec<Value>, model: &MLP) -> Value {
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
        
        let mse_loss = y_predictions
            .iter()
            .zip(y_data.iter())
            .map(|(y1, y2)|
                (&y1[0] - y2).powi(2)
            )
            .sum::<Value>() / Value::from(y_data.len() as f64);
        
        mse_loss.with_name("mse_loss")
    }

    let epochs = 1000;
    let learning_rate = 0.01;

    // Training loop
    for epoch in 0..epochs {
        // Forward pass
        let mut total_loss = loss(&x_training_data, &y_training_data, &model);

        // Backward pass
        model.zero_gradients();
        total_loss.backward();

        // Parameter update (SGD)
        for p in model.parameters() {
            p.decrement_data(learning_rate * p.gradient());
        }

        if epoch % 10 == 0 {
            println!("Epoch: {} / Loss: {}", epoch, total_loss.data());
        }
    }

    // Generate synthetic data for testing
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

    // Generate predictions for test data
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

    for (x_provided, y_predicted, y_expected) in izip!(&x_testing_data, &y_predictions, &y_testing_data) {
        println!("x: {:.2}, y: {:.2} vs {:.2}", x_provided[0].data(), y_predicted[0].data(), y_expected.data());
    }

    // Calculate MSE for test data
    let mse_test = y_predictions
        .iter()
        .zip(y_testing_data.into_iter())
        .map(|(y_predicted, y_expected)|
            (&y_expected - &y_predicted[0]).powi(2)
        )
        .sum::<Value>() / Value::from(x_testing_data.len() as f64)
        .with_name("mse_test");

    println!("Test set mean squared error: {}", mse_test.data());

    if should_output_graphviz {
        fs::write(
            "./graph.dot",
            format!("{}", create_graph(&mse_test))
        ).expect("Unable to write file: ./graph.dot");
    }
}
