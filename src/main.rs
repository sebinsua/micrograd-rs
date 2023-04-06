use rand::Rng;
use micrograd_rs::{Value, MLP};
use itertools::izip;

fn generate_x_data(start: f64, end: f64, size: usize) -> Vec<Vec<Value>> {
    (0..size)
        .map(|i| vec![Value::from(start + (i as f64) * (end - start) / ((size - 1) as f64))])
        .collect()
}

fn generate_y_value(x: &Value, rng: &mut rand::rngs::ThreadRng, with_noise: bool) -> Value {
    let noise = if with_noise == true { rng.gen_range(-0.1..=0.1) } else { 0.0 };
    Value::from(2.0 * x.data() + 3.0 + noise)
}

fn main() {
    let mut rng = rand::thread_rng();

    // Generate synthetic data for training
    let x_training_data = generate_x_data(
        -2.0,
        2.0,
        50 as usize
    );

    let y_training_data: Vec<Value> = x_training_data
        .iter()
        .map(|x| generate_y_value(&x[0], &mut rng, true))
        .collect();

    for (x, y) in x_training_data.iter().zip(y_training_data.iter()) {
        println!("x: {:.2}, y: {:.2}", x[0].data(), y.data());
    }

    // Creating a model
    let mut model = MLP::new(1, vec![20, 10, 1]);

    fn loss(x_data: &[Vec<Value>], y_data: &[Value], model: &MLP) -> Value {
        let y_pred = x_data
            .iter()
            .map(|x| model.forward(&x))
            .collect::<Vec<_>>();
        
        let mse_loss = y_pred
            .iter()
            .zip(y_data.iter())
            .map(|(y1, y2)|
                (y1[0].clone() - y2.clone()).powi(2)
            )
            .sum::<Value>() / Value::from(y_data.len() as f64);
        
        mse_loss
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
    let x_testing_data = generate_x_data(
        -3.0,
        3.0,
        10 as usize
    );

    let y_testing_data = x_testing_data
        .iter()
        .map(|x| generate_y_value(&x[0], &mut rng, false))
        .collect::<Vec<_>>();

    // Generate predictions for test data
    let y_predictions = x_testing_data
        .iter()
        .map(|x| model.forward(&x))
        .collect::<Vec<_>>();
    
    for (x_provided, y_predicted, y_expected) in izip!(&x_testing_data, &y_predictions, &y_testing_data) {
        println!("x: {:.2}, y: {:.2} vs {:.2}", x_provided[0].data(), y_predicted[0].data(), y_expected.data());
    }

    // Calculate MSE for test data
    let mse_test = y_predictions
        .iter()
        .zip(y_testing_data.into_iter())
        .map(|(y_predicted, y_expected)|
            (y_expected - y_predicted[0].clone()).powi(2)
        )
        .sum::<Value>() / Value::from(x_testing_data.len() as f64);

    println!("Test set mean squared error: {}", mse_test.data());
}
