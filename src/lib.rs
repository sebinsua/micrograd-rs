mod engine;
pub use crate::engine::{Operation, Value};

mod nn;
pub use crate::nn::{MLP, Layer, Neuron};

mod graph;
pub use crate::graph::{create_graph};