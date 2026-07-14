//! Verification harness integration tests against `NumPy` ground truth.

use serde::Deserialize;
use vearo::{Shape, Tensor, init};

#[derive(Deserialize, Debug)]
struct OracleTensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct OracleTestCase {
    op: String,
    inputs: Vec<OracleTensor>,
    output: OracleTensor,
}

fn assert_tensor_close(got: &Tensor, expected_shape: &[usize], expected_data: &[f32]) {
    assert_eq!(got.shape().dims(), expected_shape);

    let contiguous_got = got.contiguous();
    let shard = contiguous_got.storage_id().shard_idx as usize;
    let slot_idx = contiguous_got.storage_id().slot_idx as usize;
    // Copy the data out under the lock (temporary guard drops at the match end),
    // then compare unlocked.
    let got_vec = match &vearo::core::get_cpu_shard(shard).lock().unwrap().slots[slot_idx]
        .as_ref()
        .expect("Output slot was empty")
        .storage
    {
        vearo::core::CpuStorage::F32(v) => v.as_ref().clone(),
        _ => panic!("Expected F32 storage"),
    };

    assert_eq!(got_vec.len(), expected_data.len());
    for (i, (&g, &e)) in got_vec.iter().zip(expected_data.iter()).enumerate() {
        assert!(
            (g - e).abs() <= 1e-5,
            "Mismatch at index {i}: got {g}, expected {e}"
        );
    }
}

#[test]
fn test_numpy_oracle_parity() {
    init();

    let oracles_json = include_str!("../../../test_data/oracles.json");
    let cases: Vec<OracleTestCase> = serde_json::from_str(oracles_json).unwrap();

    for case in cases {
        match case.op.as_str() {
            "add" | "add_transposed" | "add_broadcasted" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.add(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "sub" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.sub(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "mul" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.mul(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "div" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.div(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "matmul_2d" | "matmul_batched" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let y = Tensor::from_f32(&case.inputs[1].data, case.inputs[1].shape.clone());
                let got = x.matmul(&y);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "reshape" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let out_shape: Shape = case.output.shape.clone().into();
                let got = x.reshape(out_shape);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "transpose" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let got = x.transpose(0, 1);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            "permute" => {
                let x = Tensor::from_f32(&case.inputs[0].data, case.inputs[0].shape.clone());
                let got = x.permute([2, 0, 1]);
                assert_tensor_close(&got, &case.output.shape, &case.output.data);
            }
            _ => unreachable!("Unknown operation: {}", case.op),
        }
    }
}

#[test]
#[allow(clippy::items_after_statements)]
fn test_mlp_overfitting() {
    use vearo::nn::Module;
    vearo::init();

    // 1. Define XOR dataset: inputs [4, 2], targets [4, 1]
    let inputs = Tensor::from_f32(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2]);
    let targets = Tensor::from_f32(&[0.0, 1.0, 1.0, 0.0], [4, 1]);

    // 2. Define a tiny MLP: Linear(2 -> 4) -> ReLU -> Linear(4 -> 1)
    struct Mlp {
        fc1: vearo::nn::Linear,
        fc2: vearo::nn::Linear,
    }

    impl Mlp {
        fn forward(&self, x: &Tensor) -> Tensor {
            let h = self.fc1.forward(x).relu();
            self.fc2.forward(&h)
        }

        fn parameters(&self) -> Vec<Tensor> {
            let mut params = self.fc1.parameters();
            params.extend(self.fc2.parameters());
            params
        }
    }

    // Initialize with a fixed seed for deterministic behavior
    let mlp = Mlp {
        fc1: vearo::nn::Linear::new(2, 4, true, 42),
        fc2: vearo::nn::Linear::new(4, 1, true, 43),
    };

    // 3. Setup AdamW optimizer
    let mut optimizer = vearo::optim::AdamW::new(
        mlp.parameters(),
        0.1,   // learning rate
        0.9,   // beta1
        0.999, // beta2
        1e-8,  // eps
        0.0,   // weight decay
    );

    // 4. Run training loop to overfit
    let mut final_loss = 1.0;
    for _epoch in 0..100 {
        // Zero gradients and reset tape
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        // Forward
        let pred = mlp.forward(&inputs);

        // Compute loss: MSE = mean((pred - target)^2)
        let diff = pred.sub(&targets);
        let squared = diff.mul(&diff);
        // Reduce over batch dimension (0), then over output dimension (0)
        let loss = squared.mean(0, false).mean(0, false);

        final_loss = loss.get_f32(0);

        // Backward
        loss.backward();

        // Step
        optimizer.step();
    }

    println!("Overfitting finished. Final MSE Loss: {final_loss}");
    // Verify that the loss successfully converged to near zero
    assert!(
        final_loss < 5e-3,
        "MLP failed to overfit XOR batch; final loss was {final_loss}"
    );
}

#[test]
fn test_gpt_overfitting() {
    vearo::init();

    // 1. Inputs: shape [2, 4], targets: shape [2, 4] (predict next token)
    let inputs = Tensor::from_f32(&[0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0], [2, 4]);
    let targets = Tensor::from_f32(&[1.0, 2.0, 3.0, 0.0, 2.0, 3.0, 0.0, 1.0], [2, 4]);

    // 2. Initialize SimpleGPT
    let gpt = vearo::nn::SimpleGPT::new(
        8,  // vocab_size
        4,  // max_seq_len
        8,  // n_embd
        2,  // n_head
        1,  // n_layer
        16, // mlp_dim
        42, // seed
    );

    // 3. Setup AdamW optimizer
    let mut optimizer = vearo::optim::AdamW::new(
        gpt.parameters(),
        0.02, // lr
        0.9,  // beta1
        0.95, // beta2
        1e-8, // eps
        0.01, // weight_decay
    );

    // 4. Run training loop to overfit
    let mut final_loss = 5.0;
    for _epoch in 0..150 {
        // Zero gradients and reset tape
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();

        // Forward
        let (_logits, loss_opt) = gpt.forward(&inputs, Some(&targets));
        let loss = loss_opt.unwrap();

        final_loss = loss.get_f32(0);

        // Backward
        loss.backward();

        // Step
        optimizer.step();
    }

    println!("GPT overfitting finished. Final Cross Entropy Loss: {final_loss}");
    assert!(
        final_loss < 0.05,
        "SimpleGPT failed to overfit; final loss was {final_loss}"
    );
}
