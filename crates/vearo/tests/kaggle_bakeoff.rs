//! Kaggle bakeoff comparative benchmarks.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::uninlined_format_args,
    clippy::identity_op,
    clippy::expect_fun_call
)]

// This file exercises the CUDA backend directly, so it only builds with the
// `cuda` feature. CPU coverage of the same ops lives in the other test files.
#![cfg(feature = "cuda")]

use std::time::Instant;
use vearo::nn::Module;
use vearo::{Device, Tensor};

/// Resolves a dataset path: `$VEARO_DATA_DIR`, then `<repo>/data/kaggle`, then legacy
/// developer locations. Populate it with `scripts/setup_data.sh`.
fn data_path(suffix: &str) -> String {
    if let Ok(dir) = std::env::var("VEARO_DATA_DIR") {
        let p = format!("{dir}/{suffix}");
        if std::path::Path::new(&p).exists() {
            return p;
        }
    }
    let repo = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/kaggle");
    let p = format!("{repo}/{suffix}");
    if std::path::Path::new(&p).exists() {
        return p;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let legacy = format!("{home}/projects/kaggle_competitions/{suffix}");
    if std::path::Path::new(&legacy).exists() {
        return legacy;
    }
    format!("{home}/kaggle_competitions/{suffix}")
}

fn load_bin_f32_host(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(&format!("Failed to read {}", path));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct TabularMlp {
    fc1: vearo::nn::Linear,
    fc2: vearo::nn::Linear,
    fc3: vearo::nn::Linear,
}

impl TabularMlp {
    fn new() -> Self {
        Self {
            fc1: vearo::nn::Linear::new(46, 64, true, 42),
            fc2: vearo::nn::Linear::new(64, 32, true, 43),
            fc3: vearo::nn::Linear::new(32, 1, true, 44),
        }
    }

    fn to(&self, device: Device) -> Self {
        Self {
            fc1: self.fc1.to(device),
            fc2: self.fc2.to(device),
            fc3: self.fc3.to(device),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.fc1.forward(x).relu();
        let h2 = self.fc2.forward(&h1).relu();
        self.fc3.forward(&h2)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

struct ImageMlp {
    fc1: vearo::nn::Linear,
    fc2: vearo::nn::Linear,
    fc3: vearo::nn::Linear,
}

impl ImageMlp {
    fn new() -> Self {
        Self {
            fc1: vearo::nn::Linear::new(3072, 128, true, 42),
            fc2: vearo::nn::Linear::new(128, 64, true, 43),
            fc3: vearo::nn::Linear::new(64, 17, true, 44),
        }
    }

    fn to(&self, device: Device) -> Self {
        Self {
            fc1: self.fc1.to(device),
            fc2: self.fc2.to(device),
            fc3: self.fc3.to(device),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.fc1.forward(x).relu();
        let h2 = self.fc2.forward(&h1).relu();
        self.fc3.forward(&h2)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

fn run_tabular(device: Device) -> f64 {
    println!("\n--- Vearo Tabular Regression on {:?} ---", device);

    let x_train_data = load_bin_f32_host(&data_path("preprocessed/tabular_X_train.bin"));
    let y_train_data = load_bin_f32_host(&data_path("preprocessed/tabular_y_train.bin"));
    let x_val_data = load_bin_f32_host(&data_path("preprocessed/tabular_X_val.bin"));
    let y_val_data = load_bin_f32_host(&data_path("preprocessed/tabular_y_val.bin"));

    let x_val = Tensor::from_f32(&x_val_data, vec![1200, 46]).to(device);
    let y_val = Tensor::from_f32(&y_val_data, vec![1200, 1]).to(device);

    let mlp = TabularMlp::new().to(device);
    let mut optimizer = vearo::optim::AdamW::new(mlp.parameters(), 0.005, 0.9, 0.999, 1e-8, 0.0);

    let batch_size = 128;
    let num_samples = 4800;

    let start = Instant::now();

    for epoch in 0..20 {
        let mut epoch_loss = 0.0;
        let mut batches = 0;

        for i in (0..num_samples).step_by(batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + batch_size, num_samples);
            let size = end_idx - i;

            let x_batch_slice = &x_train_data[i * 46..end_idx * 46];
            let y_batch_slice = &y_train_data[i * 1..end_idx * 1];

            let x_batch = Tensor::from_f32(x_batch_slice, vec![size, 46]).to(device);
            let y_batch = Tensor::from_f32(y_batch_slice, vec![size, 1]).to(device);

            let pred = mlp.forward(&x_batch);

            let diff = pred.sub(&y_batch);
            let squared = diff.mul(&diff);
            let loss = squared.mean(0, false);

            epoch_loss += loss.to_vec_f32()[0];
            batches += 1;

            loss.backward();
            optimizer.step();
        }

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();
            let val_pred = mlp.forward(&x_val);
            let val_diff = val_pred.sub(&y_val);
            let val_squared = val_diff.mul(&val_diff);
            let val_loss = val_squared.mean(0, false).to_vec_f32()[0];

            println!(
                "Epoch {:02} | Train Loss: {:.6} | Val Loss: {:.6}",
                epoch + 1,
                epoch_loss / batches as f32,
                val_loss
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("Tabular finished in: {:.4} seconds", elapsed);
    elapsed
}

fn run_image(device: Device) -> f64 {
    println!("\n--- Vearo Image Classification on {:?} ---", device);

    let x_train_data = load_bin_f32_host(&data_path("preprocessed/image_X_train.bin"));
    let y_train_data = load_bin_f32_host(&data_path("preprocessed/image_y_train.bin"));
    let x_val_data = load_bin_f32_host(&data_path("preprocessed/image_X_val.bin"));
    let y_val_data = load_bin_f32_host(&data_path("preprocessed/image_y_val.bin"));

    // Sizes are derived from the data, not hardcoded: re-exporting the dataset
    // (e.g. the CHW fix) changes the split sizes and would otherwise break this test.
    let val_size = x_val_data.len() / 3072;
    let x_val = Tensor::from_f32(&x_val_data, vec![val_size, 3072]).to(device);
    let y_val = Tensor::from_f32(&y_val_data, vec![val_size]).to(device);

    let mlp = ImageMlp::new().to(device);
    let mut optimizer = vearo::optim::AdamW::new(mlp.parameters(), 0.003, 0.9, 0.999, 1e-8, 0.0);

    let batch_size = 128;
    let num_samples = x_train_data.len() / 3072;

    let start = Instant::now();

    for epoch in 0..15 {
        let mut epoch_loss = 0.0;
        let mut batches = 0;

        for i in (0..num_samples).step_by(batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + batch_size, num_samples);
            let size = end_idx - i;

            let x_batch_slice = &x_train_data[i * 3072..end_idx * 3072];
            let y_batch_slice = &y_train_data[i * 1..end_idx * 1];

            let x_batch = Tensor::from_f32(x_batch_slice, vec![size, 3072]).to(device);
            let y_batch = Tensor::from_f32(y_batch_slice, vec![size]).to(device);

            let pred = mlp.forward(&x_batch);
            let loss = pred.cross_entropy(&y_batch);

            epoch_loss += loss.to_vec_f32()[0];
            batches += 1;

            loss.backward();
            optimizer.step();
        }

        if (epoch + 1) % 5 == 0 || epoch == 0 {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();
            let val_pred = mlp.forward(&x_val);
            let val_loss = val_pred.cross_entropy(&y_val).to_vec_f32()[0];

            let val_pred_vec = val_pred.to_vec_f32();
            let val_y_vec = y_val.to_vec_f32();
            let mut correct = 0;
            for s in 0..val_size {
                let mut max_idx = 0;
                let mut max_val = -1e30f32;
                for c in 0..17 {
                    let val = val_pred_vec[s * 17 + c];
                    if val > max_val {
                        max_val = val;
                        max_idx = c;
                    }
                }
                if max_idx == val_y_vec[s] as usize {
                    correct += 1;
                }
            }
            let val_acc = correct as f32 / val_size as f32;

            println!(
                "Epoch {:02} | Train Loss: {:.6} | Val Loss: {:.6} | Val Acc: {:.2}%",
                epoch + 1,
                epoch_loss / batches as f32,
                val_loss,
                val_acc * 100.0
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("Image finished in: {:.4} seconds", elapsed);
    elapsed
}

#[test]
#[ignore = "requires datasets; run scripts/setup_data.sh then: cargo test --release --test kaggle_bakeoff -- --ignored"]
fn test_kaggle_bakeoff() {
    vearo::init();

    // 1. Run CPU Vearo
    let cpu_tab = run_tabular(Device::Cpu);
    let cpu_img = run_image(Device::Cpu);

    // 2. Run CUDA Vearo
    let cuda_tab = run_tabular(Device::Cuda(0));
    let cuda_img = run_image(Device::Cuda(0));

    println!("\nBakeoff speedups:");
    println!("Tabular CPU to CUDA speedup: {:.2}x", cpu_tab / cuda_tab);
    println!("Image CPU to CUDA speedup: {:.2}x", cpu_img / cuda_img);
}
