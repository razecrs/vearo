//! Script to generate Kaggle submission files on CUDA using Vearo.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap,
    clippy::uninlined_format_args,
    clippy::identity_op,
    clippy::expect_fun_call,
    clippy::missing_const_for_fn,
    clippy::manual_is_multiple_of,
    clippy::manual_range_contains,
    clippy::too_many_lines
)]

use std::fs::File;
use std::io::Write;
use vearo::nn::Module;
use vearo::{Device, Tensor};


/// Resolves a dataset path: $VEARO_DATA_DIR, then <repo>/data/kaggle, then legacy
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

#[test]
#[ignore = "requires datasets; run scripts/setup_data.sh then: cargo test --release --test generate_submissions -- --ignored"]
fn test_generate_submissions() {
    vearo::init();
    let device = Device::Cuda(0);

    // ----------------- 1. Train Tabular Regression Model & Generate Submission -----------------
    println!("Training Tabular Regression Model on GPU...");
    let x_train_data = load_bin_f32_host(&data_path("preprocessed/tabular_X_train.bin"));
    let y_train_data = load_bin_f32_host(&data_path("preprocessed/tabular_y_train.bin"));
    let x_test_data = load_bin_f32_host(&data_path("preprocessed/tabular_X_test.bin"));

    let mlp_tab = TabularMlp::new().to(device);
    let mut opt_tab = vearo::optim::AdamW::new(mlp_tab.parameters(), 0.005, 0.9, 0.999, 1e-8, 0.0);

    let batch_size = 128;
    let num_samples = 4800;

    for epoch in 0..30 {
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        for i in (0..num_samples).step_by(batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + batch_size, num_samples);
            let size = end_idx - i;

            let x_batch = Tensor::from_f32(&x_train_data[i * 46..end_idx * 46], vec![size, 46]).to(device);
            let y_batch = Tensor::from_f32(&y_train_data[i * 1..end_idx * 1], vec![size, 1]).to(device);

            let pred = mlp_tab.forward(&x_batch);
            let diff = pred.sub(&y_batch);
            let squared = diff.mul(&diff);
            let loss = squared.mean(0, false);

            epoch_loss += loss.to_vec_f32()[0];
            batches += 1;

            loss.backward();
            opt_tab.step();
        }
        if (epoch + 1) % 10 == 0 {
            println!("Tabular Epoch {} | Train Loss: {:.6}", epoch + 1, epoch_loss / batches as f32);
        }
    }

    // Run inference on the tabular test set
    println!("Generating predictions for tabular test set...");
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    
    let x_test = Tensor::from_f32(&x_test_data, vec![2523, 46]).to(device);
    let tab_preds = mlp_tab.forward(&x_test).to(Device::Cpu).to_vec_f32();

    // Write tabular submission CSV
    let tab_sub_path = &data_path("item_price_submission.csv");
    let mut file_tab = File::create(tab_sub_path).unwrap();
    writeln!(file_tab, "row_id,Y").unwrap();
    for (i, val) in tab_preds.iter().enumerate() {
        writeln!(file_tab, "{},{}", i, val).unwrap();
    }
    println!("Saved tabular submission to: {}", tab_sub_path);

    // ----------------- 2. Train Image Classification Model & Generate Submission -----------------
    println!("\nTraining Image Classification Model on GPU...");
    let x_train_img = load_bin_f32_host(&data_path("preprocessed/image_X_train.bin"));
    let y_train_img = load_bin_f32_host(&data_path("preprocessed/image_y_train.bin"));
    let x_test_img = load_bin_f32_host(&data_path("preprocessed/image_X_test.bin"));

    let mlp_img = ImageMlp::new().to(device);
    let mut opt_img = vearo::optim::AdamW::new(mlp_img.parameters(), 0.003, 0.9, 0.999, 1e-8, 0.0);

    let num_img_samples = 1360;

    for epoch in 0..25 {
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        for i in (0..num_img_samples).step_by(batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + batch_size, num_img_samples);
            let size = end_idx - i;

            let x_batch = Tensor::from_f32(&x_train_img[i * 3072..end_idx * 3072], vec![size, 3072]).to(device);
            let y_batch = Tensor::from_f32(&y_train_img[i * 1..end_idx * 1], vec![size]).to(device);

            let pred = mlp_img.forward(&x_batch);
            let loss = pred.cross_entropy(&y_batch);

            epoch_loss += loss.to_vec_f32()[0];
            batches += 1;

            loss.backward();
            opt_img.step();
        }
        if (epoch + 1) % 5 == 0 {
            println!("Image Epoch {} | Train Loss: {:.6}", epoch + 1, epoch_loss / batches as f32);
        }
    }

    // Run inference on the image test set (5482 samples)
    println!("Generating predictions for image test set...");
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();

    let x_test_t = Tensor::from_f32(&x_test_img, vec![5482, 3072]).to(device);
    let img_preds = mlp_img.forward(&x_test_t).to(Device::Cpu).to_vec_f32();

    // Read the test image names from sample_submission.csv
    let sample_sub_content = std::fs::read_to_string(&data_path("scene_style/sample_submission.csv")).unwrap();
    let image_names: Vec<String> = sample_sub_content
        .lines()
        .skip(1)
        .filter_map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts[0].to_string())
            }
        })
        .collect();

    assert_eq!(image_names.len(), 5482, "Image test names count mismatch");

    // Write image submission CSV
    let img_sub_path = &data_path("scene_style_submission.csv");
    let mut file_img = File::create(img_sub_path).unwrap();
    writeln!(file_img, "ImageName,ClassLabel").unwrap();
    for s in 0..5482 {
        let mut max_idx = 0;
        let mut max_val = -1e30f32;
        for c in 0..17 {
            let val = img_preds[s * 17 + c];
            if val > max_val {
                max_val = val;
                max_idx = c;
            }
        }
        writeln!(file_img, "{},{}", image_names[s], max_idx).unwrap();
    }
    println!("Saved image submission to: {}", img_sub_path);
}
