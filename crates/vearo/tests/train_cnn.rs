//! Integration test to train the high-quality CNN model on the full image style dataset
//! and generate the final submissions.
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
use std::time::Instant;
use vearo::nn::Module;
use vearo::{Device, Tensor};

/// Resolves a dataset path: `$VEARO_DATA_DIR`, then `<repo>/data/kaggle`, then legacy
/// developer locations. Populate it with `scripts/setup_data.sh`.
/// Image side length, matching whatever `scripts/preprocess.py --size N` produced.
/// Read at runtime so changing resolution needs no recompile:
///     `python3 scripts/preprocess.py --size 64`
///     `VEARO_IMG_SIZE=64 cargo test --release -p vearo --test train_cnn -- --ignored`
fn img_size() -> usize {
    std::env::var("VEARO_IMG_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(32)
}

/// Floats per image: three channels at `s` by `s`.
fn px(s: usize) -> usize {
    3 * s * s
}

/// Resolves a preprocessed image file for side length `s`.
///
/// Each resolution lives in `preprocessed/img_<s>/`. Data produced before that
/// split sits directly in `preprocessed/`, so 32px falls back there rather than
/// forcing everyone to re-run preprocessing.
fn image_path(s: usize, name: &str) -> String {
    let sized = get_kaggle_path(&format!("preprocessed/img_{s}/{name}"));
    if std::path::Path::new(&sized).exists() {
        return sized;
    }
    let legacy = get_kaggle_path(&format!("preprocessed/{name}"));
    assert!(
        s == 32 || std::path::Path::new(&legacy).exists(),
        "no preprocessed images for size {s}. Run: python3 scripts/preprocess.py --size {s}"
    );
    legacy
}

/// Feature count produced by `preprocess_tabular` in scripts/preprocess.py.
const TAB_FEATURES: usize = 49;
/// Tabular epochs. Best epoch is picked by validation, so a longer budget is safe.
const TAB_EPOCHS: usize = 60;

fn get_kaggle_path(relative_suffix: &str) -> String {
    if let Ok(dir) = std::env::var("VEARO_DATA_DIR") {
        return format!("{dir}/{relative_suffix}");
    }
    let repo = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/kaggle");
    let p = format!("{repo}/{relative_suffix}");
    if std::path::Path::new(&p).exists() {
        return p;
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let legacy = format!("{home}/projects/kaggle_competitions/{relative_suffix}");
    if std::path::Path::new(&legacy).exists() {
        return legacy;
    }
    format!("{home}/kaggle_competitions/{relative_suffix}")
}

fn load_bin_f32_host(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).expect(&format!("Failed to read {}", path));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Simple host-side data augmentation (random horizontal flips and crops/shifts)
fn augment_batch(batch_data: &mut [f32], size: usize, rng_seed: &mut u64, s: usize) {
    // Simple LCG RNG
    fn next_rand(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 32) as u32
    }

    let plane = s * s;
    let shift = (s / 8).max(1) as i32; // same relative jitter at any resolution
    for b in 0..size {
        let img_offset = b * px(s);
        let img = &mut batch_data[img_offset..img_offset + px(s)];

        // 1. Random Horizontal Flip (50% probability)
        if next_rand(rng_seed) % 2 == 0 {
            for c in 0..3 {
                let c_offset = c * plane;
                for y in 0..s {
                    let row_offset = c_offset + y * s;
                    for x in 0..s / 2 {
                        img.swap(row_offset + x, row_offset + (s - 1 - x));
                    }
                }
            }
        }

        // 2. Random Translation / Shift (up to +/- 4 pixels with zero padding)
        let span = (shift * 2 + 1) as u32;
        let dx = (next_rand(rng_seed) % span) as i32 - shift;
        let dy = (next_rand(rng_seed) % span) as i32 - shift;

        if dx != 0 || dy != 0 {
            let mut temp = vec![0.0f32; px(s)];
            for c in 0..3 {
                let c_offset = c * plane;
                for y in 0..s {
                    let target_y = y as i32 + dy;
                    if target_y < 0 || target_y >= s as i32 {
                        continue;
                    }
                    for x in 0..s {
                        let target_x = x as i32 + dx;
                        if target_x < 0 || target_x >= s as i32 {
                            continue;
                        }
                        let src_idx = c_offset + y * s + x;
                        let dest_idx = c_offset + (target_y as usize) * s + (target_x as usize);
                        temp[dest_idx] = img[src_idx];
                    }
                }
            }
            img.copy_from_slice(&temp);
        }
    }
}

struct TabularMlp {
    fc1: vearo::nn::Linear,
    fc2: vearo::nn::Linear,
    fc3: vearo::nn::Linear,
}

impl TabularMlp {
    fn new() -> Self {
        Self {
            fc1: vearo::nn::Linear::new(TAB_FEATURES, 64, true, 42),
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
        self.fc3
            .forward(&self.fc2.forward(&self.fc1.forward(x).relu()).relu())
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params
    }
}

#[allow(dead_code)]
struct TabularResMlp {
    fc1: vearo::nn::Linear,
    fc2: vearo::nn::Linear,
    fc3: vearo::nn::Linear,
    fc4: vearo::nn::Linear,
}

#[allow(dead_code)]
impl TabularResMlp {
    fn new() -> Self {
        Self {
            fc1: vearo::nn::Linear::new(TAB_FEATURES, 128, true, 42),
            fc2: vearo::nn::Linear::new(128, 128, true, 43),
            fc3: vearo::nn::Linear::new(128, 64, true, 44),
            fc4: vearo::nn::Linear::new(64, 1, true, 48),
        }
    }

    fn to(&self, device: Device) -> Self {
        Self {
            fc1: self.fc1.to(device),
            fc2: self.fc2.to(device),
            fc3: self.fc3.to(device),
            fc4: self.fc4.to(device),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self.fc1.forward(x).relu();
        let h2 = self.fc2.forward(&h1).relu();
        let h2_skip = &h1 + &h2;
        let h3 = self.fc3.forward(&h2_skip).relu();
        self.fc4.forward(&h3)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
        params.extend(self.fc4.parameters());
        params
    }
}

struct StyleCnn {
    conv1: vearo::nn::Conv2d,
    bn1: vearo::nn::BatchNorm2d,
    pool1: vearo::nn::MaxPool2d,

    conv2: vearo::nn::Conv2d,
    bn2: vearo::nn::BatchNorm2d,
    pool2: vearo::nn::MaxPool2d,

    conv3: vearo::nn::Conv2d,
    bn3: vearo::nn::BatchNorm2d,
    pool3: vearo::nn::MaxPool2d,

    fc1: vearo::nn::Linear,
    dropout: vearo::nn::Dropout,
    fc2: vearo::nn::Linear,
}

impl StyleCnn {
    /// Three 2x pools, so the feature map is `s / 8` on a side and the
    /// classifier input is `128 * (s/8)^2`. At 32px that is the original 2048.
    fn flat_dim(s: usize) -> usize {
        128 * (s / 8) * (s / 8)
    }

    fn new(s: usize) -> Self {
        Self {
            conv1: vearo::nn::Conv2d::new(3, 32, 3, 1, 1, true, 42),
            bn1: vearo::nn::BatchNorm2d::new(32, 0.1, 1e-5),
            pool1: vearo::nn::MaxPool2d::new(2, 2, 0),

            conv2: vearo::nn::Conv2d::new(32, 64, 3, 1, 1, true, 43),
            bn2: vearo::nn::BatchNorm2d::new(64, 0.1, 1e-5),
            pool2: vearo::nn::MaxPool2d::new(2, 2, 0),

            conv3: vearo::nn::Conv2d::new(64, 128, 3, 1, 1, true, 44),
            bn3: vearo::nn::BatchNorm2d::new(128, 0.1, 1e-5),
            pool3: vearo::nn::MaxPool2d::new(2, 2, 0),

            fc1: vearo::nn::Linear::new(Self::flat_dim(s), 128, true, 45),
            dropout: vearo::nn::Dropout::new(0.4, 46),
            fc2: vearo::nn::Linear::new(128, 17, true, 47),
        }
    }

    fn to(&self, device: Device) -> Self {
        Self {
            conv1: self.conv1.to(device),
            bn1: self.bn1.to(device),
            pool1: self.pool1.to(device),

            conv2: self.conv2.to(device),
            bn2: self.bn2.to(device),
            pool2: self.pool2.to(device),

            conv3: self.conv3.to(device),
            bn3: self.bn3.to(device),
            pool3: self.pool3.to(device),

            fc1: self.fc1.to(device),
            dropout: self.dropout.to(device),
            fc2: self.fc2.to(device),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = self
            .pool1
            .forward(&self.bn1.forward(&self.conv1.forward(x)).relu());
        let h2 = self
            .pool2
            .forward(&self.bn2.forward(&self.conv2.forward(&h1)).relu());
        let h3 = self
            .pool3
            .forward(&self.bn3.forward(&self.conv3.forward(&h2)).relu());

        let dims = h3.shape().dims().to_vec();
        let b = dims[0];
        let flat = h3.reshape([b, dims[1] * dims[2] * dims[3]]);
        let h4 = self.dropout.forward(&self.fc1.forward(&flat).relu());
        self.fc2.forward(&h4)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.conv1.parameters();
        params.extend(self.bn1.parameters());

        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());

        params.extend(self.conv3.parameters());
        params.extend(self.bn3.parameters());

        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

#[test]
#[ignore = "run programmatic sweep over hyperparameters for tabular model; cargo test --release --test train_cnn test_sweep_tabular -- --ignored --nocapture"]
fn test_sweep_tabular() {
    vearo::init();
    // CPU-only builds have no CUDA backend registered; run there instead of
    // dispatching to a device that cannot execute.
    let device = if vearo::cuda_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    let x_train_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_train.bin"));
    let y_train_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_y_train.bin"));
    let x_val_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_val.bin"));
    let y_val_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_y_val.bin"));

    let tab_val_size = y_val_tab.len();
    let tab_train_size = y_train_tab.len();

    let tab_val_inputs = Tensor::from_f32(&x_val_tab, vec![tab_val_size, TAB_FEATURES]).to(device);
    let tab_val_targets = Tensor::from_f32(&y_val_tab, vec![tab_val_size, 1]).to(device);

    let lr_sweep = vec![0.002, 0.005, 0.01];
    let wd_sweep = vec![0.005, 0.01, 0.02, 0.03, 0.05, 0.1];
    let bs_sweep = vec![64, 128, 256];

    println!("\n| LR | WD | BS | Best Epoch | Best Val RMSE |");
    println!("|---|---|---|---|---|");

    for &lr in &lr_sweep {
        for &wd in &wd_sweep {
            for &bs in &bs_sweep {
                let mlp_tab = TabularMlp::new().to(device);
                let mut opt_tab =
                    vearo::optim::AdamW::new(mlp_tab.parameters(), lr, 0.9, 0.999, 1e-8, wd);

                let mut best_val = f32::INFINITY;
                let mut best_epoch = 0;

                let mut current_lr = lr;
                for epoch in 0..TAB_EPOCHS {
                    if epoch > 0 && epoch % 15 == 0 {
                        current_lr *= 0.5;
                        opt_tab.set_lr(current_lr);
                    }
                    for i in (0..tab_train_size).step_by(bs) {
                        vearo::autograd::zero_gradients();
                        vearo::autograd::reset_active_tape();

                        let end_idx = std::cmp::min(i + bs, tab_train_size);
                        let size = end_idx - i;

                        let x_batch = Tensor::from_f32(
                            &x_train_tab[i * TAB_FEATURES..end_idx * TAB_FEATURES],
                            vec![size, TAB_FEATURES],
                        )
                        .to(device);
                        let y_batch =
                            Tensor::from_f32(&y_train_tab[i * 1..end_idx * 1], vec![size, 1])
                                .to(device);

                        let pred = mlp_tab.forward(&x_batch);
                        let diff = pred.sub(&y_batch);
                        let squared = diff.mul(&diff);
                        let loss = squared.mean(0, false);

                        loss.backward();
                        opt_tab.step();
                    }

                    vearo::autograd::zero_gradients();
                    vearo::autograd::reset_active_tape();
                    let val_pred = mlp_tab.forward(&tab_val_inputs);
                    let val_diff = val_pred.sub(&tab_val_targets);
                    let val_squared = val_diff.mul(&val_diff);
                    let val_loss = val_squared.mean(0, false).to_vec_f32()[0];

                    if val_loss < best_val {
                        best_val = val_loss;
                        best_epoch = epoch + 1;
                    }
                }
                println!(
                    "| {:.4} | {:.4} | {} | {} | {:.5} |",
                    lr,
                    wd,
                    bs,
                    best_epoch,
                    best_val.sqrt()
                );
            }
        }
    }
}

#[test]
#[ignore = "run only the tabular regression model; cargo test --release --test train_cnn test_train_tabular_only -- --ignored --nocapture"]
fn test_train_tabular_only() {
    vearo::init();
    // CPU-only builds have no CUDA backend registered; run there instead of
    // dispatching to a device that cannot execute.
    let device = if vearo::cuda_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };

    println!("=== Training Tabular Regression Model ===");
    let x_train_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_train.bin"));
    let y_train_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_y_train.bin"));
    let x_val_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_val.bin"));
    let y_val_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_y_val.bin"));
    let x_test_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_test.bin"));

    let tab_val_size = y_val_tab.len();
    let tab_train_size = y_train_tab.len();
    let tab_test_size = x_test_tab.len() / TAB_FEATURES;

    let tab_val_inputs = Tensor::from_f32(&x_val_tab, vec![tab_val_size, TAB_FEATURES]).to(device);
    let tab_val_targets = Tensor::from_f32(&y_val_tab, vec![tab_val_size, 1]).to(device);

    let mlp_tab = TabularMlp::new().to(device);
    let mut tab_lr = 0.002f32;
    // Weight decay: adjustable hyperparameter. Default 0.01, optimal 0.10
    let mut opt_tab =
        vearo::optim::AdamW::new(mlp_tab.parameters(), tab_lr, 0.9, 0.999, 1e-8, 0.10);

    let tab_batch_size = 256;

    let mut best_val = f32::INFINITY;
    let mut best_epoch = 0;
    let mut best_tab_preds: Vec<f32> = Vec::new();
    let x_test_t = Tensor::from_f32(&x_test_tab, vec![tab_test_size, TAB_FEATURES]).to(device);

    let start_tab = Instant::now();
    for epoch in 0..TAB_EPOCHS {
        if epoch > 0 && epoch % 15 == 0 {
            tab_lr *= 0.5;
            opt_tab.set_lr(tab_lr);
        }
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        for i in (0..tab_train_size).step_by(tab_batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + tab_batch_size, tab_train_size);
            let size = end_idx - i;

            let x_batch = Tensor::from_f32(
                &x_train_tab[i * TAB_FEATURES..end_idx * TAB_FEATURES],
                vec![size, TAB_FEATURES],
            )
            .to(device);
            let y_batch =
                Tensor::from_f32(&y_train_tab[i * 1..end_idx * 1], vec![size, 1]).to(device);

            let pred = mlp_tab.forward(&x_batch);
            let diff = pred.sub(&y_batch);
            let squared = diff.mul(&diff);
            let loss = squared.mean(0, false);

            epoch_loss += loss.to_vec_f32()[0];
            batches += 1;

            loss.backward();
            opt_tab.step();
        }

        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();
        let val_pred = mlp_tab.forward(&tab_val_inputs);
        let val_diff = val_pred.sub(&tab_val_targets);
        let val_squared = val_diff.mul(&val_diff);
        let val_loss = val_squared.mean(0, false).to_vec_f32()[0];

        if val_loss < best_val {
            best_val = val_loss;
            best_epoch = epoch + 1;
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();
            best_tab_preds = mlp_tab.forward(&x_test_t).to(Device::Cpu).to_vec_f32();
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "Tabular Epoch {:02} | Train Loss: {:.6} | Val Loss: {:.6} | Val RMSE: {:.4}",
                epoch + 1,
                epoch_loss / batches as f32,
                val_loss,
                val_loss.sqrt()
            );
        }
    }
    println!(
        "Tabular Training completed in {:.4} seconds.",
        start_tab.elapsed().as_secs_f64()
    );

    println!(
        "Best tabular epoch {best_epoch}: val MSE {best_val:.6}, val RMSE {:.4}",
        best_val.sqrt()
    );

    println!("Generating tabular submission from epoch {best_epoch}...");
    let tab_preds = best_tab_preds;

    let tab_sub_path = get_kaggle_path("item_price_submission.csv");
    let mut file_tab = File::create(&tab_sub_path).unwrap();
    writeln!(file_tab, "row_id,Y").unwrap();
    for (i, val) in tab_preds.iter().enumerate() {
        writeln!(file_tab, "{},{}", i, val).unwrap();
    }
    println!("Saved tabular submission to: {}\n", tab_sub_path);
}

#[test]
#[ignore = "long training run needing datasets; run scripts/setup_data.sh then: cargo test --release --test train_cnn -- --ignored --nocapture"]
fn test_train_cnn_full() {
    vearo::init();
    // CPU-only builds have no CUDA backend registered; run there instead of
    // dispatching to a device that cannot execute.
    let device = if vearo::cuda_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    let s = img_size();

    // ----------------- 1. Train Tabular Regression Model -----------------
    println!("=== Training Tabular Regression Model ===");
    let x_train_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_train.bin"));
    let y_train_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_y_train.bin"));
    let x_val_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_val.bin"));
    let y_val_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_y_val.bin"));
    let x_test_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_test.bin"));

    // Sizes come from the data. Hardcoding them silently breaks the moment the
    // preprocessing split changes, which has already happened once.
    let tab_val_size = y_val_tab.len();
    let tab_train_size = y_train_tab.len();
    let tab_test_size = x_test_tab.len() / TAB_FEATURES;

    let tab_val_inputs = Tensor::from_f32(&x_val_tab, vec![tab_val_size, TAB_FEATURES]).to(device);
    let tab_val_targets = Tensor::from_f32(&y_val_tab, vec![tab_val_size, 1]).to(device);

    let mlp_tab = TabularMlp::new().to(device);
    let mut tab_lr = 0.002f32;
    // Weight decay: adjustable hyperparameter. Default 0.01, optimal 0.10
    let mut opt_tab =
        vearo::optim::AdamW::new(mlp_tab.parameters(), tab_lr, 0.9, 0.999, 1e-8, 0.10);

    let tab_batch_size = 256;

    // Best-checkpoint state. Validation loss bottoms out around epoch 10 and
    // then climbs, so shipping the last epoch ships an overfit model. There is
    // no in-place tensor copy to restore weights with, so this follows the same
    // approach the CNN uses below: run test inference at each new best and keep
    // those predictions.
    let mut best_val = f32::INFINITY;
    let mut best_epoch = 0;
    let mut best_tab_preds: Vec<f32> = Vec::new();
    let x_test_t = Tensor::from_f32(&x_test_tab, vec![tab_test_size, TAB_FEATURES]).to(device);

    let start_tab = Instant::now();
    for epoch in 0..TAB_EPOCHS {
        // Decay the learning rate so the tail of training refines rather than
        // bounces around the minimum.
        if epoch > 0 && epoch % 15 == 0 {
            tab_lr *= 0.5;
            opt_tab.set_lr(tab_lr);
        }
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        for i in (0..tab_train_size).step_by(tab_batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + tab_batch_size, tab_train_size);
            let size = end_idx - i;

            let x_batch = Tensor::from_f32(
                &x_train_tab[i * TAB_FEATURES..end_idx * TAB_FEATURES],
                vec![size, TAB_FEATURES],
            )
            .to(device);
            let y_batch =
                Tensor::from_f32(&y_train_tab[i * 1..end_idx * 1], vec![size, 1]).to(device);

            let pred = mlp_tab.forward(&x_batch);
            let diff = pred.sub(&y_batch);
            let squared = diff.mul(&diff);
            let loss = squared.mean(0, false);

            epoch_loss += loss.to_vec_f32()[0];
            batches += 1;

            loss.backward();
            opt_tab.step();
        }

        // Validate every epoch: checking every tenth cannot find the true best.
        vearo::autograd::zero_gradients();
        vearo::autograd::reset_active_tape();
        let val_pred = mlp_tab.forward(&tab_val_inputs);
        let val_diff = val_pred.sub(&tab_val_targets);
        let val_squared = val_diff.mul(&val_diff);
        let val_loss = val_squared.mean(0, false).to_vec_f32()[0];

        if val_loss < best_val {
            best_val = val_loss;
            best_epoch = epoch + 1;
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();
            best_tab_preds = mlp_tab.forward(&x_test_t).to(Device::Cpu).to_vec_f32();
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            println!(
                "Tabular Epoch {:02} | Train Loss: {:.6} | Val Loss: {:.6} | Val RMSE: {:.4}",
                epoch + 1,
                epoch_loss / batches as f32,
                val_loss,
                val_loss.sqrt()
            );
        }
    }
    println!(
        "Tabular Training completed in {:.4} seconds.",
        start_tab.elapsed().as_secs_f64()
    );

    println!(
        "Best tabular epoch {best_epoch}: val MSE {best_val:.6}, val RMSE {:.4}",
        best_val.sqrt()
    );

    // Generate Tabular Prediction submission.csv from the BEST epoch, not the last.
    println!("Generating tabular submission from epoch {best_epoch}...");
    let tab_preds = best_tab_preds;

    let tab_sub_path = get_kaggle_path("item_price_submission.csv");
    let mut file_tab = File::create(&tab_sub_path).unwrap();
    writeln!(file_tab, "row_id,Y").unwrap();
    for (i, val) in tab_preds.iter().enumerate() {
        writeln!(file_tab, "{},{}", i, val).unwrap();
    }
    println!("Saved tabular submission to: {}\n", tab_sub_path);

    // ----------------- 2. Train Image Style CNN Model -----------------
    println!("=== Training Image Style CNN Model ===");
    let x_train_img = load_bin_f32_host(&image_path(s, "image_X_train.bin"));
    let y_train_img = load_bin_f32_host(&image_path(s, "image_y_train.bin"));
    let x_val_img = load_bin_f32_host(&image_path(s, "image_X_val.bin"));
    let y_val_img = load_bin_f32_host(&image_path(s, "image_y_val.bin"));
    let x_test_img = load_bin_f32_host(&image_path(s, "image_X_test.bin"));

    // Read the test image names from sample_submission.csv once
    let sample_sub_path = get_kaggle_path("scene_style/sample_submission.csv");
    let sample_sub_content = std::fs::read_to_string(&sample_sub_path).unwrap();
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

    let cnn = StyleCnn::new(s).to(device);

    let ckp_path = get_kaggle_path("style_cnn_best.ve");
    if std::path::Path::new(&ckp_path).exists() {
        println!("   --> Found existing weights checkpoint at {}, loading...", ckp_path);
        if let Err(e) = vearo::checkpoint::load_checkpoint(&cnn.parameters(), &ckp_path) {
            eprintln!("warning: failed to load weights checkpoint: {e}");
        }
    }

    // Base LR = 0.0005, Weight Decay = 0.02
    let mut current_lr = 0.0005f32;
    let mut opt_img =
        vearo::optim::AdamW::new(cnn.parameters(), current_lr, 0.9, 0.999, 1e-8, 0.02);

    let img_batch_size = 256;
    let img_train_size = 10530;
    let mut rng_seed = 42u64;

    let mut best_val_acc = -1.0f32;
    // Live dashboard in a terminal; plain one-line-per-epoch when piped to a log.
    // Also stream to JSONL so `vearo-watch` can follow the run live, including
    // when it is running headless on a remote box under nohup.
    let metrics_path =
        std::env::var("VEARO_METRICS").unwrap_or_else(|_| "runs/style_cnn.jsonl".to_string());
    if let Some(dir) = std::path::Path::new(&metrics_path).parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let num_epochs = std::env::var("VEARO_IMG_EPOCHS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(35);

    let mut ui = vearo::tui::TrainingMonitor::new("style cnn", &format!("{device:?}"), num_epochs)
        .with_metrics(&metrics_path);

    let start_cnn = Instant::now();
    for epoch in 0..num_epochs {
        // Learning Rate Decay Schedule: reduce LR by 0.5 every 10 epochs
        if epoch > 0 && epoch % 10 == 0 {
            current_lr *= 0.5;
            opt_img.set_lr(current_lr);
            println!("   --> Decaying Learning Rate to: {:.8}", current_lr);
        }

        vearo::set_training(true);
        let mut epoch_loss = 0.0;
        let mut batches = 0;

        for i in (0..img_train_size).step_by(img_batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + img_batch_size, img_train_size);
            let size = end_idx - i;

            // Load batch slice and apply host-side data augmentation (stronger +/- 4 shifts)
            let mut x_batch_vec = x_train_img[i * px(s)..end_idx * px(s)].to_vec();
            augment_batch(&mut x_batch_vec, size, &mut rng_seed, s);

            let x_batch_raw = Tensor::from_f32(&x_batch_vec, vec![size, px(s)]).to(device);
            let x_batch = x_batch_raw.reshape(vec![size, 3, s, s]);
            let y_batch = Tensor::from_f32(&y_train_img[i * 1..end_idx * 1], vec![size]).to(device);

            let pred = cnn.forward(&x_batch);
            let loss = pred.cross_entropy(&y_batch);

            epoch_loss += loss.to_vec_f32()[0];
            batches += 1;

            loss.backward();
            opt_img.step();
        }

        // Run validation check in batches
        vearo::set_training(false);
        let mut val_loss_sum = 0.0;
        let mut val_batches = 0;
        let mut correct = 0;
        let val_batch_size = 256;
        let val_size = 2633;

        for i in (0..val_size).step_by(val_batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + val_batch_size, val_size);
            let size = end_idx - i;

            let x_batch_raw =
                Tensor::from_f32(&x_val_img[i * px(s)..end_idx * px(s)], vec![size, px(s)])
                    .to(device);
            let x_batch = x_batch_raw.reshape(vec![size, 3, s, s]);
            let y_batch = Tensor::from_f32(&y_val_img[i * 1..end_idx * 1], vec![size]).to(device);

            let pred = cnn.forward(&x_batch);
            let loss = pred.cross_entropy(&y_batch);

            val_loss_sum += loss.to_vec_f32()[0];
            val_batches += 1;

            let pred_vec = pred.to_vec_f32();
            let y_vec = y_batch.to_vec_f32();
            for s in 0..size {
                let mut max_idx = 0;
                let mut max_val = -1e30f32;
                for c in 0..17 {
                    let val = pred_vec[s * 17 + c];
                    if val > max_val {
                        max_val = val;
                        max_idx = c;
                    }
                }
                if max_idx == y_vec[s] as usize {
                    correct += 1;
                }
            }
        }
        let val_acc = correct as f32 / val_size as f32;
        let val_loss = val_loss_sum / val_batches as f32;

        // Live device slots, for the leak watch in the dashboard. Zero on a
        // CPU-only build, which holds no device slots at all.
        #[cfg(feature = "cuda")]
        let (slots_len, free_len) = (
            vearo::backend_cuda::CUDA_SLOTS.lock().unwrap().len(),
            vearo::backend_cuda::FREE_CUDA_SLOTS.lock().unwrap().len(),
        );
        #[cfg(not(feature = "cuda"))]
        let (slots_len, free_len) = (0usize, 0usize);
        ui.update(
            epoch + 1,
            epoch_loss / batches as f32,
            Some(val_loss),
            Some(val_acc),
        );
        let _ = slots_len - free_len;

        // Fail-safe Best Checkpoint Selection: Run test inference and *immediately* write it to disk.
        if val_acc > best_val_acc {
            best_val_acc = val_acc;
            ui.set_note(&format!(
                "new best val acc {:.2}% - running test inference and saving submission",
                val_acc * 100.0
            ));

            let test_size = 5482;
            let test_batch_size = 256;
            let mut current_img_preds = Vec::with_capacity(test_size * 17);

            for i in (0..test_size).step_by(test_batch_size) {
                vearo::autograd::zero_gradients();
                vearo::autograd::reset_active_tape();

                let end_idx = std::cmp::min(i + test_batch_size, test_size);
                let size = end_idx - i;

                let x_batch_raw =
                    Tensor::from_f32(&x_test_img[i * px(s)..end_idx * px(s)], vec![size, px(s)])
                        .to(device);
                let x_batch = x_batch_raw.reshape(vec![size, 3, s, s]);

                let pred = cnn.forward(&x_batch).to(Device::Cpu).to_vec_f32();
                current_img_preds.extend(pred);
            }

            // Immediately write the best predictions to scene_style_submission.csv
            let img_sub_path = get_kaggle_path("scene_style_submission.csv");
            let mut file_img = File::create(&img_sub_path).unwrap();
            writeln!(file_img, "ImageName,ClassLabel").unwrap();
            for s in 0..5482 {
                let mut max_idx = 0;
                let mut max_val = -1e30f32;
                for c in 0..17 {
                    let val = current_img_preds[s * 17 + c];
                    if val > max_val {
                        max_val = val;
                        max_idx = c;
                    }
                }
                writeln!(file_img, "{},{}", image_names[s], max_idx).unwrap();
            }

            // Save weight checkpoint
            let ckp_path = get_kaggle_path("style_cnn_best.ve");
            if let Err(e) = vearo::checkpoint::save_checkpoint(&cnn.parameters(), &ckp_path) {
                eprintln!("warning: could not save weights checkpoint: {e}");
            }

            println!("   --> Saved peak performance submission to disk!");
        }
    }
    ui.finish();
    println!(
        "CNN Training completed in {:.4} seconds.",
        start_cnn.elapsed().as_secs_f64()
    );
    println!("CNN Peak Validation Accuracy: {:.2}%", best_val_acc * 100.0);

    #[cfg(feature = "cuda")]
    {
        let peak_vram = vearo::backend_cuda::get_peak_memory();
        println!(
            "PEAK CUDA VRAM ALLOCATED: {:.3} MB",
            peak_vram as f64 / 1024.0 / 1024.0
        );
    }
}
