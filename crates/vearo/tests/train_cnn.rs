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
fn get_kaggle_path(relative_suffix: &str) -> String {
    if let Ok(dir) = std::env::var("VEARO_DATA_DIR") {
        let p = format!("{dir}/{relative_suffix}");
        if std::path::Path::new(&p).exists() {
            return p;
        }
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
fn augment_batch(batch_data: &mut [f32], size: usize, rng_seed: &mut u64) {
    // Simple LCG RNG
    fn next_rand(seed: &mut u64) -> u32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*seed >> 32) as u32
    }

    for b in 0..size {
        let img_offset = b * 3072;
        let img = &mut batch_data[img_offset..img_offset + 3072];

        // 1. Random Horizontal Flip (50% probability)
        if next_rand(rng_seed) % 2 == 0 {
            for c in 0..3 {
                let c_offset = c * 1024;
                for y in 0..32 {
                    let row_offset = c_offset + y * 32;
                    for x in 0..16 {
                        img.swap(row_offset + x, row_offset + (31 - x));
                    }
                }
            }
        }

        // 2. Random Translation / Shift (up to +/- 4 pixels with zero padding)
        let dx = (next_rand(rng_seed) % 9) as i32 - 4; // -4, -3, -2, -1, 0, 1, 2, 3, 4
        let dy = (next_rand(rng_seed) % 9) as i32 - 4;

        if dx != 0 || dy != 0 {
            let mut temp = vec![0.0f32; 3072];
            for c in 0..3 {
                let c_offset = c * 1024;
                for y in 0..32 {
                    let target_y = y as i32 + dy;
                    if target_y < 0 || target_y >= 32 {
                        continue;
                    }
                    for x in 0..32 {
                        let target_x = x as i32 + dx;
                        if target_x < 0 || target_x >= 32 {
                            continue;
                        }
                        let src_idx = c_offset + y * 32 + x;
                        let dest_idx = c_offset + (target_y as usize) * 32 + (target_x as usize);
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
        self.fc3.forward(&self.fc2.forward(&self.fc1.forward(x).relu()).relu())
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.fc1.parameters();
        params.extend(self.fc2.parameters());
        params.extend(self.fc3.parameters());
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
    fn new() -> Self {
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

            fc1: vearo::nn::Linear::new(2048, 128, true, 45),
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
        let h1 = self.pool1.forward(&self.bn1.forward(&self.conv1.forward(x)).relu());
        let h2 = self.pool2.forward(&self.bn2.forward(&self.conv2.forward(&h1)).relu());
        let h3 = self.pool3.forward(&self.bn3.forward(&self.conv3.forward(&h2)).relu());
        
        let b = x.shape().dims()[0];
        let flat = h3.reshape([b, 2048]);
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
#[ignore = "long training run needing datasets; run scripts/setup_data.sh then: cargo test --release --test train_cnn -- --ignored --nocapture"]
fn test_train_cnn_full() {
    vearo::init();
    let device = Device::Cuda(0);

    // ----------------- 1. Train Tabular Regression Model -----------------
    println!("=== Training Tabular Regression Model ===");
    let x_train_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_train.bin"));
    let y_train_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_y_train.bin"));
    let x_val_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_val.bin"));
    let y_val_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_y_val.bin"));
    let x_test_tab = load_bin_f32_host(&get_kaggle_path("preprocessed/tabular_X_test.bin"));

    let tab_val_inputs = Tensor::from_f32(&x_val_tab, vec![1200, 46]).to(device);
    let tab_val_targets = Tensor::from_f32(&y_val_tab, vec![1200, 1]).to(device);

    let mlp_tab = TabularMlp::new().to(device);
    let mut opt_tab = vearo::optim::AdamW::new(mlp_tab.parameters(), 0.005, 0.9, 0.999, 1e-8, 0.0);

    let tab_batch_size = 128;
    let tab_train_size = 4800;

    let start_tab = Instant::now();
    for epoch in 0..30 {
        let mut epoch_loss = 0.0;
        let mut batches = 0;
        for i in (0..tab_train_size).step_by(tab_batch_size) {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();

            let end_idx = std::cmp::min(i + tab_batch_size, tab_train_size);
            let size = end_idx - i;

            let x_batch = Tensor::from_f32(&x_train_tab[i * 46..end_idx * 46], vec![size, 46]).to(device);
            let y_batch = Tensor::from_f32(&y_train_tab[i * 1..end_idx * 1], vec![size, 1]).to(device);

            let pred = mlp_tab.forward(&x_batch);
            let diff = pred.sub(&y_batch);
            let squared = diff.mul(&diff);
            let loss = squared.mean(0, false);

            epoch_loss += loss.to_vec_f32()[0];
            batches += 1;

            loss.backward();
            opt_tab.step();
        }

        if (epoch + 1) % 10 == 0 || epoch == 0 {
            vearo::autograd::zero_gradients();
            vearo::autograd::reset_active_tape();
            let val_pred = mlp_tab.forward(&tab_val_inputs);
            let val_diff = val_pred.sub(&tab_val_targets);
            let val_squared = val_diff.mul(&val_diff);
            let val_loss = val_squared.mean(0, false).to_vec_f32()[0];

            println!(
                "Tabular Epoch {:02} | Train Loss: {:.6} | Val Loss: {:.6}",
                epoch + 1,
                epoch_loss / batches as f32,
                val_loss
            );
        }
    }
    println!("Tabular Training completed in {:.4} seconds.", start_tab.elapsed().as_secs_f64());

    // Generate Tabular Prediction submission.csv
    println!("Generating tabular submission...");
    vearo::autograd::zero_gradients();
    vearo::autograd::reset_active_tape();
    let x_test_t = Tensor::from_f32(&x_test_tab, vec![2523, 46]).to(device);
    let tab_preds = mlp_tab.forward(&x_test_t).to(Device::Cpu).to_vec_f32();

    let tab_sub_path = get_kaggle_path("item_price_submission.csv");
    let mut file_tab = File::create(&tab_sub_path).unwrap();
    writeln!(file_tab, "row_id,Y").unwrap();
    for (i, val) in tab_preds.iter().enumerate() {
        writeln!(file_tab, "{},{}", i, val).unwrap();
    }
    println!("Saved tabular submission to: {}\n", tab_sub_path);


    // ----------------- 2. Train Image Style CNN Model -----------------
    println!("=== Training Image Style CNN Model ===");
    let x_train_img = load_bin_f32_host(&get_kaggle_path("preprocessed/image_X_train.bin"));
    let y_train_img = load_bin_f32_host(&get_kaggle_path("preprocessed/image_y_train.bin"));
    let x_val_img = load_bin_f32_host(&get_kaggle_path("preprocessed/image_X_val.bin"));
    let y_val_img = load_bin_f32_host(&get_kaggle_path("preprocessed/image_y_val.bin"));
    let x_test_img = load_bin_f32_host(&get_kaggle_path("preprocessed/image_X_test.bin"));

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

    let cnn = StyleCnn::new().to(device);
    
    // Base LR = 0.0005, Weight Decay = 0.02
    let mut current_lr = 0.0005f32;
    let mut opt_img = vearo::optim::AdamW::new(cnn.parameters(), current_lr, 0.9, 0.999, 1e-8, 0.02);

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
    let mut ui = vearo::tui::TrainingMonitor::new("style cnn", &format!("{device:?}"), 75)
        .with_metrics(&metrics_path);

    let start_cnn = Instant::now();
    for epoch in 0..75 {
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
            let mut x_batch_vec = x_train_img[i * 3072..end_idx * 3072].to_vec();
            augment_batch(&mut x_batch_vec, size, &mut rng_seed);

            let x_batch_raw = Tensor::from_f32(&x_batch_vec, vec![size, 3072]).to(device);
            let x_batch = x_batch_raw.reshape(vec![size, 3, 32, 32]);
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

            let x_batch_raw = Tensor::from_f32(&x_val_img[i * 3072..end_idx * 3072], vec![size, 3072]).to(device);
            let x_batch = x_batch_raw.reshape(vec![size, 3, 32, 32]);
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

        let slots_len = vearo::backend_cuda::CUDA_SLOTS.lock().unwrap().len();
        let free_len = vearo::backend_cuda::FREE_CUDA_SLOTS.lock().unwrap().len();
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

                let x_batch_raw = Tensor::from_f32(&x_test_img[i * 3072..end_idx * 3072], vec![size, 3072]).to(device);
                let x_batch = x_batch_raw.reshape(vec![size, 3, 32, 32]);

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
            println!("   --> Saved peak performance submission to disk!");
        }
    }
    ui.finish();
    println!("CNN Training completed in {:.4} seconds.", start_cnn.elapsed().as_secs_f64());
    println!("CNN Peak Validation Accuracy: {:.2}%", best_val_acc * 100.0);

    let peak_vram = vearo::backend_cuda::get_peak_memory();
    println!("PEAK CUDA VRAM ALLOCATED: {:.3} MB", peak_vram as f64 / 1024.0 / 1024.0);
}
