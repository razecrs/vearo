//! Preview the training dashboard without waiting for a real run.
//!
//!     cargo run --release -p vearo --example tui_demo
//!
//! Replays a realistic loss/accuracy curve so you can see the layout. Pipe it to
//! a file to check the non-interactive fallback:
//!
//!     cargo run --release -p vearo --example tui_demo | head
#![allow(clippy::cast_precision_loss)]

use std::thread::sleep;
use std::time::Duration;
use vearo::tui::TrainingMonitor;

fn main() {
    let total = 75;
    let mut ui = TrainingMonitor::new("style cnn (demo)", "Cuda(0)", total);
    // Optional second output: `... --example tui_demo -- run.jsonl` also streams
    // metrics to a file, which is what `vearo-watch` reads.
    if let Some(path) = std::env::args().nth(1) {
        ui = ui.with_metrics(path);
    }

    for epoch in 1..=total {
        let t = epoch as f32 / total as f32;
        // plausible curves: loss decays, accuracy saturates, both a bit noisy
        let noise = ((epoch as f32) * 12.9898).sin() * 0.01;
        let train = 2.80f32.mul_add((-1.6 * t).exp(), 0.55) + noise;
        let val = 2.78f32.mul_add((-1.5 * t).exp(), 0.60) + noise * 1.5;
        let acc = 0.34f32.mul_add(1.0 - (-3.0 * t).exp(), 0.105) + noise;

        ui.update(epoch, train, Some(val), Some(acc));
        sleep(Duration::from_millis(40));
    }
    ui.finish();
}
