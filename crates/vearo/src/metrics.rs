//! A newline-delimited JSON sink for training metrics.
//!
//! One JSON object per line: a `run` record when training starts, then an `epoch`
//! record per epoch. Readers tail the file as it grows.
//!
//! JSONL rather than a socket or a binary format because a training run is often
//! headless and remote. A plain appended file survives `nohup`, survives the
//! process being killed, can be `tail -f`'d over ssh, rsynced while still being
//! written, and needs no reader to be listening when the run starts.
//!
//! The encoder here is hand-written so the training crate keeps no runtime
//! serialisation dependency.
//!
//! ```no_run
//! use vearo::metrics::MetricsSink;
//!
//! let mut sink = MetricsSink::create("run.jsonl", "style cnn", "Cuda(0)", 75).unwrap();
//! sink.epoch(1, 2.77, Some(2.74), Some(0.108), Some(657), 1.0, 41.2);
//! ```

use std::fmt::Write as _;
use std::fs::File;
use std::io::{BufWriter, Result as IoResult, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Appends training metrics to a JSONL file.
pub struct MetricsSink {
    out: BufWriter<File>,
}

impl MetricsSink {
    /// Creates (or truncates) `path` and writes the run header record.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be created or the header cannot be
    /// written.
    pub fn create(
        path: impl AsRef<Path>,
        title: &str,
        device: &str,
        total_epochs: usize,
    ) -> IoResult<Self> {
        let mut sink = Self {
            out: BufWriter::new(File::create(path)?),
        };
        let line = format!(
            r#"{{"type":"run","title":{},"device":{},"total_epochs":{},"started":{}}}"#,
            json_str(title),
            json_str(device),
            total_epochs,
            now_ms()
        );
        sink.write_line(&line)?;
        Ok(sink)
    }

    /// Records one epoch.
    ///
    /// # Errors
    ///
    /// Returns an error if the record cannot be written.
    #[allow(clippy::too_many_arguments)]
    pub fn epoch(
        &mut self,
        epoch: usize,
        train_loss: f32,
        val_loss: Option<f32>,
        val_acc: Option<f32>,
        vram_mb: Option<usize>,
        ram_gb: f64,
        elapsed_s: f64,
    ) -> IoResult<()> {
        let line = format!(
            r#"{{"type":"epoch","epoch":{epoch},"train_loss":{train_loss},"val_loss":{},"val_acc":{},"vram_mb":{},"ram_gb":{ram_gb:.4},"elapsed_s":{elapsed_s:.3},"ts":{}}}"#,
            json_f32(val_loss),
            json_f32(val_acc),
            json_usize(vram_mb),
            now_ms()
        );
        self.write_line(&line)
    }

    /// Records a free-text note, such as a new best checkpoint.
    ///
    /// # Errors
    ///
    /// Returns an error if the record cannot be written.
    pub fn note(&mut self, text: &str) -> IoResult<()> {
        let line = format!(
            r#"{{"type":"note","text":{},"ts":{}}}"#,
            json_str(text),
            now_ms()
        );
        self.write_line(&line)
    }

    /// Records the terminating record so a reader can tell a finished run from a
    /// crashed one.
    ///
    /// # Errors
    ///
    /// Returns an error if the record cannot be written.
    pub fn done(&mut self, best_acc: f32, best_epoch: usize, elapsed_s: f64) -> IoResult<()> {
        let line = format!(
            r#"{{"type":"done","best_acc":{best_acc},"best_epoch":{best_epoch},"elapsed_s":{elapsed_s:.3},"ts":{}}}"#,
            now_ms()
        );
        self.write_line(&line)
    }

    /// Writes a record and flushes.
    ///
    /// Flushing every line is deliberate: a buffered tail of a run that later
    /// crashes would lose exactly the records explaining the crash.
    fn write_line(&mut self, line: &str) -> IoResult<()> {
        self.out.write_all(line.as_bytes())?;
        self.out.write_all(b"\n")?;
        self.out.flush()
    }
}

/// Encodes a `f32` as a JSON number, or `null`. Non-finite values become `null`
/// because `NaN` and `Infinity` are not valid JSON.
fn json_f32(v: Option<f32>) -> String {
    v.filter(|x| x.is_finite())
        .map_or_else(|| "null".to_string(), |x| format!("{x}"))
}

/// Encodes an optional count as a JSON number, or `null`.
fn json_usize(v: Option<usize>) -> String {
    v.map_or_else(|| "null".to_string(), |x| x.to_string())
}

/// Encodes a string as a JSON string literal, escaping what the spec requires.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Milliseconds since the Unix epoch.
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}
