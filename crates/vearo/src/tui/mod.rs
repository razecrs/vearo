//! A dependency-free terminal dashboard for training runs.
//!
//! Renders a live, in-place view of progress, losses, accuracy, memory and ETA
//! using nothing but ANSI escapes. No external crates.
//!
//! Charts are drawn on a braille canvas (see [`canvas`]), which packs a 2x4 grid
//! of dots into every character cell and so gives eight times the resolution of
//! a block-character plot.
//!
//! # Degrading gracefully
//!
//! Three rendering modes, picked automatically:
//!
//! - **full**: interactive terminal, Unicode and truecolor
//! - **ascii**: interactive but `VEARO_TUI=ascii`, a non-UTF-8 locale, or
//!   `NO_COLOR` / `TERM=dumb`
//! - **plain**: stdout is not a terminal (piped, or under `nohup`), so it emits
//!   one greppable line per epoch instead of escape codes
//!
//! ```no_run
//! use vearo::tui::TrainingMonitor;
//!
//! let mut ui = TrainingMonitor::new("style cnn", "Cuda(0)", 75);
//! for epoch in 1..=75 {
//!     // ... train ...
//!     ui.update(epoch, 2.4, Some(2.39), Some(0.228));
//! }
//! ui.finish();
//! ```

// This module only formats numbers for display; lossy casts are intentional and
// cannot affect training results.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

pub mod canvas;

use canvas::Braille;
use std::fmt::Write as _;
use std::io::{IsTerminal, Write};
use std::time::Instant;

/// Inner width of each chart panel, in characters.
const PANEL_W: usize = 34;
/// Height of each chart panel's plot area, in lines.
const PANEL_H: usize = 6;
/// Total dashboard width: two panels (`PANEL_W` + 4 each) plus the 2-space gap.
const TOTAL_W: usize = (PANEL_W + 4) * 2 + 2;
/// Width of the progress bar, in characters.
const BAR_W: usize = 46;
/// How many samples the memory sparklines keep on screen.
const SPARK_N: usize = 24;

// ---------------------------------------------------------------------------
// theme
// ---------------------------------------------------------------------------

type Rgb = (u8, u8, u8);

const ACCENT: Rgb = (94, 234, 212); // teal
const ACCENT2: Rgb = (167, 139, 250); // violet
const GOOD: Rgb = (74, 222, 128); // green
const BAD: Rgb = (248, 113, 113); // red
const DIM: Rgb = (100, 116, 139); // slate
const TEXT: Rgb = (226, 232, 240); // near white

/// Which glyph set and colour support the terminal gets.
struct Theme {
    unicode: bool,
    color: bool,
}

impl Theme {
    /// Detects terminal capabilities from the environment.
    fn detect() -> Self {
        let forced_ascii =
            std::env::var("VEARO_TUI").is_ok_and(|v| v.eq_ignore_ascii_case("ascii"));
        let dumb = std::env::var("TERM").is_ok_and(|t| t == "dumb");
        let utf8 = ["LC_ALL", "LC_CTYPE", "LANG"]
            .iter()
            .filter_map(|k| std::env::var(k).ok())
            .any(|v| v.to_uppercase().contains("UTF-8") || v.to_uppercase().contains("UTF8"));
        Self {
            unicode: !forced_ascii && !dumb && utf8,
            color: !forced_ascii && !dumb && std::env::var("NO_COLOR").is_err(),
        }
    }

    /// Wraps `s` in a truecolor escape, or returns it unchanged without colour.
    fn paint(&self, s: &str, c: Rgb) -> String {
        if self.color {
            format!("\x1b[38;2;{};{};{}m{s}\x1b[0m", c.0, c.1, c.2)
        } else {
            s.to_string()
        }
    }

    /// Like [`Theme::paint`], but bold.
    fn bold(&self, s: &str, c: Rgb) -> String {
        if self.color {
            format!("\x1b[1;38;2;{};{};{}m{s}\x1b[0m", c.0, c.1, c.2)
        } else {
            s.to_string()
        }
    }

    // Box-drawing glyphs, with ASCII stand-ins.
    const fn tl(&self) -> &'static str {
        if self.unicode { "\u{256d}" } else { "+" }
    }
    const fn tr(&self) -> &'static str {
        if self.unicode { "\u{256e}" } else { "+" }
    }
    const fn bl(&self) -> &'static str {
        if self.unicode { "\u{2570}" } else { "+" }
    }
    const fn br(&self) -> &'static str {
        if self.unicode { "\u{256f}" } else { "+" }
    }
    const fn h(&self) -> &'static str {
        if self.unicode { "\u{2500}" } else { "-" }
    }
    const fn v(&self) -> &'static str {
        if self.unicode { "\u{2502}" } else { "|" }
    }
    const fn bar_full(&self) -> &'static str {
        if self.unicode { "\u{2588}" } else { "#" }
    }
    const fn bar_empty(&self) -> &'static str {
        if self.unicode { "\u{2591}" } else { "." }
    }
    const fn up(&self) -> &'static str {
        if self.unicode { "\u{25b2}" } else { "^" }
    }
    const fn down(&self) -> &'static str {
        if self.unicode { "\u{25bc}" } else { "v" }
    }
}

/// Block characters for sparklines, lowest to highest.
const BLOCKS: [&str; 8] = [
    "\u{2581}", "\u{2582}", "\u{2583}", "\u{2584}", "\u{2585}", "\u{2586}", "\u{2587}", "\u{2588}",
];

// ---------------------------------------------------------------------------
// monitor
// ---------------------------------------------------------------------------

/// One recorded epoch.
#[derive(Clone, Copy)]
struct Point {
    train_loss: f32,
    val_loss: Option<f32>,
    val_acc: Option<f32>,
    vram_mb: Option<usize>,
    ram_gb: f64,
}

/// A live training dashboard.
pub struct TrainingMonitor {
    title: String,
    device: String,
    total_epochs: usize,
    history: Vec<Point>,
    best_acc: f32,
    best_epoch: usize,
    start: Instant,
    last_tick: Instant,
    epoch_secs: f64,
    interactive: bool,
    theme: Theme,
    lines_drawn: usize,
    note: String,
    sink: Option<crate::metrics::MetricsSink>,
}

impl TrainingMonitor {
    /// Creates a monitor for a run of `total_epochs` epochs.
    #[must_use]
    pub fn new(title: &str, device: &str, total_epochs: usize) -> Self {
        let now = Instant::now();
        Self {
            title: title.to_string(),
            device: device.to_string(),
            total_epochs,
            history: Vec::new(),
            best_acc: f32::NEG_INFINITY,
            best_epoch: 0,
            start: now,
            last_tick: now,
            epoch_secs: 0.0,
            interactive: std::io::stdout().is_terminal(),
            theme: Theme::detect(),
            lines_drawn: 0,
            note: String::new(),
            sink: None,
        }
    }

    /// Also stream metrics to `path` as JSONL, for `vearo-watch` or any other
    /// reader to tail.
    ///
    /// A failure to open the file is reported and then ignored: a dashboard that
    /// cannot be written to must never take a training run down with it.
    #[must_use]
    pub fn with_metrics(mut self, path: impl AsRef<std::path::Path>) -> Self {
        let p = path.as_ref();
        match crate::metrics::MetricsSink::create(p, &self.title, &self.device, self.total_epochs) {
            Ok(s) => {
                if !self.interactive {
                    println!("streaming metrics to {}", p.display());
                }
                self.sink = Some(s);
            }
            Err(e) => eprintln!("warning: could not open metrics file {}: {e}", p.display()),
        }
        self
    }

    /// Sets the status line shown inside the dashboard.
    ///
    /// Use this instead of `println!` during a run: a raw print would scroll the
    /// terminal and tear the in-place redraw apart. In non-interactive mode it is
    /// printed as an ordinary line.
    pub fn set_note(&mut self, note: &str) {
        if let Some(sink) = self.sink.as_mut() {
            let _ = sink.note(note);
        }
        if self.interactive {
            self.note = note.to_string();
        } else {
            println!("  {note}");
        }
    }

    /// Records an epoch and redraws. `epoch` is 1-based.
    ///
    /// `val_loss` and `val_acc` are optional so you can validate every N epochs.
    pub fn update(
        &mut self,
        epoch: usize,
        train_loss: f32,
        val_loss: Option<f32>,
        val_acc: Option<f32>,
    ) {
        self.epoch_secs = self.last_tick.elapsed().as_secs_f64();
        self.last_tick = Instant::now();

        let point = Point {
            train_loss,
            val_loss,
            val_acc,
            vram_mb: peak_vram_mb(),
            ram_gb: host_rss_gb(),
        };
        self.history.push(point);

        let elapsed = self.start.elapsed().as_secs_f64();
        if let Some(sink) = self.sink.as_mut() {
            let _ = sink.epoch(
                epoch,
                train_loss,
                val_loss,
                val_acc,
                point.vram_mb,
                point.ram_gb,
                elapsed,
            );
        }

        if let Some(acc) = val_acc
            && acc > self.best_acc
        {
            self.best_acc = acc;
            self.best_epoch = epoch;
        }

        if self.interactive {
            self.draw(epoch);
        } else {
            Self::draw_plain(epoch, self.total_epochs, train_loss, val_loss, val_acc);
        }
    }

    /// Prints a final summary.
    pub fn finish(&mut self) {
        let elapsed = self.start.elapsed().as_secs_f64();
        if let Some(sink) = self.sink.as_mut() {
            let _ = sink.done(self.best_acc, self.best_epoch, elapsed);
        }
        if !self.interactive {
            println!(
                "\n{} finished: {} epochs in {} | best val acc {:.2}% at epoch {}",
                self.title,
                self.history.len(),
                fmt_dur(elapsed),
                self.best_acc * 100.0,
                self.best_epoch
            );
            return;
        }

        let t = &self.theme;
        let peak_vram = self
            .history
            .iter()
            .filter_map(|p| p.vram_mb)
            .max()
            .map_or_else(|| "-".to_string(), |m| format!("{m} MiB"));
        let host_peak = self.history.iter().map(|p| p.ram_gb).fold(0.0, f64::max);

        let body = [
            format!(
                "{} epochs in {}   ({:.1}s/epoch)",
                self.history.len(),
                fmt_dur(elapsed),
                elapsed / self.history.len().max(1) as f64
            ),
            format!(
                "best val acc {:.2}% at epoch {}",
                self.best_acc * 100.0,
                self.best_epoch
            ),
            format!("peak vram {peak_vram}   peak host ram {host_peak:.1} GB"),
        ];

        println!();
        println!(
            "{}",
            t.paint(
                &format!("{}{}{}", t.tl(), t.h().repeat(TOTAL_W), t.tr()),
                ACCENT
            )
        );
        println!(
            "{} {} {}",
            t.paint(t.v(), ACCENT),
            t.bold(&pad(&format!("{} complete", self.title), TOTAL_W - 2), GOOD),
            t.paint(t.v(), ACCENT)
        );
        for line in &body {
            println!(
                "{} {} {}",
                t.paint(t.v(), ACCENT),
                t.paint(&pad(line, TOTAL_W - 2), TEXT),
                t.paint(t.v(), ACCENT)
            );
        }
        println!(
            "{}",
            t.paint(
                &format!("{}{}{}", t.bl(), t.h().repeat(TOTAL_W), t.br()),
                ACCENT
            )
        );
    }

    fn draw_plain(
        epoch: usize,
        total: usize,
        train_loss: f32,
        val_loss: Option<f32>,
        val_acc: Option<f32>,
    ) {
        let mut line = format!("epoch {epoch:>3}/{total} | train {train_loss:.6}");
        if let Some(v) = val_loss {
            let _ = write!(line, " | val {v:.6}");
        }
        if let Some(a) = val_acc {
            let _ = write!(line, " | acc {:.2}%", a * 100.0);
        }
        println!("{line}");
    }

    fn draw(&mut self, epoch: usize) {
        let t = &self.theme;
        let mut out = String::with_capacity(8192);

        // Move back over the previous frame rather than clearing the screen, so
        // scrollback survives. \x1b[K wipes each line's tail, so a shorter frame
        // cannot leave debris behind.
        if self.lines_drawn > 0 {
            let _ = write!(out, "\x1b[{}A", self.lines_drawn);
        }
        let mut push = |s: &str| {
            out.push_str(s);
            out.push_str("\x1b[K\n");
        };

        let elapsed = self.start.elapsed().as_secs_f64();
        let remaining = (elapsed / epoch as f64) * (self.total_epochs.saturating_sub(epoch)) as f64;
        let frac = epoch as f64 / self.total_epochs as f64;
        let last = self.history[self.history.len() - 1];
        let prev = if self.history.len() >= 2 {
            Some(self.history[self.history.len() - 2])
        } else {
            None
        };

        for line in self.head_lines(epoch, frac, remaining) {
            push(&line);
        }

        // ---- charts ----------------------------------------------------------
        let train: Vec<f32> = self.history.iter().map(|p| p.train_loss).collect();
        let val: Vec<f32> = self.history.iter().filter_map(|p| p.val_loss).collect();
        let accs: Vec<f32> = self.history.iter().filter_map(|p| p.val_acc).collect();

        let loss_panel = self.panel("loss", &[&train, &val], &[ACCENT, ACCENT2]);
        let acc_panel = self.panel("val accuracy", &[&accs], &[GOOD]);
        for (a, b) in loss_panel.iter().zip(acc_panel.iter()) {
            push(&format!(" {a}  {b}"));
        }

        // ---- metrics under each chart ---------------------------------------
        let d_train = prev.map_or(0.0, |p| last.train_loss - p.train_loss);
        let loss_line = format!(
            "  {} {} {}  {} {}",
            t.paint("train", DIM),
            t.bold(&format!("{:.4}", last.train_loss), ACCENT),
            self.delta(d_train, true),
            t.paint("val", DIM),
            last.val_loss.map_or_else(
                || t.paint("-", DIM),
                |v| t.bold(&format!("{v:.4}"), ACCENT2)
            ),
        );
        let acc_line = format!(
            "  {} {}   {} {}",
            t.paint("now", DIM),
            last.val_acc.map_or_else(
                || t.paint("-", DIM),
                |a| t.bold(&format!("{:.2}%", a * 100.0), GOOD)
            ),
            t.paint("best", DIM),
            if self.best_acc.is_finite() {
                t.bold(
                    &format!("{:.2}% (ep {})", self.best_acc * 100.0, self.best_epoch),
                    TEXT,
                )
            } else {
                t.paint("-", DIM)
            }
        );
        push(&format!(
            " {}{}",
            pad_visible(&loss_line, PANEL_W + 4),
            acc_line
        ));
        push("");

        for line in self.foot_lines(last, elapsed) {
            push(&line);
        }

        self.lines_drawn = out.matches('\n').count();
        print!("{out}");
        let _ = std::io::stdout().flush();
    }

    /// Title bar, rule, and the gradient progress bar with ETA.
    fn head_lines(&self, epoch: usize, frac: f64, remaining: f64) -> Vec<String> {
        let t = &self.theme;

        let left = format!("{} {}", t.bold("VEARO", ACCENT), t.paint(&self.title, TEXT));
        let right = format!(
            "{} {}",
            t.paint(&self.device, ACCENT2),
            t.bold(&format!("epoch {epoch}/{}", self.total_epochs), TEXT)
        );
        let gap = TOTAL_W
            .saturating_sub(visible_len(&left) + visible_len(&right))
            .max(1);

        let filled = ((frac * BAR_W as f64).round() as usize).min(BAR_W);
        let mut bar = String::new();
        for i in 0..BAR_W {
            if i < filled {
                // gradient violet -> teal across the filled span
                bar.push_str(
                    &t.paint(t.bar_full(), lerp(ACCENT2, ACCENT, i as f32 / BAR_W as f32)),
                );
            } else {
                bar.push_str(&t.paint(t.bar_empty(), DIM));
            }
        }

        vec![
            format!(" {left}{}{right}", " ".repeat(gap)),
            format!(" {}", t.paint(&t.h().repeat(TOTAL_W), DIM)),
            format!(
                " {bar} {}  {} {}",
                t.bold(&format!("{:>3.0}%", frac * 100.0), TEXT),
                t.paint("eta", DIM),
                t.paint(&fmt_dur(remaining), TEXT)
            ),
            String::new(),
        ]
    }

    /// Memory sparklines, timing, and the status note.
    fn foot_lines(&self, last: Point, elapsed: f64) -> Vec<String> {
        let t = &self.theme;

        // A flat sparkline here is the point: it is what a leak-free run looks
        // like. Growth shows up long before an OOM kill does.
        let vram_hist: Vec<f32> = self
            .history
            .iter()
            .filter_map(|p| p.vram_mb.map(|m| m as f32))
            .collect();
        let ram_hist: Vec<f32> = self.history.iter().map(|p| p.ram_gb as f32).collect();

        let vram_txt = last
            .vram_mb
            .map_or_else(|| "-".to_string(), |m| format!("{m} MiB"));
        let mem_left = format!(
            "  {} {} {}",
            t.paint("vram", DIM),
            self.spark(&vram_hist, ACCENT),
            t.bold(&vram_txt, TEXT)
        );
        let mem_right = format!(
            "  {} {} {}",
            t.paint("ram", DIM),
            self.spark(&ram_hist, ACCENT2),
            t.bold(&format!("{:.1} GB", last.ram_gb), TEXT)
        );

        vec![
            format!(" {}{}", pad_visible(&mem_left, PANEL_W + 4), mem_right),
            format!(
                "  {} {}   {} {}",
                t.paint("elapsed", DIM),
                t.paint(&fmt_dur(elapsed), TEXT),
                t.paint("per epoch", DIM),
                t.paint(&format!("{:.1}s", self.epoch_secs), TEXT),
            ),
            if self.note.is_empty() {
                String::new()
            } else {
                let n: String = self.note.chars().take(TOTAL_W - 6).collect();
                format!("  {} {}", t.paint("*", GOOD), t.paint(&n, GOOD))
            },
        ]
    }

    /// Renders one bordered chart panel with any number of overlaid series.
    fn panel(&self, title: &str, series: &[&[f32]], colors: &[Rgb]) -> Vec<String> {
        let t = &self.theme;
        let mut lines = Vec::with_capacity(PANEL_H + 2);

        // top border with an inline title
        let head = format!("{}{} {title} ", t.tl(), t.h());
        let used = visible_len(&head);
        lines.push(t.paint(
            &format!(
                "{head}{}{}",
                t.h().repeat((PANEL_W + 3).saturating_sub(used)),
                t.tr()
            ),
            DIM,
        ));

        let body = if self.theme.unicode {
            self.braille_rows(series)
        } else {
            ascii_rows(series.first().copied().unwrap_or(&[]))
        };

        // Overlaid series share one canvas per colour, so each rendered layer is
        // painted separately and then merged character by character.
        for row in body {
            lines.push(format!(
                "{} {} {}",
                t.paint(t.v(), DIM),
                colorize_row(&row, colors, t),
                t.paint(t.v(), DIM)
            ));
        }

        lines.push(t.paint(
            &format!("{}{}{}", t.bl(), t.h().repeat(PANEL_W + 2), t.br()),
            DIM,
        ));
        lines
    }

    /// Rasterises each series onto its own canvas and stacks them per row.
    fn braille_rows(&self, series: &[&[f32]]) -> Vec<Vec<(char, usize)>> {
        let _ = self;
        // Shared scale across series so overlaid curves stay comparable.
        let (lo, hi) = series
            .iter()
            .flat_map(|s| s.iter())
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });

        let layers: Vec<Vec<String>> = series
            .iter()
            .map(|s| {
                let mut c = Braille::new(PANEL_W, PANEL_H);
                if !s.is_empty() && lo.is_finite() {
                    let scaled: Vec<f32> = if (hi - lo).abs() < f32::EPSILON {
                        s.iter().map(|_| 0.5).collect()
                    } else {
                        s.iter().map(|v| (v - lo) / (hi - lo)).collect()
                    };
                    c.plot(&scaled);
                }
                c.rows()
            })
            .collect();

        (0..PANEL_H)
            .map(|r| {
                (0..PANEL_W)
                    .map(|col| {
                        // topmost non-blank layer wins the cell
                        layers
                            .iter()
                            .enumerate()
                            .find_map(|(li, rows)| {
                                let ch = rows[r].chars().nth(col).unwrap_or('\u{2800}');
                                (ch != '\u{2800}').then_some((ch, li))
                            })
                            .unwrap_or((' ', 0))
                    })
                    .collect()
            })
            .collect()
    }

    /// A compact block-character sparkline scaled from zero, so a flat series
    /// renders as a flat line rather than noise amplified to full height.
    fn spark(&self, values: &[f32], color: Rgb) -> String {
        let t = &self.theme;
        if values.is_empty() {
            return t.paint(&"-".repeat(SPARK_N / 2), DIM);
        }
        let tail = &values[values.len().saturating_sub(SPARK_N)..];
        let hi = tail.iter().copied().fold(0.0f32, f32::max) * 1.15;
        let s: String = tail
            .iter()
            .map(|&v| {
                if hi <= 0.0 {
                    BLOCKS[0]
                } else {
                    let idx = ((v / hi) * (BLOCKS.len() - 1) as f32).round() as usize;
                    BLOCKS[idx.min(BLOCKS.len() - 1)]
                }
            })
            .collect();
        if t.unicode {
            t.paint(&s, color)
        } else {
            t.paint(&"=".repeat(tail.len()), color)
        }
    }

    /// A coloured delta marker. `lower_is_better` flips which direction is green.
    fn delta(&self, d: f32, lower_is_better: bool) -> String {
        let t = &self.theme;
        if d.abs() < 1e-6 {
            return t.paint("-", DIM);
        }
        let improving = (d < 0.0) == lower_is_better;
        let arrow = if d < 0.0 { t.down() } else { t.up() };
        t.paint(
            &format!("{arrow}{:.4}", d.abs()),
            if improving { GOOD } else { BAD },
        )
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Paints each cell of a rasterised chart row in its series' colour.
fn colorize_row(row: &[(char, usize)], colors: &[Rgb], t: &Theme) -> String {
    let mut out = String::with_capacity(row.len() * 12);
    for &(ch, layer) in row {
        if ch == ' ' {
            out.push(' ');
        } else {
            out.push_str(&t.paint(&ch.to_string(), *colors.get(layer).unwrap_or(&TEXT)));
        }
    }
    out
}

/// ASCII fallback plot, used when the terminal cannot render braille.
fn ascii_rows(values: &[f32]) -> Vec<Vec<(char, usize)>> {
    if values.is_empty() {
        return vec![vec![(' ', 0); PANEL_W]; PANEL_H];
    }
    let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = if (hi - lo).abs() < f32::EPSILON {
        1.0
    } else {
        hi - lo
    };

    // Downsample to the panel width by bucket maximum.
    let n = PANEL_W.min(values.len());
    let mut cols = vec![f32::NEG_INFINITY; n];
    for (i, v) in values.iter().enumerate() {
        let c = ((i * n) / values.len()).min(n - 1);
        cols[c] = cols[c].max(*v);
    }

    (0..PANEL_H)
        .map(|row| {
            let top = 1.0 - (row as f32 / PANEL_H as f32);
            let bot = 1.0 - ((row + 1) as f32 / PANEL_H as f32);
            (0..PANEL_W)
                .map(|c| {
                    cols.get(c).map_or((' ', 0), |&v| {
                        let norm = (v - lo) / span;
                        if norm >= bot && norm < top + f32::EPSILON {
                            ('*', 0)
                        } else if norm >= top {
                            ('|', 0)
                        } else {
                            (' ', 0)
                        }
                    })
                })
                .collect()
        })
        .collect()
}

/// Linearly interpolates between two colours.
fn lerp(a: Rgb, b: Rgb, t: f32) -> Rgb {
    let f = |x: u8, y: u8| (f32::from(y) - f32::from(x)).mul_add(t, f32::from(x)) as u8;
    (f(a.0, b.0), f(a.1, b.1), f(a.2, b.2))
}

/// Length of `s` ignoring ANSI escape sequences.
fn visible_len(s: &str) -> usize {
    let mut n = 0;
    let mut in_esc = false;
    for ch in s.chars() {
        if in_esc {
            if ch == 'm' {
                in_esc = false;
            }
        } else if ch == '\x1b' {
            in_esc = true;
        } else {
            n += 1;
        }
    }
    n
}

/// Right-pads a plain string to `w` characters.
fn pad(s: &str, w: usize) -> String {
    let mut out = s.to_string();
    for _ in visible_len(s)..w {
        out.push(' ');
    }
    out
}

/// Right-pads a string that may contain escape sequences to `w` visible columns.
fn pad_visible(s: &str, w: usize) -> String {
    pad(s, w)
}

/// Peak CUDA memory in MiB, if the CUDA backend is active.
///
/// A CPU-only build has no VRAM to report, so the dashboard shows a dash rather
/// than a misleading zero.
#[cfg(feature = "cuda")]
fn peak_vram_mb() -> Option<usize> {
    let bytes = crate::backend_cuda::get_peak_memory();
    if bytes == 0 {
        None
    } else {
        Some(bytes / (1024 * 1024))
    }
}

/// Peak CUDA memory in MiB. Always `None` without the `cuda` feature.
#[cfg(not(feature = "cuda"))]
const fn peak_vram_mb() -> Option<usize> {
    None
}

/// Resident set size of this process, in GB.
fn host_rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/statm")
        .ok()
        .and_then(|s| s.split_whitespace().nth(1).map(str::to_string))
        .and_then(|p| p.parse::<u64>().ok())
        .map_or(0.0, |pages| (pages * 4096) as f64 / 1_073_741_824.0)
}

/// Formats seconds as `H:MM:SS` or `M:SS`.
fn fmt_dur(secs: f64) -> String {
    let s = secs.max(0.0) as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, (s % 3600) / 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}
