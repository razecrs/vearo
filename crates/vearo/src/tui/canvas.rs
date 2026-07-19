//! A braille-based plotting canvas.
//!
//! Each terminal cell holds a 2x4 grid of braille dots, so a chart of `w` by `h`
//! characters has `2w` by `4h` addressable pixels. That is what makes the curves
//! look smooth instead of blocky.

/// Braille dot bit for each (x, y) offset inside a cell.
/// Layout per the Unicode braille patterns block:
///   (0,0)=0x01  (1,0)=0x08
///   (0,1)=0x02  (1,1)=0x10
///   (0,2)=0x04  (1,2)=0x20
///   (0,3)=0x40  (1,3)=0x80
const DOTS: [[u8; 4]; 2] = [[0x01, 0x02, 0x04, 0x40], [0x08, 0x10, 0x20, 0x80]];

/// A drawing surface addressed in braille sub-pixels.
pub struct Braille {
    cols: usize,
    rows: usize,
    cells: Vec<u8>,
}

impl Braille {
    /// Creates a canvas `cols` characters wide and `rows` characters tall.
    #[must_use]
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            cols,
            rows,
            cells: vec![0; cols * rows],
        }
    }

    /// Pixel width of the canvas.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.cols * 2
    }

    /// Pixel height of the canvas.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.rows * 4
    }

    /// Lights the pixel at `(x, y)`, with `(0, 0)` at the top left.
    pub fn set(&mut self, x: usize, y: usize) {
        if x >= self.width() || y >= self.height() {
            return;
        }
        let (cx, cy) = (x / 2, y / 4);
        self.cells[cy * self.cols + cx] |= DOTS[x % 2][y % 4];
    }

    /// Draws a straight line between two pixels (Bresenham).
    pub fn line(&mut self, x0: isize, y0: isize, x1: isize, y1: isize) {
        let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
        let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
        let (mut x, mut y, mut err) = (x0, y0, dx + dy);
        loop {
            if x >= 0 && y >= 0 {
                self.set(x as usize, y as usize);
            }
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * err;
            if e2 >= dy {
                err += dy;
                x += sx;
            }
            if e2 <= dx {
                err += dx;
                y += sy;
            }
        }
    }

    /// Plots a series, scaling it to fill the canvas. Returns nothing if empty.
    pub fn plot(&mut self, values: &[f32]) {
        if values.len() < 2 {
            if let Some(&v) = values.first() {
                let y = self.height() / 2;
                let _ = v;
                self.set(0, y);
            }
            return;
        }
        let lo = values.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let span = if (hi - lo).abs() < f32::EPSILON {
            1.0
        } else {
            hi - lo
        };

        let px = self.width().saturating_sub(1) as f32;
        let py = self.height().saturating_sub(1) as f32;
        let n = (values.len() - 1) as f32;

        let mut prev: Option<(isize, isize)> = None;
        for (i, &v) in values.iter().enumerate() {
            let x = (i as f32 / n * px).round() as isize;
            // invert: high values at the top
            let y = ((1.0 - (v - lo) / span) * py).round() as isize;
            if let Some((px0, py0)) = prev {
                self.line(px0, py0, x, y);
            }
            prev = Some((x, y));
        }
    }

    /// Renders the canvas to one string per row.
    #[must_use]
    pub fn rows(&self) -> Vec<String> {
        (0..self.rows)
            .map(|r| {
                (0..self.cols)
                    .map(|c| {
                        let bits = self.cells[r * self.cols + c];
                        char::from_u32(0x2800 + u32::from(bits)).unwrap_or(' ')
                    })
                    .collect()
            })
            .collect()
    }
}
