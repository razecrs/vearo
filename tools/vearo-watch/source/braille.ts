/**
 * Braille plotting.
 *
 * Every character cell in the Unicode braille block holds a 2x4 grid of dots, so
 * a chart `w` cells wide has `2w` by `4h` addressable pixels. That is eight times
 * the resolution of a block-character plot, which is what makes a loss curve read
 * as a curve rather than a staircase.
 */

/** Dot bit for each (x, y) offset inside a cell. */
const DOTS = [
	[0x01, 0x02, 0x04, 0x40],
	[0x08, 0x10, 0x20, 0x80],
] as const;

const BLANK = 0x2800;

export class Braille {
	readonly #cols: number;
	readonly #rows: number;
	readonly #cells: Uint8Array;

	constructor(cols: number, rows: number) {
		this.#cols = cols;
		this.#rows = rows;
		this.#cells = new Uint8Array(cols * rows);
	}

	get width(): number {
		return this.#cols * 2;
	}

	get height(): number {
		return this.#rows * 4;
	}

	set(x: number, y: number): void {
		if (x < 0 || y < 0 || x >= this.width || y >= this.height) return;
		const cx = Math.floor(x / 2);
		const cy = Math.floor(y / 4);
		this.#cells[cy * this.#cols + cx]! |= DOTS[x % 2]![y % 4]!;
	}

	/** Bresenham, so segments between samples are continuous. */
	line(x0: number, y0: number, x1: number, y1: number): void {
		const dx = Math.abs(x1 - x0);
		const dy = -Math.abs(y1 - y0);
		const sx = x0 < x1 ? 1 : -1;
		const sy = y0 < y1 ? 1 : -1;
		let err = dx + dy;
		let x = x0;
		let y = y0;
		for (;;) {
			this.set(x, y);
			if (x === x1 && y === y1) break;
			const e2 = 2 * err;
			if (e2 >= dy) {
				err += dy;
				x += sx;
			}
			if (e2 <= dx) {
				err += dx;
				y += sy;
			}
		}
	}

	/**
	 * Plots `values` normalised against an explicit range, so several series
	 * drawn on separate canvases stay comparable.
	 */
	plot(values: number[], lo: number, hi: number): void {
		if (values.length === 0) return;
		const span = hi - lo === 0 ? 1 : hi - lo;
		const px = this.width - 1;
		const py = this.height - 1;

		if (values.length === 1) {
			this.set(0, Math.round((1 - (values[0]! - lo) / span) * py));
			return;
		}

		const n = values.length - 1;
		let prev: [number, number] | undefined;
		for (const [i, v] of values.entries()) {
			const x = Math.round((i / n) * px);
			// inverted: larger values sit higher on screen
			const y = Math.round((1 - (v - lo) / span) * py);
			if (prev) this.line(prev[0], prev[1], x, y);
			prev = [x, y];
		}
	}

	/** One string per row, blanks preserved so rows keep their width. */
	rows(): string[] {
		const out: string[] = [];
		for (let r = 0; r < this.#rows; r++) {
			let line = '';
			for (let c = 0; c < this.#cols; c++) {
				line += String.fromCodePoint(BLANK + this.#cells[r * this.#cols + c]!);
			}
			out.push(line);
		}
		return out;
	}
}

/** Block characters for compact sparklines, lowest to highest. */
const BLOCKS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/**
 * A one-line sparkline scaled from zero rather than from the series minimum.
 *
 * Scaling from zero is the point for memory: a flat allocation renders as a flat
 * line. Scaling to the series range would amplify byte-level jitter into a
 * dramatic sawtooth and make a healthy run look like a leak.
 */
export function sparkline(values: number[], width: number): string {
	if (values.length === 0) return '';
	const tail = values.slice(-width);
	const hi = Math.max(...tail) * 1.15;
	if (hi <= 0) return BLOCKS[0]!.repeat(tail.length);
	return tail
		.map(v => {
			const i = Math.round((v / hi) * (BLOCKS.length - 1));
			return BLOCKS[Math.min(Math.max(i, 0), BLOCKS.length - 1)]!;
		})
		.join('');
}
