/**
 * Reads a Vearo metrics stream and follows it as it grows.
 *
 * The producer appends one JSON object per line and flushes after each. This
 * reader keeps a byte offset and only reads what was appended, so following a
 * long run costs nothing per update.
 */
import {createReadStream, statSync, watch, type FSWatcher} from 'node:fs';
import {EventEmitter} from 'node:events';

export type RunRecord = {
	type: 'run';
	title: string;
	device: string;
	total_epochs: number;
	started: number;
};

export type EpochRecord = {
	type: 'epoch';
	epoch: number;
	train_loss: number;
	val_loss: number | null;
	val_acc: number | null;
	vram_mb: number | null;
	ram_gb: number;
	elapsed_s: number;
	ts: number;
};

export type NoteRecord = {type: 'note'; text: string; ts: number};

export type DoneRecord = {
	type: 'done';
	best_acc: number;
	best_epoch: number;
	elapsed_s: number;
	ts: number;
};

export type Record_ = RunRecord | EpochRecord | NoteRecord | DoneRecord;

export type Snapshot = {
	run?: RunRecord;
	epochs: EpochRecord[];
	notes: NoteRecord[];
	done?: DoneRecord;
	/** True when the file has grown recently, so the run looks alive. */
	live: boolean;
	error?: string;
};

/**
 * Tails a JSONL metrics file, emitting a full snapshot on every change.
 *
 * Emits `update` with a {@link Snapshot}. Poll-based rather than purely
 * `fs.watch`, because `fs.watch` does not fire reliably for a file being
 * appended to over NFS or a synced directory, which is exactly how a remote
 * training run gets watched.
 */
export class MetricsStream extends EventEmitter {
	readonly #path: string;
	#offset = 0;
	#partial = '';
	#snapshot: Snapshot = {epochs: [], notes: [], live: false};
	#watcher?: FSWatcher;
	#timer?: NodeJS.Timeout;
	#lastGrowth = Date.now();

	constructor(path: string) {
		super();
		this.#path = path;
	}

	get snapshot(): Snapshot {
		return this.#snapshot;
	}

	start(): void {
		this.#poll();
		// Two triggers: fs.watch for immediate response on a local file, and a
		// slow poll as the fallback that actually works everywhere.
		try {
			this.#watcher = watch(this.#path, () => this.#poll());
		} catch {
			// File may not exist yet; the poll below will pick it up.
		}
		this.#timer = setInterval(() => this.#poll(), 500);
	}

	stop(): void {
		this.#watcher?.close();
		if (this.#timer) clearInterval(this.#timer);
	}

	#poll(): void {
		let size: number;
		try {
			size = statSync(this.#path).size;
		} catch {
			this.#emit({error: `waiting for ${this.#path}`});
			return;
		}

		// A shrunken file means the run was restarted and the file truncated.
		if (size < this.#offset) {
			this.#offset = 0;
			this.#partial = '';
			this.#snapshot = {epochs: [], notes: [], live: true};
		}

		if (size === this.#offset) {
			// No growth. A run with no new epoch for a while is probably dead,
			// but only say so once it has actually started and not finished.
			const stale = Date.now() - this.#lastGrowth > 120_000;
			if (this.#snapshot.live && stale && !this.#snapshot.done) {
				this.#emit({live: false});
			}
			return;
		}

		const stream = createReadStream(this.#path, {
			start: this.#offset,
			end: size - 1,
			encoding: 'utf8',
		});
		let chunk = '';
		stream.on('data', d => {
			chunk += d;
		});
		stream.on('end', () => {
			this.#offset = size;
			this.#lastGrowth = Date.now();
			this.#ingest(chunk);
		});
		stream.on('error', () => {
			/* transient; the next poll retries */
		});
	}

	#ingest(chunk: string): void {
		const text = this.#partial + chunk;
		const lines = text.split('\n');
		// The final element is either empty or a half-written line still being
		// flushed; hold it back until the rest arrives.
		this.#partial = lines.pop() ?? '';

		const epochs = [...this.#snapshot.epochs];
		const notes = [...this.#snapshot.notes];
		let {run, done} = this.#snapshot;

		for (const line of lines) {
			if (!line.trim()) continue;
			let rec: Record_;
			try {
				rec = JSON.parse(line) as Record_;
			} catch {
				continue; // skip a corrupt line rather than kill the dashboard
			}
			switch (rec.type) {
				case 'run':
					run = rec;
					break;
				case 'epoch':
					epochs.push(rec);
					break;
				case 'note':
					notes.push(rec);
					break;
				case 'done':
					done = rec;
					break;
			}
		}

		this.#snapshot = {run, epochs, notes, done, live: !done};
		this.emit('update', this.#snapshot);
	}

	#emit(patch: Partial<Snapshot>): void {
		this.#snapshot = {...this.#snapshot, ...patch};
		this.emit('update', this.#snapshot);
	}
}
