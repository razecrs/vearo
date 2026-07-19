/**
 * Discovers other runs sitting alongside the one being watched.
 *
 * Each run is a JSONL file, so a directory of them is a run history. Only the
 * head and tail of each file are needed for a summary, but the files are small
 * enough (one short line per epoch) that reading them whole is simpler and
 * still cheap.
 */
import {readdirSync, readFileSync, statSync} from 'node:fs';
import {dirname, join, basename} from 'node:path';

export type RunSummary = {
	path: string;
	name: string;
	title: string;
	device: string;
	epochs: number;
	totalEpochs: number;
	bestAcc: number | null;
	status: 'running' | 'finished' | 'stalled' | 'empty';
	mtime: number;
};

/** Reads one run file into a summary, or undefined if it is unreadable. */
export function summarise(path: string): RunSummary | undefined {
	let text: string;
	let mtime: number;
	try {
		text = readFileSync(path, 'utf8');
		mtime = statSync(path).mtimeMs;
	} catch {
		return undefined;
	}

	let title = basename(path, '.jsonl');
	let device = '';
	let totalEpochs = 0;
	let epochs = 0;
	let bestAcc: number | null = null;
	let done = false;

	for (const line of text.split('\n')) {
		if (!line.trim()) continue;
		let rec: any;
		try {
			rec = JSON.parse(line);
		} catch {
			continue;
		}
		if (rec.type === 'run') {
			title = rec.title ?? title;
			device = rec.device ?? '';
			totalEpochs = rec.total_epochs ?? 0;
		} else if (rec.type === 'epoch') {
			epochs = rec.epoch ?? epochs + 1;
			if (rec.val_acc !== null && rec.val_acc !== undefined) {
				bestAcc = bestAcc === null ? rec.val_acc : Math.max(bestAcc, rec.val_acc);
			}
		} else if (rec.type === 'done') {
			done = true;
		}
	}

	// A file untouched for two minutes with no terminating record is a run that
	// died rather than one that is merely slow.
	const stale = Date.now() - mtime > 120_000;
	const status: RunSummary['status'] =
		epochs === 0 ? 'empty' : done ? 'finished' : stale ? 'stalled' : 'running';

	return {path, name: basename(path), title, device, epochs, totalEpochs, bestAcc, status, mtime};
}

/** Lists every run in the same directory, newest first. */
export function listRuns(anyRunPath: string): RunSummary[] {
	const dir = dirname(anyRunPath) || '.';
	let names: string[];
	try {
		names = readdirSync(dir).filter(n => n.endsWith('.jsonl'));
	} catch {
		return [];
	}
	return names
		.map(n => summarise(join(dir, n)))
		.filter((r): r is RunSummary => r !== undefined)
		.sort((a, b) => b.mtime - a.mtime);
}
