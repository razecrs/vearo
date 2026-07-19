import React from 'react';
import {Box, Text} from 'ink';
import {theme, memory} from '../theme.js';
import {fit} from './Panel.js';
import type {RunSummary} from '../runs.js';

/** Vertical bar levels, one eighth apart. */
const BARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/**
 * A bar histogram, scaled from zero.
 *
 * Scaling from zero rather than from the series range is the whole point for
 * memory: steady allocation renders as a flat band. Range-scaling would turn
 * megabyte-level jitter into a dramatic sawtooth and make a healthy run look
 * like it is leaking.
 */
export function Histogram({
	values,
	width,
	color,
	highlightFrom,
	highlightColor = theme.accent2,
}: {
	values: number[];
	width: number;
	color: string;
	/** Index from which bars take the highlight colour (the recent window). */
	highlightFrom?: number;
	highlightColor?: string;
}) {
	if (values.length === 0) {
		return <Text color={theme.dim}>{' '.repeat(width)}</Text>;
	}
	const tail = values.slice(-width);
	const offset = values.length - tail.length;
	const hi = Math.max(...tail) * 1.15;

	return (
		<Box>
			{tail.map((v, i) => {
				const level = hi <= 0 ? 0 : Math.round((v / hi) * (BARS.length - 1));
				const hot = highlightFrom !== undefined && offset + i >= highlightFrom;
				return (
					<Text key={i} color={hot ? highlightColor : color}>
						{BARS[Math.min(Math.max(level, 0), BARS.length - 1)]}
					</Text>
				);
			})}
			<Text>{' '.repeat(Math.max(width - tail.length, 0))}</Text>
		</Box>
	);
}

const STATUS_STYLE: Record<RunSummary['status'], {tag: string; color: string}> = {
	running: {tag: 'RUNNING', color: theme.accent},
	finished: {tag: 'SUCCESS', color: theme.good},
	stalled: {tag: 'STALLED', color: theme.bad},
	empty: {tag: 'EMPTY  ', color: theme.dim},
};

/**
 * The run browser: every run in the directory, drawn as a tree.
 *
 * A framework's run history is the closest real analogue to a trajectory list,
 * and unlike an in-process dashboard it can show runs that already finished.
 */
export function runListRows({
	runs,
	selected,
	width,
	rows,
}: {
	runs: RunSummary[];
	selected: number;
	width: number;
	rows: number;
}): React.ReactNode[] {
	if (runs.length === 0) {
		return [<Text color={theme.dim}>{fit(' no runs found in this directory', width)}</Text>];
	}

	// Keep the selection in view when there are more runs than rows.
	const start = Math.max(0, Math.min(selected - Math.floor(rows / 2), runs.length - rows));
	const view = runs.slice(Math.max(start, 0), Math.max(start, 0) + rows);

	return view.map((r, i) => {
				const idx = Math.max(start, 0) + i;
				const isLast = idx === runs.length - 1;
				const style = STATUS_STYLE[r.status];
				const on = idx === selected;
				const acc = r.bestAcc === null ? '  -  ' : `${(r.bestAcc * 100).toFixed(1)}%`;
				const body = ` ${r.epochs}/${r.totalEpochs} ${acc}`;
				const head = `${isLast ? '└' : '├'}─${on ? '▸' : ' '}`;
				// tag renders as " [TAG]", so it costs its length plus three.
				const name = fit(
					r.title,
					Math.max(width - head.length - style.tag.length - body.length - 3, 4),
				);
				return (
					<Box key={r.path}>
						<Text color={theme.dim}>{head}</Text>
						<Text color={on ? theme.text : theme.dim} bold={on}>
							{name}
						</Text>
						<Text color={style.color}>{` [${style.tag}]`}</Text>
						<Text color={theme.dim}>{body}</Text>
					</Box>
				);
			});
}

/** Colour for each event tag in the trace. */
const TAG_COLOR: Record<string, string> = {
	START: theme.accent2,
	BEST: theme.good,
	EPOCH: theme.dim,
	DONE: theme.good,
	STALL: theme.bad,
};

export type TraceEvent = {tag: keyof typeof TAG_COLOR | string; text: string};

/**
 * The event trace: what actually happened, newest last.
 *
 * This replaces the reference mockup's prompt-diff panel, which has no analogue
 * in a training framework. Checkpoints, epochs and terminal state are the
 * events a training run genuinely produces.
 */
export function traceRows({
	events,
	width,
	rows,
}: {
	events: TraceEvent[];
	width: number;
	rows: number;
}): React.ReactNode[] {
	const view = events.slice(-rows);
	return view.map((e, i) => {
				const tag = `[${e.tag}]`;
				return (
					<Box key={i}>
						<Text color={TAG_COLOR[e.tag] ?? theme.dim}>{tag}</Text>
						<Text color={theme.text}>
							{fit(` ${e.text}`, Math.max(width - tag.length, 0))}
						</Text>
					</Box>
				);
			});
}

/** A `label  value` line, padded so a column of them aligns. */
export function Field({
	label,
	value,
	width,
	color = theme.text,
	labelWidth = 12,
}: {
	label: string;
	value: string;
	width: number;
	color?: string;
	labelWidth?: number;
}) {
	return (
		<Box>
			<Text color={theme.dim}>{fit(` ${label}`, labelWidth)}</Text>
			<Text color={color}>{fit(value, Math.max(width - labelWidth, 0))}</Text>
		</Box>
	);
}

/** Formats memory for the resource panel. */
export function memLabel(vramMb: number | null | undefined, ramGb: number | undefined): string {
	const v = vramMb === null || vramMb === undefined ? '-' : `${vramMb} MiB`;
	const r = ramGb === undefined ? '-' : memory(ramGb);
	return `vram ${v}   ram ${r}`;
}
