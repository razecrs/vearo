import React from 'react';
import {Box, Text} from 'ink';
import Spinner from 'ink-spinner';
import Gradient from 'ink-gradient';
import {sparkline} from '../braille.js';
import {theme, duration, memory} from '../theme.js';
import type {EpochRecord, NoteRecord, Snapshot} from '../stream.js';

export function Header({snap, width}: {snap: Snapshot; width: number}) {
	const title = snap.run?.title ?? 'waiting for run';
	const device = snap.run?.device ?? '';

	const status = snap.done ? (
		<Text color={theme.good}>finished</Text>
	) : snap.live ? (
		<Text color={theme.accent}>
			<Spinner type="dots" /> running
		</Text>
	) : (
		<Text color={theme.warn}>stalled</Text>
	);

	// Explicit widths rather than flex sizing. Leaving the right-hand box to
	// shrink lets Yoga collapse it to a single column, and Ink then wraps the
	// device name one character per line straight down through the layout.
	const label = snap.done ? 'finished' : snap.live ? '  running' : 'stalled';
	const rightW = device.length + 1 + label.length;
	const leftW = Math.max(width - rightW, 8);

	return (
		<Box width={width}>
			<Box width={leftW}>
				<Gradient name="teen">
					<Text bold>VEARO</Text>
				</Gradient>
				<Text color={theme.text} wrap="truncate">
					{' '}
					{title}
				</Text>
			</Box>
			<Box width={rightW} flexShrink={0} justifyContent="flex-end">
				<Text color={theme.accent2}>{device} </Text>
				{status}
			</Box>
		</Box>
	);
}

export function Progress({snap, width}: {snap: Snapshot; width: number}) {
	const total = snap.run?.total_epochs ?? 0;
	const last = snap.epochs.at(-1);
	const epoch = last?.epoch ?? 0;
	const frac = total > 0 ? Math.min(epoch / total, 1) : 0;

	// Per-epoch cost from the run's own history, not a fixed guess, so the ETA
	// tracks a run that speeds up or slows down.
	const perEpoch = last && epoch > 0 ? last.elapsed_s / epoch : 0;
	const eta = perEpoch * Math.max(total - epoch, 0);

	const barW = Math.max(width - 26, 10);
	const filled = Math.round(frac * barW);

	return (
		<Box width={width}>
			<Text color={theme.accent}>{'█'.repeat(filled)}</Text>
			<Text color={theme.dim}>{'░'.repeat(barW - filled)}</Text>
			<Text bold color={theme.text}>
				{` ${String(Math.round(frac * 100)).padStart(3)}%`}
			</Text>
			<Text color={theme.dim}>{'  ep '}</Text>
			<Text color={theme.text}>
				{epoch}/{total}
			</Text>
			<Text color={theme.dim}>{snap.done ? '  done ' : '  eta '}</Text>
			<Text color={theme.text}>
				{duration(snap.done ? (snap.done.elapsed_s ?? 0) : eta)}
			</Text>
		</Box>
	);
}

function Metric({
	label,
	value,
	color,
	delta,
	lowerIsBetter,
	width,
}: {
	label: string;
	value: string;
	color: string;
	delta?: number | undefined;
	lowerIsBetter?: boolean;
	width: number;
}) {
	const show = delta !== undefined && Math.abs(delta) > 1e-6;
	const improving = show && delta! < 0 === Boolean(lowerIsBetter);
	return (
		<Box flexDirection="column" width={width} flexShrink={0}>
			<Text color={theme.dim} wrap="truncate">{label}</Text>
			<Box>
				<Text bold color={color} wrap="truncate">
					{value}
				</Text>
				{show && (
					<Text color={improving ? theme.good : theme.bad}>
						{` ${delta! < 0 ? '▼' : '▲'}${Math.abs(delta!).toFixed(4)}`}
					</Text>
				)}
			</Box>
		</Box>
	);
}

export function Stats({snap, width}: {snap: Snapshot; width: number}) {
	const last = snap.epochs.at(-1);
	const prev = snap.epochs.at(-2);
	if (!last) return <Box width={width} />;

	const withAcc = snap.epochs.filter(
		(e): e is EpochRecord & {val_acc: number} => e.val_acc !== null,
	);
	const best = withAcc.reduce<(EpochRecord & {val_acc: number}) | undefined>(
		(b, e) => (!b || e.val_acc > b.val_acc ? e : b),
		undefined,
	);

	const col = Math.floor(width / 4);

	return (
		<Box width={width}>
			<Metric
				width={col}
				label="train loss"
				value={last.train_loss.toFixed(4)}
				color={theme.accent}
				delta={prev ? last.train_loss - prev.train_loss : undefined}
				lowerIsBetter
			/>
			<Metric
				width={col}
				label="val loss"
				value={last.val_loss?.toFixed(4) ?? '-'}
				color={theme.accent2}
				delta={
					prev && last.val_loss !== null && prev.val_loss !== null
						? last.val_loss - prev.val_loss
						: undefined
				}
				lowerIsBetter
			/>
			<Metric
				width={col}
				label="val acc"
				value={last.val_acc === null ? '-' : `${(last.val_acc * 100).toFixed(2)}%`}
				color={theme.good}
				delta={
					prev && last.val_acc !== null && prev.val_acc !== null
						? last.val_acc - prev.val_acc
						: undefined
				}
			/>
			<Metric
				width={col}
				label="best"
				value={best ? `${(best.val_acc * 100).toFixed(2)}% (ep ${best.epoch})` : '-'}
				color={theme.text}
			/>
		</Box>
	);
}

/**
 * Memory panel.
 *
 * A flat sparkline is the healthy result, which is why both series are scaled
 * from zero. Burn's dashboard shows no memory at all; for a framework whose
 * pitch is memory efficiency, an unbounded host allocation should be visible
 * within a few epochs rather than at the OOM kill.
 */
export function Memory({snap, width}: {snap: Snapshot; width: number}) {
	const vram = snap.epochs
		.map(e => e.vram_mb)
		.filter((v): v is number => v !== null);
	const ram = snap.epochs.map(e => e.ram_gb);
	const last = snap.epochs.at(-1);
	const sparkW = Math.max(Math.floor(width / 2) - 22, 8);

	const growth =
		ram.length >= 5 ? ram.at(-1)! - Math.min(...ram.slice(0, 3)) : 0;
	const leaking = growth > 0.5;

	return (
		<Box width={width} flexDirection="column">
			<Box>
				<Box width={Math.floor(width / 2)}>
					<Text color={theme.dim}>vram </Text>
					<Text color={theme.accent}>{sparkline(vram, sparkW)}</Text>
					<Text bold color={theme.text}>
						{' '}
						{last?.vram_mb === null || last?.vram_mb === undefined
							? '-'
							: `${last.vram_mb} MiB`}
					</Text>
				</Box>
				<Box>
					<Text color={theme.dim}>ram </Text>
					<Text color={leaking ? theme.bad : theme.accent2}>
						{sparkline(ram, sparkW)}
					</Text>
					<Text bold color={theme.text}>
						{' '}
						{last ? memory(last.ram_gb) : '-'}
					</Text>
					{leaking && (
						<Text color={theme.bad}>{` +${growth.toFixed(1)} GB`}</Text>
					)}
				</Box>
			</Box>
		</Box>
	);
}

export function Notes({
	notes,
	width,
	rows,
}: {
	notes: NoteRecord[];
	width: number;
	rows: number;
}) {
	const recent = notes.slice(-rows);
	return (
		<Box flexDirection="column" width={width} height={rows}>
			{recent.map((n, i) => (
				<Text key={`${n.ts}-${i}`} color={theme.good} wrap="truncate">
					{'* '}
					{n.text}
				</Text>
			))}
		</Box>
	);
}
