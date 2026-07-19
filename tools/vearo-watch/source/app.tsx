import React, {useEffect, useMemo, useState} from 'react';
import {Box, Text, useApp, useInput, useStdout} from 'ink';
import {axisChartRows, type Series} from './components/AxisChart.js';
import {Panel, fit} from './components/Panel.js';
import {Header, StatusBar, TABS, type Tab} from './components/Shell.js';
import {
	Histogram,
	runListRows,
	traceRows,
	Field,
	type TraceEvent,
} from './components/Widgets.js';
import {MetricsStream, type Snapshot} from './stream.js';
import {listRuns, type RunSummary} from './runs.js';
import {theme, duration, memory} from './theme.js';

/** How many epochs the recent window keeps. */
const RECENT = 30;

export function App({path}: {path: string}) {
	const {exit} = useApp();
	const {stdout} = useStdout();

	const [snap, setSnap] = useState<Snapshot>({epochs: [], notes: [], live: false});
	const [runs, setRuns] = useState<RunSummary[]>([]);
	const [selected, setSelected] = useState(0);
	const [active, setActive] = useState(path);
	const [tab, setTab] = useState<Tab>('Overview');
	const [recent, setRecent] = useState(false);
	const [cols, setCols] = useState(stdout.columns || 120);

	useEffect(() => {
		const stream = new MetricsStream(active);
		stream.on('update', (s: Snapshot) => setSnap({...s}));
		stream.start();
		return () => stream.stop();
	}, [active]);

	useEffect(() => {
		const scan = () => setRuns(listRuns(path));
		scan();
		const t = setInterval(scan, 5000);
		return () => clearInterval(t);
	}, [path]);

	useEffect(() => {
		const onResize = () => setCols(stdout.columns || 120);
		stdout.on('resize', onResize);
		return () => {
			stdout.off('resize', onResize);
		};
	}, [stdout]);

	useInput((input, key) => {
		if (input === 'q' || key.escape || (key.ctrl && input === 'c')) exit();
		if (key.tab || key.rightArrow) {
			setTab(t => TABS[(TABS.indexOf(t) + 1) % TABS.length]!);
		}
		if (key.leftArrow) {
			setTab(t => TABS[(TABS.indexOf(t) + TABS.length - 1) % TABS.length]!);
		}
		if (input === 'w') setRecent(r => !r);
		if (key.upArrow) setSelected(s => Math.max(s - 1, 0));
		if (key.downArrow) setSelected(s => Math.min(s + 1, Math.max(runs.length - 1, 0)));
		if (key.return && runs[selected]) setActive(runs[selected]!.path);
	});

	const width = Math.max(Math.min(cols - 2, 132), 76);
	// Children sit inside the outer box's paddingX, so they get width - 2.
	const contentW = width - 2;
	const shown = recent ? snap.epochs.slice(-RECENT) : snap.epochs;
	const last = snap.epochs.at(-1);
	const total = snap.run?.total_epochs ?? 0;

	const best = useMemo(() => {
		let b: {acc: number; epoch: number} | undefined;
		for (const e of snap.epochs) {
			if (e.val_acc !== null && (!b || e.val_acc > b.acc)) {
				b = {acc: e.val_acc, epoch: e.epoch};
			}
		}
		return b;
	}, [snap.epochs]);

	const status = snap.done ? 'finished' : snap.live ? 'running' : 'stalled';

	// One ordered timeline out of the run's records.
	const events = useMemo<TraceEvent[]>(() => {
		const out: TraceEvent[] = [];
		if (snap.run) {
			out.push({
				tag: 'START',
				text: `${snap.run.title} on ${snap.run.device}, ${snap.run.total_epochs} epochs`,
			});
		}
		for (const n of snap.notes) out.push({tag: 'BEST', text: n.text});
		if (last) {
			out.push({
				tag: 'EPOCH',
				text: `${last.epoch}: train ${last.train_loss.toFixed(4)}${
					last.val_loss === null ? '' : `  val ${last.val_loss.toFixed(4)}`
				}${last.val_acc === null ? '' : `  acc ${(last.val_acc * 100).toFixed(2)}%`}`,
			});
		}
		if (snap.done) {
			out.push({
				tag: 'DONE',
				text: `best ${(snap.done.best_acc * 100).toFixed(2)}% at epoch ${
					snap.done.best_epoch
				} in ${duration(snap.done.elapsed_s)}`,
			});
		} else if (status === 'stalled') {
			out.push({tag: 'STALL', text: 'no new epoch for over two minutes'});
		}
		return out;
	}, [snap, last, status]);

	if (snap.error && snap.epochs.length === 0) {
		return (
			<Box flexDirection="column" padding={1}>
				<Text color={theme.warn}>{snap.error}</Text>
				<Text color={theme.dim}>
					point at a run file, or start one with .with_metrics(...) - q to quit
				</Text>
			</Box>
		);
	}

	const lossSeries: Series[] = [
		{values: shown.map(e => e.train_loss), color: theme.accent, label: 'train'},
		{
			values: shown.map(e => e.val_loss).filter((v): v is number => v !== null),
			color: theme.accent2,
			label: 'val',
		},
	];
	const accSeries: Series[] = [
		{
			values: shown
				.map(e => e.val_acc)
				.filter((v): v is number => v !== null)
				.map(v => v * 100),
			color: theme.good,
			label: 'val acc',
		},
	];
	const vram = snap.epochs.map(e => e.vram_mb).filter((v): v is number => v !== null);
	const ram = snap.epochs.map(e => e.ram_gb);

	// Three columns, sized from the reference layout.
	const leftW = Math.max(Math.floor(contentW * 0.30), 26);
	const centerW = Math.max(Math.floor(contentW * 0.34), 30);
	const rightW = contentW - leftW - centerW - 2;

	const progress = () => {
		const frac = total > 0 && last ? Math.min(last.epoch / total, 1) : 0;
		const barW = Math.max(centerW - 20, 8);
		const filled = Math.round(frac * barW);
		const perEpoch = last && last.epoch > 0 ? last.elapsed_s / last.epoch : 0;
		const eta = perEpoch * Math.max(total - (last?.epoch ?? 0), 0);
		return (
			<Box>
				<Text color={theme.accent}>{' ' + '█'.repeat(filled)}</Text>
				<Text color={theme.dim}>{'░'.repeat(barW - filled)}</Text>
				<Text bold color={theme.text}>{` ${String(Math.round(frac * 100)).padStart(3)}%`}</Text>
				<Text color={theme.dim}>{snap.done ? ' done' : ` ${duration(eta)}`}</Text>
			</Box>
		);
	};

	const centerPanel = () => {
		if (tab === 'Memory') {
			return (
				<Panel
					title="MEMORY OVER TIME"
					width={centerW}
					rows={[
						...axisChartRows({
							series: [
								{values: vram, color: theme.accent, label: 'vram MiB'},
								{values: ram.map(g => g * 1024), color: theme.accent2, label: 'ram MiB'},
							],
							width: centerW - 2,
							height: 8,
							yFormat: v => `${Math.round(v)}`,
							xMax: total,
						}),
					]}
				/>
			);
		}
		if (tab === 'Config') {
			return (
				<Panel
					title="RUN CONFIG"
					width={centerW}
					minRows={12}
					rows={[
						<Field label="title" value={snap.run?.title ?? '-'} width={centerW - 2} />,
						<Field label="device" value={snap.run?.device ?? '-'} width={centerW - 2} color={theme.accent2} />,
						<Field label="epochs" value={`${last?.epoch ?? 0} / ${total}`} width={centerW - 2} />,
						<Field label="elapsed" value={last ? duration(last.elapsed_s) : '-'} width={centerW - 2} />,
						<Field
							label="per epoch"
							value={last && last.epoch > 0 ? `${(last.elapsed_s / last.epoch).toFixed(1)}s` : '-'}
							width={centerW - 2}
						/>,
						<Field label="file" value={active} width={centerW - 2} />,
						<Field label="window" value={recent ? `last ${RECENT}` : 'full history'} width={centerW - 2} />,
					]}
				/>
			);
		}
		const isMetrics = tab === 'Metrics';
		return (
			<Panel
				title={isMetrics ? 'VAL ACCURACY' : `PERFORMANCE (${shown.length} epochs)`}
				width={centerW}
				rows={[
					...axisChartRows({
						series: isMetrics ? accSeries : lossSeries,
						width: centerW - 2,
						height: isMetrics ? 10 : 8,
						yFormat: isMetrics ? v => `${v.toFixed(0)}%` : v => v.toFixed(2),
						xMax: total,
					}),
					...(isMetrics ? [] : [<Text> </Text>, progress()]),
				]}
			/>
		);
	};

	return (
		<Box flexDirection="column" width={width} paddingX={1}>
			<Header tab={tab} width={contentW} status={status} />
			<Text color={theme.dim}>{'─'.repeat(contentW)}</Text>

			<Box marginTop={1}>
				<Box flexDirection="column" width={leftW}>
					<Panel
						title="RUNS"
						width={leftW}
						minRows={8}
						rows={[
							...runListRows({runs, selected, width: leftW - 2, rows: 8}),
						]}
					/>
					<Panel
						title="CURRENT"
						width={leftW}
						minRows={4}
						rows={[
							<Field label="train" value={last ? last.train_loss.toFixed(4) : '-'} width={leftW - 2} color={theme.accent} labelWidth={8} />,
							<Field label="val" value={last?.val_loss?.toFixed(4) ?? '-'} width={leftW - 2} color={theme.accent2} labelWidth={8} />,
							<Field label="acc" value={last?.val_acc === null || last === undefined ? '-' : `${(last.val_acc! * 100).toFixed(2)}%`} width={leftW - 2} color={theme.good} labelWidth={8} />,
							<Field label="best" value={best ? `${(best.acc * 100).toFixed(2)}% (ep ${best.epoch})` : '-'} width={leftW - 2} labelWidth={8} />,
						]}
					/>
				</Box>

				<Box flexDirection="column" width={centerW} marginX={1}>
					{centerPanel()}
					<Panel
						title="RESOURCE USAGE"
						width={centerW}
						rows={[
							<Box>
								<Text color={theme.dim}>{' vram '}</Text>
								<Histogram values={vram} width={centerW - 9} color={theme.accent} highlightFrom={snap.epochs.length - 6} />
							</Box>,
							<Box>
								<Text color={theme.dim}>{' ram  '}</Text>
								<Histogram values={ram} width={centerW - 9} color={theme.accent2} highlightFrom={snap.epochs.length - 6} />
							</Box>,
							<Text> </Text>,
							<Text color={theme.dim}>
								{fit(
									` peak ${vram.length ? `${Math.max(...vram)} MiB vram` : 'vram -'}   ${
										ram.length ? memory(Math.max(...ram)) : '-'
									} ram`,
									centerW - 2,
								)}
							</Text>,
						]}
					/>
				</Box>

				<Box flexDirection="column" width={rightW}>
					<Panel
						title="EVENT TRACE"
						width={rightW}
						minRows={14}
						rows={traceRows({events, width: rightW - 2, rows: 14})}
					/>
				</Box>
			</Box>

			<Box marginTop={1}>
				<StatusBar
					width={contentW}
					file={active}
					epochs={snap.epochs.length}
					note={recent ? `window: last ${RECENT}` : ''}
				/>
			</Box>
		</Box>
	);
}
