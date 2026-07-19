import React, {useEffect, useState} from 'react';
import {Box, Text, useApp, useInput, useStdout} from 'ink';
import {Chart, type Series} from './components/Chart.js';
import {Header, Progress, Stats, Memory, Notes} from './components/Panels.js';
import {MetricsStream, type Snapshot} from './stream.js';
import {theme} from './theme.js';

/** How much of the run the charts show. */
type Window_ = 'full' | 'recent';

const RECENT = 30;

export function App({path}: {path: string}) {
	const {exit} = useApp();
	const {stdout} = useStdout();
	const [snap, setSnap] = useState<Snapshot>({
		epochs: [],
		notes: [],
		live: false,
	});
	const [window_, setWindow] = useState<Window_>('full');
	const [cols, setCols] = useState(stdout.columns || 100);

	useEffect(() => {
		const stream = new MetricsStream(path);
		stream.on('update', (s: Snapshot) => setSnap({...s}));
		stream.start();
		return () => stream.stop();
	}, [path]);

	useEffect(() => {
		const onResize = () => setCols(stdout.columns || 100);
		stdout.on('resize', onResize);
		return () => {
			stdout.off('resize', onResize);
		};
	}, [stdout]);

	// Same bindings as burn-train, so muscle memory carries over.
	useInput((input, key) => {
		if (input === 'q' || key.escape || (key.ctrl && input === 'c')) exit();
		if (key.leftArrow || key.rightArrow || input === 'f') {
			setWindow(w => (w === 'full' ? 'recent' : 'full'));
		}
	});

	const width = Math.max(Math.min(cols - 2, 120), 60);
	const panelW = Math.floor((width - 2) / 2);
	const chartH = 8;

	const shown =
		window_ === 'recent' ? snap.epochs.slice(-RECENT) : snap.epochs;

	const lossSeries: Series[] = [
		{
			values: shown.map(e => e.train_loss),
			color: theme.accent,
			label: 'train',
		},
		{
			values: shown
				.map(e => e.val_loss)
				.filter((v): v is number => v !== null),
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

	if (snap.error) {
		return (
			<Box flexDirection="column" padding={1}>
				<Text color={theme.warn}>{snap.error}</Text>
				<Text color={theme.dim}>
					start a run with .with_metrics(&quot;{path}&quot;), or press q to quit
				</Text>
			</Box>
		);
	}

	return (
		<Box flexDirection="column" width={width} paddingX={1}>
			<Header snap={snap} width={width - 2} />
			<Text color={theme.dim}>{'─'.repeat(width - 2)}</Text>

			<Box marginTop={1}>
				<Progress snap={snap} width={width - 2} />
			</Box>

			<Box marginTop={1}>
				<Chart
					title="loss"
					series={lossSeries}
					width={panelW}
					height={chartH}
				/>
				<Chart
					title="val accuracy"
					series={accSeries}
					width={panelW}
					height={chartH}
					format={v => `${v.toFixed(1)}%`}
				/>
			</Box>

			<Box marginTop={1}>
				<Stats snap={snap} width={width - 2} />
			</Box>

			<Box marginTop={1}>
				<Memory snap={snap} width={width - 2} />
			</Box>

			<Box marginTop={1}>
				<Notes notes={snap.notes} width={width - 2} rows={3} />
			</Box>

			<Box marginTop={1}>
				<Text color={theme.dim}>
					q quit   {'<'}
					{'>'} window: <Text color={theme.text}>{window_}</Text>
					{snap.epochs.length > 0 && `   ${snap.epochs.length} epochs`}
				</Text>
			</Box>
		</Box>
	);
}
