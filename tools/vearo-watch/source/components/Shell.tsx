import React from 'react';
import {Box, Text} from 'ink';
import Gradient from 'ink-gradient';
import Spinner from 'ink-spinner';
import {basename} from 'node:path';
import {theme} from '../theme.js';

/**
 * The wordmark.
 *
 * Three rows rather than the usual six-row block font: the dashboard is already
 * around 26 rows and a taller banner pushes the charts off an 80x24 terminal.
 */
const WORDMARK = ['╻ ╻┏━╸┏━┓┏━┓┏━┓', '┃┏┛┣╸ ┣━┫┣┳┛┃ ┃', '┗┛ ┗━╸╹ ╹╹┗╸┗━┛'];

export const TABS = ['Overview', 'Metrics', 'Memory', 'Config'] as const;
export type Tab = (typeof TABS)[number];

export function Header({
	tab,
	width,
	status,
}: {
	tab: Tab;
	width: number;
	status: 'running' | 'finished' | 'stalled';
}) {
	const pill =
		status === 'running' ? (
			<Text color={theme.accent}>
				<Spinner type="dots" /> running
			</Text>
		) : status === 'finished' ? (
			<Text color={theme.good}>finished</Text>
		) : (
			<Text color={theme.bad}>stalled</Text>
		);

	return (
		<Box width={width}>
			<Box flexDirection="column" width={17} flexShrink={0}>
				{WORDMARK.map((line, i) => (
					<Gradient key={i} name="teen">
						<Text>{line}</Text>
					</Gradient>
				))}
			</Box>
			<Box flexDirection="column" flexGrow={1} justifyContent="center">
				<Box>
					{TABS.map(t => (
						<Text
							key={t}
							bold={t === tab}
							color={t === tab ? theme.text : theme.dim}
						>
							{t === tab ? `[ ${t} ]` : `  ${t}  `}
							{'  '}
						</Text>
					))}
				</Box>
			</Box>
			<Box flexShrink={0} justifyContent="flex-end">
				{pill}
			</Box>
		</Box>
	);
}

/** Bottom status bar: identity, what is being watched, and the key bindings. */
export function StatusBar({
	width,
	file,
	epochs,
	note,
}: {
	width: number;
	file: string;
	epochs: number;
	note: string;
}) {
	return (
		<Box width={width}>
			<Text color={theme.accent}>vearo</Text>
			<Text color={theme.dim}>{' • watching '}</Text>
			<Text color={theme.text}>{basename(file)}</Text>
			<Text color={theme.dim}>{' • '}</Text>
			<Text color={theme.accent2}>{`${epochs} epochs`}</Text>
			<Text color={theme.dim}>{note ? ` • ${note}` : ''}</Text>
			<Box flexGrow={1} justifyContent="flex-end">
				<Text color={theme.dim}>
					<Text color={theme.text}>[TAB]</Text> View{' '}
					<Text color={theme.text}>[↑↓]</Text> Run{' '}
					<Text color={theme.text}>[W]</Text> Window{' '}
					<Text color={theme.text}>[Q]</Text> Quit
				</Text>
			</Box>
		</Box>
	);
}
