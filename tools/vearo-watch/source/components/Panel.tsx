import React from 'react';
import {Box, Text} from 'ink';
import {theme} from '../theme.js';

/**
 * A bordered panel whose title sits inside the top border.
 *
 * Ink's own `borderStyle` cannot inline a title, and its flex sizing collapses
 * boxes in ways that wrap content one character per line. Both borders and
 * widths are therefore drawn by hand: every row is padded to exactly `inner`
 * columns before the border characters go on, so a panel can never be ragged.
 */
export function Panel({
	title,
	width,
	rows,
	minRows,
	color = theme.dim,
	titleColor = theme.accent,
}: {
	title: string;
	width: number;
	rows: React.ReactNode[];
	/** Pad out to this many content rows so panels in a row line up. */
	minRows?: number;
	color?: string;
	titleColor?: string;
}) {
	const inner = Math.max(width - 2, 4);
	const label = ` ${title} `;
	const dashes = Math.max(inner - label.length, 0);
	const left = Math.floor(dashes / 2);

	const body = [...rows];
	if (minRows !== undefined) {
		while (body.length < minRows) body.push(<Text> </Text>);
	}

	return (
		<Box flexDirection="column" width={width}>
			<Box>
				<Text color={color}>{'╭' + '─'.repeat(left)}</Text>
				<Text color={titleColor}>{label}</Text>
				<Text color={color}>
					{'─'.repeat(dashes - left) + '╮'}
				</Text>
			</Box>
			{body.map((row, i) => (
				<Box key={i}>
					<Text color={color}>{'│'}</Text>
					<Box width={inner}>{row}</Box>
					<Text color={color}>{'│'}</Text>
				</Box>
			))}
			<Box>
				<Text color={color}>
					{'╰' + '─'.repeat(inner) + '╯'}
				</Text>
			</Box>
		</Box>
	);
}

/** Pads a plain string to an exact column count, truncating if it overflows. */
export function fit(s: string, width: number): string {
	return s.length > width ? s.slice(0, Math.max(width - 1, 0)) + '…' : s.padEnd(width);
}
