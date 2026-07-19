import React from 'react';
import {Box, Text} from 'ink';
import {Braille} from '../braille.js';
import {theme} from '../theme.js';

export type Series = {
	values: number[];
	color: string;
	label: string;
};

/** Visible width of the legend row: `── label` per series, joined by two spaces. */
function legendWidth(series: Series[]): number {
	return series.reduce((n, s, i) => n + (i > 0 ? 2 : 0) + 3 + s.label.length, 0);
}

type Props = {
	title: string;
	series: Series[];
	width: number;
	height: number;
	/** Formats the axis bounds, e.g. as a percentage. */
	format?: (v: number) => string;
};

/**
 * A bordered braille line chart with any number of overlaid series.
 *
 * Each series is rasterised on its own canvas and the layers are merged per
 * cell, because a single canvas could not carry per-series colour. Where curves
 * cross, the earlier series wins the cell.
 */
export function Chart({title, series, width, height, format}: Props) {
	const inner = Math.max(width - 4, 8);
	const withData = series.filter(s => s.values.length > 0);

	if (withData.length === 0) {
		return (
			<Box
				flexDirection="column"
				borderStyle="round"
				borderColor={theme.dim}
				width={width}
				height={height + 2}
				paddingX={1}
			>
				<Text color={theme.dim}>{title}</Text>
				<Box flexGrow={1} alignItems="center" justifyContent="center">
					<Text color={theme.dim}>waiting for data</Text>
				</Box>
			</Box>
		);
	}

	// Shared scale across series so overlaid curves are directly comparable.
	const all = withData.flatMap(s => s.values);
	const lo = Math.min(...all);
	const hi = Math.max(...all);

	const layers = withData.map(s => {
		const c = new Braille(inner, height);
		c.plot(s.values, lo, hi);
		return c.rows();
	});

	const fmt = format ?? ((v: number) => v.toFixed(3));

	const rows: React.ReactNode[] = [];
	for (let r = 0; r < height; r++) {
		const cells: React.ReactNode[] = [];
		let run = '';
		let runColor = '';

		const flush = (key: string) => {
			if (run) {
				cells.push(
					<Text key={key} color={runColor}>
						{run}
					</Text>,
				);
				run = '';
			}
		};

		for (let c = 0; c < inner; c++) {
			// topmost non-blank layer owns the cell
			let ch = ' ';
			let color: string = theme.dim;
			for (const [li, layer] of layers.entries()) {
				const candidate = layer[r]?.[c];
				if (candidate && candidate !== '⠀') {
					ch = candidate;
					color = withData[li]!.color;
					break;
				}
			}
			if (color !== runColor) {
				flush(`${r}-${c}`);
				runColor = color;
			}
			run += ch;
		}
		flush(`${r}-end`);
		rows.push(<Box key={r}>{cells}</Box>);
	}

	return (
		<Box
			flexDirection="column"
			borderStyle="round"
			borderColor={theme.dim}
			width={width}
			paddingX={1}
		>
			{/* Padded to `inner` by hand. space-between inside a bordered Box
			    sizes off content and leaves the right label short of the edge. */}
			<Box>
				<Text color={theme.dim}>
					{title.padEnd(Math.max(inner - fmt(hi).length, 0))}
					{fmt(hi)}
				</Text>
			</Box>
			{rows}
			<Box>
				{withData.map((s, i) => (
					<Text key={s.label} color={s.color}>
						{i > 0 ? '  ' : ''}
						{'── '}
						<Text color={theme.dim}>{s.label}</Text>
					</Text>
				))}
				<Text color={theme.dim}>
					{''.padEnd(Math.max(inner - legendWidth(withData) - fmt(lo).length, 0))}
					{fmt(lo)}
				</Text>
			</Box>
		</Box>
	);
}
