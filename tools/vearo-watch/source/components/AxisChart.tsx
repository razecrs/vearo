import React from 'react';
import {Box, Text} from 'ink';
import {Braille} from '../braille.js';
import {theme} from '../theme.js';

export type Series = {values: number[]; color: string; label: string};

/**
 * A braille line chart with labelled axes.
 *
 * Layout per row: `<y label> ┤ <plot>`, closed by an x axis and tick labels.
 * The y-label gutter is a fixed width so the plot area is identical on every
 * row and the axis stays straight.
 */
export function axisChartRows({
	series,
	width,
	height,
	yFormat = v => v.toFixed(2),
	xLabel,
	xMax,
}: {
	series: Series[];
	width: number;
	height: number;
	yFormat?: (v: number) => string;
	xLabel?: string;
	xMax?: number;
}): React.ReactNode[] {
	const withData = series.filter(s => s.values.length > 0);

	const all = withData.flatMap(s => s.values);
	const lo = all.length ? Math.min(...all) : 0;
	const hi = all.length ? Math.max(...all) : 1;

	const gutter = Math.max(yFormat(hi).length, yFormat(lo).length);
	const plotW = Math.max(width - gutter - 2, 8);

	const layers = withData.map(s => {
		const c = new Braille(plotW, height);
		c.plot(s.values, lo, hi);
		return c.rows();
	});

	const rows: React.ReactNode[] = [];
	for (let r = 0; r < height; r++) {
		// Only the top and bottom rows carry a value label.
		const label =
			r === 0 ? yFormat(hi) : r === height - 1 ? yFormat(lo) : '';

		const cells: React.ReactNode[] = [];
		let run = '';
		let runColor: string = theme.dim;
		const flush = (k: string) => {
			if (run) {
				cells.push(
					<Text key={k} color={runColor}>
						{run}
					</Text>,
				);
				run = '';
			}
		};

		for (let c = 0; c < plotW; c++) {
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

		rows.push(
			<Box key={r}>
				<Text color={theme.dim}>{label.padStart(gutter)}</Text>
				<Text color={theme.dim}>{r === height - 1 ? '└' : '┤'}</Text>
				{cells}
			</Box>,
		);
	}

	// x axis and its ticks
	rows.push(
		<Box key="axis">
			<Text color={theme.dim}>{' '.repeat(gutter)}</Text>
			<Text color={theme.dim}>{'└' + '─'.repeat(plotW + 1)}</Text>
		</Box>,
	);
	if (xMax !== undefined) {
		const leftTick = '0';
		const rightTick = String(xMax);
		const mid = Math.floor(xMax / 2);
		const midTick = String(mid);
		const padA = Math.max(
			Math.floor(plotW / 2) - leftTick.length - Math.floor(midTick.length / 2),
			1,
		);
		const padB = Math.max(
			plotW - leftTick.length - padA - midTick.length - rightTick.length,
			1,
		);
		rows.push(
			<Box key="ticks">
				<Text color={theme.dim}>{' '.repeat(gutter + 1)}</Text>
				<Text color={theme.dim}>
					{leftTick}
					{' '.repeat(padA)}
					{midTick}
					{' '.repeat(padB)}
					{rightTick}
				</Text>
			</Box>,
		);
	}

	// legend, coloured to match each curve
	rows.push(
		<Box key="legend" justifyContent="center">
			{withData.map((s, i) => (
				<Text key={s.label} color={s.color}>
					{i > 0 ? '   ' : ''}
					{'──'} <Text color={theme.dim}>{s.label}</Text>
				</Text>
			))}
			{xLabel && <Text color={theme.dim}>{`   ${xLabel}`}</Text>}
		</Box>,
	);

	return rows;
}
