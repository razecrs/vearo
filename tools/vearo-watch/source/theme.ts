/** Shared palette. Matches the colours the Rust-side terminal output uses. */
export const theme = {
	accent: '#5eead4', // teal, train loss
	accent2: '#a78bfa', // violet, val loss
	good: '#4ade80', // green, accuracy and improvement
	bad: '#f87171', // red, regression
	warn: '#fbbf24', // amber
	dim: '#64748b', // slate, chrome and labels
	text: '#e2e8f0',
} as const;

/** Formats seconds as `H:MM:SS` or `M:SS`. */
export function duration(secs: number): string {
	const s = Math.max(0, Math.floor(secs));
	const mm = String(Math.floor((s % 3600) / 60)).padStart(2, '0');
	const ss = String(s % 60).padStart(2, '0');
	return s >= 3600 ? `${Math.floor(s / 3600)}:${mm}:${ss}` : `${Math.floor(s / 60)}:${ss}`;
}

/** Formats a byte count given in GB, switching to MB below 1 GB. */
export function memory(gb: number): string {
	return gb < 1 ? `${Math.round(gb * 1024)} MB` : `${gb.toFixed(1)} GB`;
}
