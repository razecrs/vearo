# vearo-watch

A live terminal dashboard for Vearo training runs. Built with
[Ink](https://github.com/vadimdemedes/ink), which renders React to the terminal.

```
 ╻ ╻┏━╸┏━┓┏━┓┏━┓                                                    finished
 ┃┏┛┣╸ ┣━┫┣┳┛┃ ┃  [ Overview ]    Metrics      Memory      Config
 ┗┛ ┗━╸╹ ╹╹┗╸┗━┛
 ───────────────────────────────────────────────────────────────────────────

 ╭────── RUNS ──────╮ ╭─── PERFORMANCE (59) ───╮ ╭───── EVENT TRACE ──────╮
 │├─▸style [SUCCESS]│ │3.33┤⠒⢄⡀                │ │[BEST] new best 41.64%  │
 │└─ mlp   [RUNNING]│ │    ┤  ⠑⠢⣀⡀             │ │[BEST] new best 42.03%  │
 │                  │ │    ┤     ⠑⠤⣀⣀          │ │[EPOCH] 59: train 1.35  │
 ╰──────────────────╯ │1.35└        ⠈⠑⠒⠤⠤⢄⣀    │ │[DONE] best 42.03% ep59 │
 ╭───── CURRENT ────╮ │    └───────────────────│ ╰────────────────────────╯
 │ train  1.3527    │ │     0     37       75  │
 │ acc    42.03%    │ │── train   ── val       │
 ╰──────────────────╯ ╰────────────────────────╯
```

## Why it reads a file

The dashboard does not run the training. Vearo appends one JSON object per epoch
to a file, and this tails it. That means:

- it works for a **headless run on a remote box** under `nohup`, which is where
  most real training happens
- you can **close the dashboard and reopen it** mid-run, or replay a run that
  already finished
- a dashboard crash cannot take a training run down with it

## Setup

```sh
cd tools/vearo-watch
npm install
npm run build
```

## Use

Have the run emit metrics:

```rust
let mut ui = TrainingMonitor::new("style cnn", "Cuda(0)", 75)
    .with_metrics("runs/style_cnn.jsonl");
```

Then watch it:

```sh
node dist/cli.js runs/style_cnn.jsonl
```

For a run on another machine, keep the file synced and point at the local copy:

```sh
watch -n5 'rsync -az --partial box:~/runs/style_cnn.jsonl runs/'
node dist/cli.js runs/style_cnn.jsonl
```

## Controls

| Key            | Action                                             |
| -------------- | -------------------------------------------------- |
| `tab`, `right` | next view: Overview, Metrics, Memory, Config       |
| `left`         | previous view                                      |
| `up`/`down`    | select a run                                       |
| `enter`        | open the selected run                              |
| `w`            | toggle full history and the last 30 epochs         |
| `q`, `esc`     | quit the dashboard (the training run is untouched) |

## The memory panel

Both sparklines are scaled from zero rather than from the series range, so a
steady allocation renders as a **flat line** and real growth is obvious. Scaling
to the range would amplify byte-level jitter into a sawtooth and make a healthy
run look like it is leaking.

Host RAM turns red and shows the delta once it grows more than 0.5 GB above the
run's opening epochs. The leak this was built after only became visible when the
process was OOM-killed at epoch 32.

## Views

- **Overview** - loss curves with axes, progress, resource histogram
- **Metrics** - validation accuracy at full height
- **Memory** - VRAM and host RAM over time, the leak check
- **Config** - what this run actually was

The RUNS panel lists every `.jsonl` beside the one you opened, so past runs stay
browsable. Select one and press enter to load it.

## Layout

- `source/stream.ts` - tails the JSONL and emits snapshots
- `source/runs.ts` - summarises other runs in the directory
- `source/components/Panel.tsx` - bordered panels with inline titles
- `source/components/AxisChart.tsx` - braille charts with labelled axes
- `source/braille.ts` - braille canvas and sparklines
- `source/components/Widgets.tsx` - run list, histogram, event trace
- `source/components/Shell.tsx` - wordmark, tab bar, status bar
- `source/app.tsx` - layout and key handling

Charts use the Unicode braille block: each character cell holds a 2x4 grid of
dots, so a chart is eight times the resolution of a block-character plot. That is
what makes a loss curve read as a curve instead of a staircase.
