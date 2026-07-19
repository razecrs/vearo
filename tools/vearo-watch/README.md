# vearo-watch

A live terminal dashboard for Vearo training runs. Built with
[Ink](https://github.com/vadimdemedes/ink), which renders React to the terminal.

```
 VEARO style cnn                                          Cuda(0) ⠙ running
 ──────────────────────────────────────────────────────────────────────────

 ████████████████████████████░░░░░░░  79%  ep 59/75  eta 10:59

 ╭─ loss ───────────────────────╮╭─ val accuracy ───────────────╮
 │ ⠒⠢⣀⡀                         ││              ⣀⣀⣀⡠⠤⠒⠊⠉        │
 │     ⠈⠑⠤⢄⣀                    ││        ⢀⡠⠤⠔⠒⠉                │
 │          ⠈⠒⠢⠤⣀⡀              ││    ⣀⡠⠒⠁                      │
 ╰──────────────────────────────╯╰──────────────────────────────╯

 train loss        val loss          val acc         best
 1.3527 ▼0.0145    1.4654 ▼0.0133    42.03% ▲0.0039  42.03% (ep 59)

 vram ▇▇▇▇▇▇▇▇▇▇▇▇ 659 MiB      ram ▇▇▇▇▇▇▇▇▇▇▇▇ 1.0 GB
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

| Key          | Action                                            |
| ------------ | ------------------------------------------------- |
| `q`, `esc`   | quit the dashboard (the training run is untouched)|
| `left/right` | toggle full history and the last 30 epochs        |

## The memory panel

Both sparklines are scaled from zero rather than from the series range, so a
steady allocation renders as a **flat line** and real growth is obvious. Scaling
to the range would amplify byte-level jitter into a sawtooth and make a healthy
run look like it is leaking.

Host RAM turns red and shows the delta once it grows more than 0.5 GB above the
run's opening epochs. The leak this was built after only became visible when the
process was OOM-killed at epoch 32.

## Layout

- `source/stream.ts` - tails the JSONL and emits snapshots
- `source/braille.ts` - braille canvas and sparklines
- `source/components/Chart.tsx` - overlaid line charts
- `source/components/Panels.tsx` - header, progress, stats, memory, notes
- `source/app.tsx` - layout and key handling

Charts use the Unicode braille block: each character cell holds a 2x4 grid of
dots, so a chart is eight times the resolution of a block-character plot. That is
what makes a loss curve read as a curve instead of a staircase.
