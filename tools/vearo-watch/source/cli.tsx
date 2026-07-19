#!/usr/bin/env node
import React from 'react';
import {render} from 'ink';
import meow from 'meow';
import {App} from './app.js';

const cli = meow(
	`
  Usage
    $ vearo-watch <run.jsonl>

  Watches a Vearo training run. Produce the file from Rust with:

    let mut ui = TrainingMonitor::new("style cnn", "Cuda(0)", 75)
        .with_metrics("run.jsonl");

  The file can be tailed while the run is still going, reopened after the
  dashboard is closed, or replayed once the run has finished. To watch a run on
  a remote box, keep the file synced and point this at the local copy:

    rsync -az --partial box:~/run.jsonl . && vearo-watch run.jsonl

  Controls
    q, esc     quit (does not affect the training run)
    left/right toggle full history and recent window

  Options
    --help     show this
`,
	{importMeta: import.meta},
);

const path = cli.input[0];
if (!path) {
	cli.showHelp(1);
	process.exit(1);
}

render(<App path={path} />);
