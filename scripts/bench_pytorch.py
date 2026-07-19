"""PyTorch side of the Vearo benchmark.

Builds the same model, input shape and step count as
`crates/vearo/examples/bench_cnn.rs` and reports the same metrics, so the two
outputs can be read side by side.

    cargo run --release -p vearo --example bench_cnn
    python3 scripts/bench_pytorch.py --threads 1     # like for like
    python3 scripts/bench_pytorch.py                 # all cores

Vearo's CPU backend is single-threaded, so `--threads 1` is the comparison that
isolates the implementation. The default run says what PyTorch actually delivers
on this machine, which is the number a user would feel, and the gap between the
two is the parallelism Vearo is leaving unused.

Peak memory comes from VmHWM in /proc/self/status, the high-water mark of
resident set size, matching what the Rust side reports.
"""

import argparse
import time

import torch
import torch.nn as nn

BATCH = 32
CHANNELS = 3
SIDE = 32
CLASSES = 10
WARMUP = 5
STEPS = 20


def peak_rss_mib():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


class BenchCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(CHANNELS, 16, 3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(32 * 8 * 8, CLASSES)

    def forward(self, x):
        h = self.pool1(torch.relu(self.conv1(x)))
        h = self.pool2(torch.relu(self.conv2(h)))
        return self.fc(h.reshape(BATCH, 32 * 8 * 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=None, help="torch intra-op threads")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    torch.manual_seed(0)

    # Same deterministic input the Rust side builds, so neither is measuring
    # data generation.
    n = BATCH * CHANNELS * SIDE * SIDE
    xs = torch.tensor(
        [__import__("math").sin(i * 0.017) for i in range(n)], dtype=torch.float32
    ).reshape(BATCH, CHANNELS, SIDE, SIDE)
    ys = torch.tensor([i % CLASSES for i in range(BATCH)], dtype=torch.long)

    model = BenchCnn()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    loss_fn = nn.CrossEntropyLoss()

    rss_before = peak_rss_mib()

    def step():
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(xs), ys)
        loss.backward()
        opt.step()
        return loss.item()

    for _ in range(WARMUP):
        step()

    start = time.perf_counter()
    last = 0.0
    for _ in range(STEPS):
        last = step()
    elapsed = time.perf_counter() - start

    rss_after = peak_rss_mib()

    print(f"framework      pytorch {torch.__version__} (cpu, {torch.get_num_threads()} threads)")
    print(f"model          conv(3-16) pool conv(16-32) pool fc({32 * 8 * 8}-{CLASSES})")
    print(f"input          [{BATCH}, {CHANNELS}, {SIDE}, {SIDE}]")
    print(f"steps          {STEPS} (after {WARMUP} warmup)")
    print(f"final loss     {last:.6f}")
    print(f"ms/step        {elapsed * 1000.0 / STEPS:.2f}")
    print(f"peak rss mib   {rss_after:.1f}")
    print(f"rss growth mib {rss_after - rss_before:.1f}")


if __name__ == "__main__":
    main()
