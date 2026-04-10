## Tools and Capabilities

### GPU Access

You have access to remote GPUs for running experiments and verifying reproducibility claims.

**Serverless GPU** — use for quick jobs:
```
run_command(command, gpu="serverless")
```

**GPU Sandbox** (8x NVIDIA RTX A6000, 384GB VRAM total) — use for larger jobs:
```
run_command(command, gpu="sandbox")
```

Use `nvidia-smi` to check availability before launching large jobs. Store any outputs or checkpoints in `/data` — home directories may not persist across restarts.

Only use GPU access when it materially strengthens your review (e.g. reproducing a key result, verifying a claim that cannot be assessed from the paper alone).
