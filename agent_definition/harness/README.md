# Harness / Scaffolding

*This README was seeded from the project discussion summary. The harness subteam should update this with their own approach.*

---

The harness owns the scaffolding prompt and GPU connection skills.

Current contents:
- `scaffolding.md` — describes available GPU tools to the agent; passed into `prompt_builder.build_prompt()` as `scaffolding_prompt`
- `gpu_skills.py` — SSH-based GPU execution for reproducibility agents

### GPU backends

| Class | Server | Notes |
|-------|--------|-------|
| `ServerlessGPUSkill` | FPT Cloud (2x H100 80GB) | `ssh root@tcp-endpoint.serverless.fptcloud.com -p 34919` |
| `GPUSandboxSkill` | McGill AWS (8x RTX A6000) | `ssh -p 2222 kushasareen@ec2-35-182-158-243.ca-central-1.compute.amazonaws.com` |

Each agent writes to `/data/<agent_id>/` on the sandbox to avoid collisions.
