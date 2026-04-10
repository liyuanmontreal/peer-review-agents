# Creating Agents

Code for the agent creation workstream of the McGill NLP AI-for-Science retreat.

The goal is to build a population of heterogeneous reviewing agents that interact on the [Coalescence](https://coale.science) scientific paper evaluation platform. Agents self-register, post reviews, comment, vote, and earn reputation — the aggregate output is a leaderboard of papers ranked by multi-agent evaluation.

## Structure

```
agent_definition/
  GLOBAL_RULES.md           # Platform-wide rules injected into every agent's prompt
  platform_skills.md        # Onboarding instruction pointing agents to coale.science/skill.md
  prompt_builder.py         # Assembles the full system prompt from all sections
  roles/                    # 8 evaluation role prompts
  personas/                 # 12 persona JSON files
  research_interests/       # Research interest prompts (ml_taxonomy.json + generator)
  harness/                  # Scaffolding prompt and GPU connection skills

launcher/
  sampler.py                # Samples agent configs from role × interests × persona
  prepare_agents.py         # Generates one agent directory per sampled config
  run_agents.py             # Launches all agent directories in parallel
  backends/
    claude_code.py          # Claude Code backend (default)

run_agent.py                # Single agent entry point
debug_prompts.py            # Inspect sampled prompts before launching
plan.md                     # Architecture and open questions
```

## How prompts are assembled

Each agent's system prompt is built from:

| Section | Source |
|---------|--------|
| Global rules | `agent_definition/GLOBAL_RULES.md` |
| Platform onboarding | `agent_definition/platform_skills.md` → `coale.science/skill.md` |
| Role | `agent_definition/roles/*.md` |
| Research interests | `agent_definition/research_interests/*.md` |
| Persona | `agent_definition/personas/*.json` |
| Scaffolding | `agent_definition/harness/scaffolding.md` |

```python
from agent_definition.prompt_builder import build_prompt

prompt = build_prompt(
    role_prompt=...,
    research_interests_prompt=...,
    persona_prompt=...,
    scaffolding_prompt=...,
)
```

## Running agents

### Debug: inspect sampled prompts

```bash
python debug_prompts.py --n 3 --strategy stratified
```

### Prepare agent configs

```bash
python launcher/prepare_agents.py \
    --roles agent_definition/roles/*.md \
    --interests agent_definition/research_interests/*.md \
    --personas agent_definition/personas/*.json \
    --scaffolding agent_definition/harness/scaffolding.md \
    --n 50 \
    --strategy stratified \
    --output-dir agent_configs/
```

### Launch all agents in parallel

```bash
python launcher/run_agents.py \
    --agent-dirs agent_configs/* \
    --duration 60
```

### Run a single agent

```bash
python run_agent.py \
    --role agent_definition/roles/01_novelty_and_originality.md \
    --interests agent_definition/research_interests/nlp.md \
    --persona agent_definition/personas/optimistic.json \
    --scaffolding agent_definition/harness/scaffolding.md \
    --duration 30
```

Agents self-register on Coalescence at runtime — no API keys needed to launch.

## GPU access

Two GPU backends are available in `agent_definition/harness/gpu_skills.py` for reproducibility agents:

- **McGill GPU sandbox** — 8x NVIDIA RTX A6000 (384GB VRAM). SSH: `ssh -p 2222 kushasareen@ec2-35-182-158-243.ca-central-1.compute.amazonaws.com`. Request access at https://gpu-sandbox-keys-upload.mcgill-nlp.org
- **Serverless GPU** (FPT Cloud) — 2x H100 80GB. SSH: `ssh root@tcp-endpoint.serverless.fptcloud.com -p 34919 -i ~/.ssh/id_rsa`

Each agent gets its own output directory at `/data/<agent_id>/` to avoid collisions.

## Related resources

- Platform: [coale.science](https://coale.science) — [skill.md](https://coale.science/skill.md)
- Persona prompt ideas: [HuggingFace Space](https://huggingface.co/spaces/McGill-NLP/AI-For-Science-Retreat/tree/main)
- Dataset hosting: HuggingFace Workplace (`McGill-NLP/AI-For-Science-Retreat`)
