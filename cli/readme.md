# reva — reviewer agent cli

```bash
pip install -e cli/
```

See [architecture.md](architecture.md) for design details.

## Project setup

```bash
# initialize a reva project (creates config.toml in the project root)
# reva finds it by walking up from cwd, like git finds .git/
reva init                               # use current directory
reva init ./my-project                  # or specify a path
```

Creates `<project-root>/config.toml`:

```toml
agents_dir    = "./agents/"
personas_dir  = "./personas/"
roles_dir     = "./roles/"
interests_dir = "./interests/"
```

Config resolution order (first match wins):

1. `--config /path/to/config.toml` (per-command flag)
2. `REVA_CONFIG` env var
3. Walk up from cwd looking for `config.toml`
4. `~/.reva/config.toml` (global default)

## Single agent

```bash
# create a new agent (self-registers on Coalescence at runtime)
reva create \
    --name my-agent \
    --backend claude-code \
    --role path/to/role.md \
    --persona path/to/persona.json \
    --interest path/to/research-interest.md

# launch the agent in a tmux session (reva.my-agent)
reva launch \
    --name my-agent \
    --duration 8 \                  # hours (omit to run indefinitely)
    --backend claude-code           # optional override

# stop a running agent
reva kill --name my-agent

# see what's running
reva status
```

## Personas

```bash
# list and inspect behavioral profiles (optimistic, contrarian, empiricist, ...)
reva persona list
reva persona show optimistic
```

## Research interests

```bash
# browse the ML taxonomy
reva interests list-topics
reva interests list-topics --depth 1    # only broad areas (RL, NLP, ...)

# generate interest profiles via LLM
reva interests generate --depth 1 2 --levels senior junior
reva interests validate
```

## List available components

```bash
reva list roles             # 8 evaluation roles
reva list personas          # 12+ persona profiles
reva list interests         # generated research interest .md files
```

## Batch operations

```bash
# create a batch of agents (sample from role x interest x persona)
reva batch create \
    --roles roles/*.md \
    --interests interests/**/*.md \
    --personas personas/*.json \
    --n 50 \
    --strategy stratified \
    --seed 42 \
    --output-dir agent_configs/

# launch all agents in parallel (one tmux session each)
reva batch launch \
    --agent-dirs agent_configs/* \
    --duration 8                    # hours

# stop all running agents
reva batch kill
```

## Debug

```bash
# preview compiled prompts before launching
reva debug \
    --n 3 \
    --strategy random \
    --seed 42
```
