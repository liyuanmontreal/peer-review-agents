"""
harness.py

Multi-turn agent loop. Calls the LLM, dispatches tool calls, feeds
results back, and repeats until the agent stops or the turn budget runs out.
"""
import os
import anthropic
from .koala import KoalaClient
from .tools import get_tools, dispatch

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TURNS = 20


class Agent:
    def __init__(
        self,
        system_prompt: str,
        koala_api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        max_turns: int = DEFAULT_MAX_TURNS,
        has_gpu: bool = False,
        agent_name: str | None = None,
        smoke_artifact_dir: str | None = None,
    ):
        self.system_prompt = system_prompt
        self.model = model
        self.max_turns = max_turns
        self.agent_name = agent_name
        self._smoke_artifact_dir = smoke_artifact_dir
        self.tools = get_tools(has_gpu=has_gpu)
        self.llm = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        on_write = None
        if agent_name == "gsr_agent":
            from .gsr_artifacts import emit_gsr_artifacts
            on_write = emit_gsr_artifacts
        self.koala = KoalaClient(api_key=koala_api_key, on_write=on_write)
        self.history = []

    def run(self):
        print(f"[agent] starting — model={self.model}, max_turns={self.max_turns}")
        if self.agent_name == "gsr_agent":
            from .gsr_artifacts import write_smoke_artifact, write_local_artifact_smoke, GSR_ARTIFACT_DIR
            artifact_dir = self._smoke_artifact_dir or GSR_ARTIFACT_DIR
            write_smoke_artifact(agent_name=self.agent_name, artifact_dir=artifact_dir)
            write_local_artifact_smoke(agent_name=self.agent_name, artifact_dir=artifact_dir)

        for turn in range(self.max_turns):
            response = self.llm.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=self.history,
            )

            print(f"[turn {turn + 1}] stop_reason={response.stop_reason}")
            self.history.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                print("[agent] done.")
                break

            if response.stop_reason != "tool_use":
                print(f"[agent] unexpected stop_reason: {response.stop_reason}")
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                print(f"  -> {block.name}({block.input})")
                result = dispatch(block.name, block.input, self.koala)
                preview = result[:200] + ("..." if len(result) > 200 else "")
                print(f"     {preview}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            self.history.append({"role": "user", "content": tool_results})

        else:
            print(f"[agent] reached max turns ({self.max_turns}).")
