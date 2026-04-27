"""
harness.py

Multi-turn agent loop. Calls the LLM, dispatches tool calls, feeds
results back, and repeats until the agent stops or the turn budget runs out.
"""
import json
import os
import anthropic
from .koala import KoalaClient
from .tools import get_tools, dispatch
from .window import koala_window_state

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
        self._papers_inspected: int = 0

    def run(self):
        self._papers_inspected = 0
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

                if block.name == "get_paper":
                    if self._papers_inspected >= 2:
                        result = json.dumps({
                            "error": "paper inspection limit reached",
                            "limit": 2,
                            "inspected": self._papers_inspected,
                        })
                        print(f"     [competition] SKIP get_paper limit reached ({self._papers_inspected}/2)")
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        })
                        continue
                    self._papers_inspected += 1
                    print(f"[competition] SELECT get_paper #{self._papers_inspected} paper_id={block.input.get('paper_id', '?')}")

                result = dispatch(block.name, block.input, self.koala)

                if block.name == "get_papers":
                    result = _filter_papers_by_window(result)

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


def _filter_papers_by_window(result: str) -> str:
    try:
        papers = json.loads(result)
    except (json.JSONDecodeError, ValueError):
        return result
    if not isinstance(papers, list):
        return result

    open_papers = []
    for paper in papers:
        pid = paper.get("id", paper.get("paper_id", "?"))
        ws = koala_window_state(paper)
        print(
            f"[competition] WINDOW paper_id={pid} phase={ws['phase']} "
            f"open={ws['open']} seconds_left={ws['seconds_left']:.0f}"
        )
        if ws["open"]:
            paper["_window"] = ws
            open_papers.append(paper)
        else:
            print(f"[competition] SKIP paper_id={pid} window closed")

    open_papers.sort(key=lambda p: (0 if p.get("status") == "deliberating" else 1))

    for paper in open_papers:
        pid = paper.get("id", paper.get("paper_id", "?"))
        print(
            f"[competition] SELECT paper_id={pid} status={paper.get('status')} "
            f"phase={paper.get('_window', {}).get('phase', '?')}"
        )

    return json.dumps(open_papers)
