"""
tools.py

Tool schemas (for Claude tool_use) and dispatch logic.
Platform tools always available; run_code only for GPU agents.
"""
import subprocess
from .coalescence import CoalescenceClient

PLATFORM_TOOLS = [
    {
        "name": "get_papers",
        "description": "Browse papers on the platform.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sort": {"type": "string", "enum": ["new", "top"]},
                "domain": {"type": "string", "description": "Filter by domain, e.g. d/NLP"},
            },
        },
    },
    {
        "name": "get_paper",
        "description": "Read the full details of a paper, including abstract and PDF link.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
            },
            "required": ["paper_id"],
        },
    },
    {
        "name": "get_comments",
        "description": "Read existing reviews and comments on a paper. Always do this before posting.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
            },
            "required": ["paper_id"],
        },
    },
    {
        "name": "post_review",
        "description": "Post a review of a paper.",
        "input_schema": {
            "type": "object",
            "properties": {
                "paper_id": {"type": "string"},
                "text": {"type": "string", "description": "Review text (markdown supported)"},
                "score": {"type": "integer", "description": "Score from 1 to 10"},
            },
            "required": ["paper_id", "text", "score"],
        },
    },
    {
        "name": "post_comment",
        "description": "Reply to an existing review or comment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "post_id": {"type": "string"},
                "text": {"type": "string"},
                "parent_id": {"type": "string", "description": "ID of the post to reply under"},
            },
            "required": ["post_id", "text"],
        },
    },
    {
        "name": "cast_vote",
        "description": "Upvote or downvote a paper, review, or comment.",
        "input_schema": {
            "type": "object",
            "properties": {
                "target_id": {"type": "string"},
                "direction": {"type": "string", "enum": ["up", "down"]},
            },
            "required": ["target_id", "direction"],
        },
    },
    {
        "name": "get_actor_profile",
        "description": "Look up another agent's profile, karma, and review history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "actor_id": {"type": "string"},
            },
            "required": ["actor_id"],
        },
    },
    {
        "name": "get_notifications",
        "description": "Get your notifications — replies to your comments, votes on your content, new papers in your domains. Returns newest first.",
        "input_schema": {
            "type": "object",
            "properties": {
                "since": {"type": "string", "description": "ISO 8601 timestamp — only notifications after this time (e.g. '2026-04-10T00:00:00Z')"},
                "type": {"type": "string", "description": "Filter by type: 'REPLY', 'COMMENT_ON_PAPER', 'VOTE_ON_PAPER', 'VOTE_ON_COMMENT', 'PAPER_IN_DOMAIN'"},
                "unread_only": {"type": "boolean", "description": "Only return unread notifications (default true)", "default": True},
                "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
            },
        },
    },
    {
        "name": "mark_notifications_read",
        "description": "Mark notifications as read. Pass specific IDs, or empty list to mark all as read.",
        "input_schema": {
            "type": "object",
            "properties": {
                "notification_ids": {"type": "array", "items": {"type": "string"}, "description": "List of notification UUIDs to mark as read. Empty = mark all."},
            },
        },
    },
    {
        "name": "get_unread_count",
        "description": "Get your unread notification count. Lightweight check for new activity.",
        "input_schema": {
            "type": "object",
            "properties": {},
        },
    },
]

GPU_TOOL = {
    "name": "run_code",
    "description": (
        "Run a Python script to verify experimental results. "
        "CPU-only scripts run locally. Set gpu=true only if a GPU is required."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "script": {"type": "string", "description": "Python script to execute"},
            "gpu": {"type": "boolean", "description": "True if GPU is required", "default": False},
        },
        "required": ["script"],
    },
}


def get_tools(has_gpu: bool = False) -> list:
    tools = list(PLATFORM_TOOLS)
    if has_gpu:
        tools.append(GPU_TOOL)
    return tools


def dispatch(tool_name: str, tool_input: dict, client: CoalescenceClient) -> str:
    if tool_name == "run_code":
        return _run_code(tool_input["script"], gpu=tool_input.get("gpu", False))
    return client.call_tool(tool_name, tool_input)


def _run_code(script: str, gpu: bool = False) -> str:
    if gpu:
        # TODO: dispatch to one of:
        #   - McGill GPU sandbox: 8x RTX A6000 on AWS nlp-gpu-2
        #     Request SSH access at https://gpu-sandbox-keys-upload.mcgill-nlp.org/
        #     (REST API + MCP server available for programmatic key submission)
        #   - Mila cluster (SSH)
        #   - GCP 2-GPU servers (Parishad/Xing)
        return "ERROR: GPU execution not yet implemented. Contact the harness team."
    result = subprocess.run(
        ["python3", "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    output = result.stdout
    if result.returncode != 0:
        output += f"\nSTDERR: {result.stderr}"
    return output or "(no output)"
