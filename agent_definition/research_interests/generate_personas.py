"""Batch-generate reviewer research interest personas from a topic taxonomy.

Reads ml_taxonomy.json and generates personas at any level of the tree —
leaves, parent categories, or both. Each persona is written as a standalone
.md file ready to pass as --interests to run_agent.py.

The taxonomy tree has 4 depth levels:
  depth 0: root ("Machine Learning")           —   1 node
  depth 1: broad areas (e.g., "RL", "NLP")     —  16 nodes
  depth 2: specific topics (mostly leaves)      — 106 nodes
  depth 3: LLM sub-topics                      —   7 nodes

Usage:
    # Generate for all nodes at all depths (excluding root)
    python generate_personas.py

    # Only leaf nodes (default before this change)
    python generate_personas.py --nodes leaves

    # Only broad parent categories (depth 1)
    python generate_personas.py --depth 1

    # Depth 1 and 2 (parents + their children)
    python generate_personas.py --depth 1 2

    # Only specific expertise levels
    python generate_personas.py --levels senior junior

    # Single ad-hoc topic
    python generate_personas.py --topic "AI Safety" "Alignment & RLHF" \
        "Techniques for aligning model behavior with human intent"

    # List all nodes from taxonomy (with depth info)
    python generate_personas.py --list-topics

    # Dry run
    python generate_personas.py --dry-run
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
META_PROMPT_PATH = SCRIPT_DIR / "meta_prompt.md"
TAXONOMY_PATH = SCRIPT_DIR / "ml_taxonomy.json"
OUTPUT_DIR = SCRIPT_DIR / "generated_personas"

EXPERTISE_LEVELS = ["senior", "mid", "junior", "adjacent"]


def load_meta_prompt() -> str:
    return META_PROMPT_PATH.read_text()


def load_taxonomy(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def walk_taxonomy(
    node: dict,
    parent_path: list[str] | None = None,
    depth: int = 0,
) -> list[dict]:
    """Walk the taxonomy tree and extract ALL nodes with their metadata.

    Returns a flat list of dicts, one per node, with:
      - name: the node's own name
      - track_name: ancestor path (excluding root and self)
      - subtrack_name: the node's own name (what we're generating a persona for)
      - full_path: complete path from root
      - depth: depth in the tree (0 = root)
      - is_leaf: whether this node has no children
      - children_names: list of direct child names (for context in the prompt)
    """
    parent_path = parent_path or []
    current_path = parent_path + [node["name"]]
    children = node.get("children", [])
    is_leaf = len(children) == 0

    # Build track_name from path (skip root, skip self)
    path_without_root = current_path[1:]  # drop "Machine Learning"
    if len(path_without_root) <= 1:
        track_name = path_without_root[0] if path_without_root else node["name"]
        subtrack_name = track_name
    else:
        subtrack_name = path_without_root[-1]
        track_name = " > ".join(path_without_root[:-1])

    # Build slugified path segments for folder hierarchy (excluding root)
    path_slugs = [slugify(p) for p in path_without_root]

    entry = {
        "name": node["name"],
        "track_name": track_name,
        "subtrack_name": subtrack_name,
        "full_path": " > ".join(current_path),
        "path_slugs": path_slugs,
        "depth": depth,
        "is_leaf": is_leaf,
        "children_names": [c["name"] for c in children],
    }

    results = [entry]
    for child in children:
        results.extend(walk_taxonomy(child, current_path, depth + 1))
    return results


def filter_nodes(
    nodes: list[dict],
    depths: list[int] | None = None,
    node_filter: str = "all",
) -> list[dict]:
    """Filter nodes by depth and/or leaf status.

    Args:
        nodes: flat list from walk_taxonomy
        depths: if set, only include nodes at these depths
        node_filter: "all" (every node), "leaves" (only leaves),
                     "parents" (only non-leaf nodes)
    """
    # Always exclude root (depth 0)
    filtered = [n for n in nodes if n["depth"] > 0]

    if depths is not None:
        filtered = [n for n in filtered if n["depth"] in depths]

    if node_filter == "leaves":
        filtered = [n for n in filtered if n["is_leaf"]]
    elif node_filter == "parents":
        filtered = [n for n in filtered if not n["is_leaf"]]

    return filtered


def build_track_description(node: dict) -> str:
    """Build a descriptive string for the prompt template from node metadata."""
    desc = f"Research area focusing on {node['subtrack_name']}."

    if node["track_name"] != node["subtrack_name"]:
        desc = f"Research area within {node['track_name']}, focusing on {node['subtrack_name']}."

    desc += f" Full taxonomy path: {node['full_path']}."

    if node["children_names"]:
        children_str = ", ".join(node["children_names"])
        desc += f" This area encompasses the following sub-areas: {children_str}."

    return desc


def render_prompt(
    template: str,
    track_name: str,
    subtrack_name: str,
    track_description: str,
    expertise_level: str,
) -> str:
    return (
        template
        .replace("{{track_name}}", track_name)
        .replace("{{subtrack_name}}", subtrack_name)
        .replace("{{track_description}}", track_description)
        .replace("{{expertise_level}}", expertise_level)
    )


def call_llm(prompt: str, model: str, provider: str) -> str:
    if provider == "anthropic":
        return _call_anthropic(prompt, model)
    elif provider == "openai":
        return _call_openai(prompt, model)
    elif provider == "gemini":
        return _call_gemini(prompt, model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(prompt: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


def _call_openai(prompt: str, model: str) -> str:
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_gemini(prompt: str, model: str) -> str:
    from google import genai

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text


def slugify(text: str) -> str:
    return (
        text.lower()
        .replace(" > ", "_")
        .replace(" ", "_")
        .replace("/", "_")
        .replace("-", "_")
        .replace("&", "and")
    )


def generate_persona(
    template: str,
    node: dict,
    expertise_level: str,
    model: str,
    provider: str,
    output_dir: Path,
    dry_run: bool = False,
) -> dict:
    """Generate a single persona and write it to disk."""
    track_description = build_track_description(node)
    prompt = render_prompt(
        template, node["track_name"], node["subtrack_name"],
        track_description, expertise_level,
    )

    tag = "leaf" if node["is_leaf"] else "broad"
    label = f"{node['full_path']} [{expertise_level}] ({tag})"
    print(f"  Generating: {label}")

    # Build output path first so we can check if it already exists
    path_slugs = node.get("path_slugs", [slugify(node["subtrack_name"])])
    if len(path_slugs) <= 1:
        subdir = output_dir / expertise_level
    else:
        subdir = output_dir / expertise_level / Path(*path_slugs[:-1])
    filename = f"{path_slugs[-1]}.md"
    filepath = subdir / filename

    if filepath.exists():
        print(f"    [skip] Already exists: {filepath}")
        return {
            "label": label,
            "file": str(filepath),
            "depth": node["depth"],
            "is_leaf": node["is_leaf"],
            "expertise_level": expertise_level,
            "status": "skipped",
        }

    if dry_run:
        print(f"    [dry-run] Would call {provider}/{model}")
        return {
            "label": label,
            "depth": node["depth"],
            "is_leaf": node["is_leaf"],
            "expertise_level": expertise_level,
            "status": "dry-run",
        }

    persona_text = call_llm(prompt, model, provider)
    subdir.mkdir(parents=True, exist_ok=True)
    filepath.write_text(persona_text)
    print(f"    -> {filepath}")

    return {
        "label": label,
        "file": str(filepath),
        "depth": node["depth"],
        "is_leaf": node["is_leaf"],
        "expertise_level": expertise_level,
        "status": "ok",
    }


def generate_from_taxonomy(
    taxonomy_path: Path | None = None,
    output_dir: Path | None = None,
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    levels: list[str] | None = None,
    depths: list[int] | None = None,
    node_filter: str = "all",
    dry_run: bool = False,
) -> list[dict]:
    """Generate personas from a taxonomy JSON."""
    template = load_meta_prompt()
    taxonomy = load_taxonomy(taxonomy_path or TAXONOMY_PATH)
    all_nodes = walk_taxonomy(taxonomy)
    nodes = filter_nodes(all_nodes, depths=depths, node_filter=node_filter)
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = levels or EXPERTISE_LEVELS

    n_parents = sum(1 for n in nodes if not n["is_leaf"])
    n_leaves = sum(1 for n in nodes if n["is_leaf"])
    print(f"Taxonomy: {taxonomy_path or TAXONOMY_PATH}")
    print(f"Selected {len(nodes)} nodes ({n_parents} parents, {n_leaves} leaves)")
    print(f"Expertise levels: {', '.join(levels)}")
    print(f"Total personas: {len(nodes) * len(levels)}\n")

    results = []
    for node in nodes:
        print(f"\n[depth={node['depth']}] {node['full_path']}")
        for level in levels:
            result = generate_persona(
                template=template,
                node=node,
                expertise_level=level,
                model=model,
                provider=provider,
                output_dir=output_dir,
                dry_run=dry_run,
            )
            results.append(result)

    _write_manifest(results, output_dir)
    return results


def generate_single_topic(
    track_name: str,
    subtrack_name: str,
    track_description: str,
    output_dir: Path | None = None,
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    levels: list[str] | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Generate personas for a single topic across specified levels."""
    template = load_meta_prompt()
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    levels = levels or EXPERTISE_LEVELS

    node = {
        "name": subtrack_name,
        "track_name": track_name,
        "subtrack_name": subtrack_name,
        "full_path": f"{track_name} > {subtrack_name}",
        "path_slugs": [slugify(track_name), slugify(subtrack_name)],
        "depth": -1,
        "is_leaf": True,
        "children_names": [],
    }

    results = []
    print(f"\n{track_name} > {subtrack_name}")
    for level in levels:
        result = generate_persona(
            template=template,
            node=node,
            expertise_level=level,
            model=model,
            provider=provider,
            output_dir=output_dir,
            dry_run=dry_run,
        )
        results.append(result)

    _write_manifest(results, output_dir)
    return results


def _write_manifest(results: list[dict], output_dir: Path):
    manifest_path = output_dir / "manifest.json"
    existing = []
    if manifest_path.exists():
        with open(manifest_path) as f:
            existing = json.load(f)
    seen = {r["label"] for r in results}
    merged = [r for r in existing if r["label"] not in seen] + results
    with open(manifest_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nManifest written to {manifest_path} ({len(merged)} entries)")


def list_topics(taxonomy_path: Path | None, depths: list[int] | None, node_filter: str):
    """Print all nodes matching the filters."""
    taxonomy = load_taxonomy(taxonomy_path or TAXONOMY_PATH)
    all_nodes = walk_taxonomy(taxonomy)
    nodes = filter_nodes(all_nodes, depths=depths, node_filter=node_filter)

    for i, node in enumerate(nodes, 1):
        tag = "leaf" if node["is_leaf"] else f"parent, {len(node['children_names'])} children"
        print(f"  {i:3d}. [depth={node['depth']}] {node['full_path']}  ({tag})")

    n_parents = sum(1 for n in nodes if not n["is_leaf"])
    n_leaves = sum(1 for n in nodes if n["is_leaf"])
    print(f"\n{len(nodes)} nodes ({n_parents} parents, {n_leaves} leaves)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate reviewer research interest personas from a topic taxonomy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--taxonomy", type=Path, default=None,
        help="Path to taxonomy JSON (default: ml_taxonomy.json)",
    )
    input_group.add_argument(
        "--topic", nargs=3, metavar=("TRACK", "SUBTRACK", "DESCRIPTION"),
        help="Generate for a single topic",
    )

    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model to use")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini"], default="anthropic")
    parser.add_argument(
        "--levels", nargs="+", choices=EXPERTISE_LEVELS, default=None,
        help="Expertise levels to generate (default: all four)",
    )
    parser.add_argument(
        "--depth", type=int, nargs="+", default=None,
        help="Tree depths to include (e.g., --depth 1 2). Default: all depths except root",
    )
    parser.add_argument(
        "--nodes", choices=["all", "leaves", "parents"], default="all",
        help="Which nodes to generate for: all, leaves only, or parents only (default: all)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview without calling API")
    parser.add_argument(
        "--list-topics", action="store_true",
        help="List matching nodes from taxonomy and exit",
    )
    args = parser.parse_args()

    if args.list_topics:
        list_topics(args.taxonomy, args.depth, args.nodes)
        return

    if args.topic:
        generate_single_topic(
            track_name=args.topic[0],
            subtrack_name=args.topic[1],
            track_description=args.topic[2],
            output_dir=args.output,
            model=args.model,
            provider=args.provider,
            levels=args.levels,
            dry_run=args.dry_run,
        )
    else:
        generate_from_taxonomy(
            taxonomy_path=args.taxonomy,
            output_dir=args.output,
            model=args.model,
            provider=args.provider,
            levels=args.levels,
            depths=args.depth,
            node_filter=args.nodes,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
