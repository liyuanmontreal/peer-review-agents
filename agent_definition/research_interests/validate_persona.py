"""Bias validation pass for generated personas.

Screens each persona for evaluative language, prescriptive statements,
and methodology preferences that could bias the reviewer agent.

Can run as:
  1. LLM-based validation: sends each persona to an LLM for nuanced analysis
  2. Rule-based validation: fast keyword/pattern check (no API needed)
"""

import argparse
import re
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PERSONAS_DIR = SCRIPT_DIR / "generated_personas"

# Words and phrases that signal evaluative bias
BANNED_WORDS = [
    "superior", "inferior", "state-of-the-art", "sota",
    "promising", "unpromising", "powerful", "elegant",
    "simple", "naive", "trivial", "limited", "outdated",
    "dominant", "leading", "best", "worst", "optimal",
    "suboptimal", "gold standard", "breakthrough",
]

PRESCRIPTIVE_PATTERNS = [
    r"\bshould\b", r"\bmust\b", r"\bideally\b",
    r"\bit is important\b", r"\bit is essential\b",
    r"\bit is crucial\b", r"\bit is necessary\b",
    r"\bexpects?\b", r"\bvalues?\s+\w+\s+(analysis|rigor|novelty)",
    r"\bstrong papers?\b", r"\bweak papers?\b",
    r"\bgood papers?\b", r"\bbad papers?\b",
]

PREFERENCE_PATTERNS = [
    r"\bhas shown the most\b", r"\bhave shown the most\b",
    r"\bmost promising\b", r"\bmost effective\b",
    r"\bkey challenge\b", r"\bthe future of\b",
    r"\bis the (?:primary|main|central) (?:approach|method)\b",
]

VALIDATION_PROMPT = """Analyze the following reviewer persona for bias. Identify any sentences that:

1. Express a preference for one methodology over another
2. Set a quality bar or define what makes work strong/weak
3. Use prescriptive language (should, must, ideally, it is important)
4. Frame any research direction as more promising, dominant, or outdated
5. Imply what a paper in this area "needs" to have
6. Use comparative value language (superior, powerful, elegant, naive, limited)

For each issue found, return:
- The exact sentence
- The type of bias (methodology preference / quality bar / prescriptive / temporal bias / value language)
- A brief explanation of why it is problematic

If no issues are found, return exactly: PASS

Persona to validate:
---
{persona_text}
---"""


def rule_based_check(text: str) -> list[dict]:
    """Fast keyword/pattern screening. Returns list of findings."""
    findings = []
    lines = text.split("\n")

    for i, line in enumerate(lines, 1):
        line_lower = line.lower().strip()
        if not line_lower or line_lower.startswith("---") or line_lower.startswith("#"):
            continue

        # Check banned words
        for word in BANNED_WORDS:
            if word in line_lower:
                findings.append({
                    "line": i,
                    "text": line.strip(),
                    "type": "banned_word",
                    "detail": f"Contains banned word: '{word}'",
                })

        # Check prescriptive patterns
        for pattern in PRESCRIPTIVE_PATTERNS:
            if re.search(pattern, line_lower):
                findings.append({
                    "line": i,
                    "text": line.strip(),
                    "type": "prescriptive",
                    "detail": f"Matches prescriptive pattern: {pattern}",
                })

        # Check preference patterns
        for pattern in PREFERENCE_PATTERNS:
            if re.search(pattern, line_lower):
                findings.append({
                    "line": i,
                    "text": line.strip(),
                    "type": "preference",
                    "detail": f"Matches preference pattern: {pattern}",
                })

    return findings


def llm_check(text: str, model: str, provider: str) -> str:
    """LLM-based nuanced bias analysis."""
    prompt = VALIDATION_PROMPT.format(persona_text=text)

    if provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    elif provider == "openai":
        import openai
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
    elif provider == "gemini":
        from google import genai
        client = genai.Client()
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(max_output_tokens=2048),
        )
        return response.text
    else:
        raise ValueError(f"Unknown provider: {provider}")


def extract_persona_text(filepath: Path) -> str:
    """Read a persona file, stripping the YAML frontmatter."""
    content = filepath.read_text()
    # Strip frontmatter between --- markers
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            return parts[2].strip()
    return content.strip()


def validate_file(filepath: Path, use_llm: bool, model: str, provider: str) -> dict:
    """Validate a single persona file. Returns result dict."""
    text = extract_persona_text(filepath)

    result = {
        "file": str(filepath),
        "rule_based": rule_based_check(text),
    }

    if use_llm:
        result["llm_analysis"] = llm_check(text, model, provider)

    n_issues = len(result["rule_based"])
    llm_pass = result.get("llm_analysis", "").strip() == "PASS" if use_llm else None

    result["status"] = "PASS" if n_issues == 0 and (llm_pass is None or llm_pass) else "FAIL"
    return result


def validate_all(
    personas_dir: Path | None = None,
    use_llm: bool = False,
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
) -> list[dict]:
    personas_dir = personas_dir or DEFAULT_PERSONAS_DIR

    if not personas_dir.exists():
        print(f"No personas directory found at {personas_dir}")
        return []

    persona_files = sorted(personas_dir.glob("*.md"))
    if not persona_files:
        print(f"No persona files found in {personas_dir}")
        return []

    results = []
    for filepath in persona_files:
        print(f"Validating: {filepath.name}")
        result = validate_file(filepath, use_llm, model, provider)

        if result["status"] == "PASS":
            print(f"  PASS")
        else:
            print(f"  FAIL — {len(result['rule_based'])} rule-based issues")
            for finding in result["rule_based"]:
                print(f"    L{finding['line']}: [{finding['type']}] {finding['detail']}")
                print(f"           \"{finding['text'][:80]}...\"" if len(finding['text']) > 80 else f"           \"{finding['text']}\"")
            if use_llm and result.get("llm_analysis"):
                print(f"  LLM analysis:\n    {result['llm_analysis'][:500]}")

        results.append(result)

    # Summary
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate generated personas for bias")
    parser.add_argument("--dir", type=Path, default=None, help="Directory containing persona .md files")
    parser.add_argument("--file", type=Path, default=None, help="Validate a single persona file")
    parser.add_argument("--llm", action="store_true", help="Also run LLM-based validation")
    parser.add_argument("--model", default="claude-sonnet-4-20250514", help="Model for LLM validation")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini"], default="anthropic")
    args = parser.parse_args()

    if args.file:
        result = validate_file(args.file, args.llm, args.model, args.provider)
        if result["status"] == "FAIL":
            sys.exit(1)
    else:
        results = validate_all(args.dir, args.llm, args.model, args.provider)
        if any(r["status"] == "FAIL" for r in results):
            sys.exit(1)


if __name__ == "__main__":
    main()
