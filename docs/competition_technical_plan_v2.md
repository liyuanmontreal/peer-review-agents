# Technical Plan for the Koala Science ICML 2026 Agent Review Competition


## 1. Project Goal

The goal of this project is to build a competitive **GSR Meta-Review Agent** for the Koala Science ICML 2026 Agent Review Competition by leveraging and extending our existing **GSR (Good Samaritan Review)** system. This agent will read papers, monitor multi-agent discussions, publish high-quality comments during the review phase, and submit a calibrated verdict during the verdict phase when the competition conditions are satisfied. The primary objective is to produce decisions that are as close as possible to the actual ICML 2026 accept/reject outcomes while also generating useful, evidence-grounded contributions that other agents may cite.



## 2. Agent Design

The GSR Meta-Review Agent is designed as an evidence-grounded, literature-aware, decision-calibrated meta-reviewer rather than a general-purpose reviewer. Its two main goals are: first, to publish short, concrete, verifiable, and citation-friendly comments during the discussion stage; and second, to submit a more accurate 0–10 verdict during the verdict stage by integrating the paper content, the discussion from other agents, the novelty of the contribution, the severity of identified issues,and the implicit standards of ICML reviewing.

Internally, the agent consists of five modules.

The **Evidence Checker** reads comments from other agents, extracts verifiable claims from those comments, and uses the existing GSR evidence retrieval and verification pipeline to determine whether those claims are supported, contradicted, or insufficiently supported by the paper. Its main role is to generate grounded corrections or evidence-backed additions to the discussion.

The **Novelty Checker** evaluates whether the paper’s claimed contribution is genuinely novel, moderately incremental, or likely overstated relative to prior work. This module is especially important because other agents may systematically overestimate novelty. It uses the **Lacuna API** as the primary literature search backend to retrieve related work and compare the current paper against relevant prior papers. If needed, additional literature sources such as Semantic Scholar, arXiv, or Crossref can be used as fallback or supplemental sources. The module is not intended to perform exhaustive literature review, but rather to detect whether the current discussion is materially over- or under-estimating the paper’s novelty.

The **Severe Issue Scanner** is a dedicated phase that explicitly looks for decision-critical or desk-reject-level failures that general LLM reasoning often misses unless prompted separately. This includes issues such as missing bibliography, missing experiments, lack of empirical evidence for empirical claims, absent or fundamentally inadequate baselines, broken or suspicious citations, major methodological omissions, or severe mismatch between claimed contributions and presented evidence. The purpose of this module is not to enumerate ordinary weaknesses, but to identify flaws that should strongly affect the final verdict.

The **Decision Calibrator** is responsible for estimating whether an identified issue is actually decision-critical. It does not simply penalize every weakness. Instead, it integrates the paper’s strengths, weaknesses, novelty signal, severe-issue signal,and the external discussion consensus into a structured internal reasoning state. Crucially, the revised design uses a **two-stage scoring mechanism**: an instruct-tuned model produces structured reasoning and qualitative assessments, and then a **base model (non-instruct-tuned)** maps that reasoning into the final numerical 0–10 score. This is motivated by the expectation that base models may be better calibrated for numerical outputs than instruct-tuned models.

The **Opportunity Manager** decides which papers and threads are worth participating in, controls karma usage, checks whether verdict eligibility conditions are met, and maintains a pool of external comments that can later be cited in a verdict. It also decides when the expected value of joining a discussion is too low relative to karma cost or too risky due to insufficient future verdict eligibility.

## 3. Overall Architecture

The system follows a **three-layer architecture**: a **Koala Integration Layer**, a **GSR Core Layer**, and a **Competition Strategy Layer**.

The **Koala Integration Layer** is responsible for interacting with the competition platform. It handles discovery of newly opened papers, synchronization of paper state, retrieval of discussion comments, posting comments, submitting verdicts, and logging platform-side events and API errors. Conceptually, this layer is the orchestration interface between our system and the Koala competition environment.

The **GSR Core Layer** is the main technical asset we already have. It includes paper PDF download and parsing, section extraction, text/table/figure evidence object construction, hybrid retrieval (BM25 + embeddings), claim verification, and the broader claim-to-evidence pipeline already developed in GSR. In the Koala setting, discussion comments become a new type of “review source.” The system extracts verifiable claims from these comments and then reuses the GSR pipeline to retrieve relevant evidence from the paper and verify those claims.

The **Competition Strategy Layer** sits on top of the GSR core and contains the five internal modules: Evidence Checker, Novelty Checker, Severe Issue Scanner, Decision Calibrator, and Opportunity Manager. The Evidence Checker validates concrete claims made in the discussion. The Novelty Checker assesses whether the novelty of the method is being overstated or understated relative to prior work. The Severe Issue Scanner searches explicitly for fatal or decision-critical flaws. The Decision Calibrator maintains an internal model of the paper’s overall quality and likely ICML decision boundary, and then produces a calibrated final score during the verdict phase. The Opportunity Manager handles paper selection, karma budgeting, verdict eligibility checks, and management of candidate external comments for citation.

The entire system is driven by a scheduler that periodically polls or syncs the platform and executes different actions depending on the current phase of each paper. During the review window it focuses on monitoring,evidence checking, novelty assessment, severe-issue detection,and a small number of high-value comments. During the verdict window it focuses on eligibility checks, citation assembly, structured reasoning synthesis, and calibrated final score generation.

![alt text](<mermaid-diagram (4).png>)

The agent continuously reads paper and discussion state from the Koala platform, invokes the GSR evidence engine to ground-check claims, passes evidence signals into the meta-review decision layer, and finally emits concrete actions (comments or verdicts) back to the platform.In the revised architecture, the decision layer also receives literature-aware novelty signals from the Lacuna API and severe-issue signals from a dedicated fatal-flaw scan. raw inputs, intermediate evidence artifacts, and decision traces are persisted into a shared data store for reproducibility and later dataset construction.


The official Koala API requires every comment and verdict to include a github_file_url pointing to an audit file in the public transparency repository. Therefore, the agent will generate local transparency logs for every external action and include the corresponding GitHub file URL in all comment and verdict payloads.

## 4. Integration with the Existing GSR System

This plan is designed to maximize reuse of the existing GSR system while adding the missing capabilities needed for competition-level performance. The current GSR components for claim extraction, paper evidence retrieval, text/table/figure evidence chaining, and claim verification can be directly reused by the Evidence Checker. In practice, Koala discussion comments are treated as a new type of review-like input: the system extracts verifiable claims from those comments and then passes them into GSR for evidence retrieval and verification against the paper.

On the paper side, we will continue to reuse GSR’s PDF parsing, section extraction, evidence object construction, figure/table indexing, and hybrid retrieval capabilities. This is especially valuable because GSR already supports structured evidence objects for text, tables, and figures, as well as layout-aware retrieval and verification. That gives us a strong foundation for producing evidence-grounded comments rather than generic LLM-generated opinions.

The new work is primarily in the orchestration and strategy layers around GSR. Specifically, we need to add Koala platform integration, a discussion message data model, comment generation logic, verdict calibration, karma-aware paper selection strategy,novelty search via the **Lacuna API**, severe-issue scanning, base-model numerical scoring, and trajectory logging. In other words, this is not a rewrite of GSR. Instead, it is a competition-specific extension that places a new agent orchestration and decision layer on top of an already strong evidence verification engine.

The key design principle is that **GSR remains the grounding engine**, while the new competition-specific modules supply the additional signals needed to improve final verdict quality: literature-aware novelty estimation, explicit fatal-flaw detection, and calibrated score production.

## 5. Runtime Flow and Alignment with the Competition Phases

According to the competition rules, each paper has a 72-hour lifecycle: the first 48 hours are the Review phase, and the final 24 hours are the Verdict phase. An agent must first participate in the discussion before it is allowed to submit a verdict. Therefore, the system must explicitly adapt its behavior to these two stages.

At runtime, a scheduler periodically retrieves newly opened papers, current paper states, and updated discussion comments. For papers that are considered worth monitoring, the system downloads the PDF, builds a local index, and continuously syncs new comments. This allows the agent to maintain a live internal state for each active paper.

During the **0–48 hour Review phase**, the Opportunity Manager first determines whether the paper is worth participating in, so that the agent avoids wasting karma on low-value or low-probability opportunities. The Evidence Checker monitors the discussion for concrete, verifiable claims and only posts when the signal is strong, the claim is actually checkable, and the resulting comment is likely to add meaningful information. In parallel, the Novelty Checker runs when the paper or discussion suggests that novelty is central to the decision, or when the discussion appears to be overestimating or disputing novelty. It retrieves related work through the Lacuna API and produces a novelty signal that can be surfaced either as a direct comment or as an internal signal for later verdict calibration. The Severe Issue Scanner also runs as a dedicated stage, explicitly checking for fatal flaws that ordinary review discussion may miss. During the same phase, the Decision Calibrator continuously maintains an internal view of the paper’s strengths, weaknesses,novelty, current discussion consensus, and the severity of unresolved concerns.

During the **48–72 hour Verdict phase**, the Opportunity Manager first performs hard eligibility checks: whether the agent has already participated in the discussion, whether there are at least 5 distinct other agents whose comments can be cited, and whether the paper is currently inside the correct verdict window. If these conditions are not met, the agent should skip the verdict. If the conditions are satisfied, the Decision Calibrator combines the paper’s overall quality, the external discussion, the most useful comments, the Evidence Checker’s verification results, the Novelty Checker’s related-work signal, and the Severe Issue Scanner’s fatal-flaw findingsto produce a structured reasoning state.That structured reasoning is then passed to a **base model** to emit the final calibrated 0–10 score.The final verdict cites at least 5 external agents in compliance with the rules.

This two-phase runtime design ensures that the agent is not merely reacting to comments, but is actively building an internal decision state throughout the paper lifecycle and then converting that state into a calibrated final verdict when the competition rules allow it.

![alt text](<mermaid-diagram (5).png>)

## 6. Winning-Oriented Strategy

The central strategy of this system is not to identify the maximum number of flaws. Instead, it is to identify the **highest-value issues** and correctly estimate how much those issues should affect the final ICML decision. Many papers have weaknesses, but weaknesses do not necessarily imply rejection. Likewise, missing code, missing variance reporting, or incomplete ablations are not always fatal flaws. Therefore, the Decision Calibrator must avoid being overly harsh and should instead classify issues into categories such as minor issue, moderate weakness, major concern, and decision-critical issue, and then map those signals to an ICML-style acceptance likelihood.

The revised strategy has three pillars.

First, the agent should produce **high-value, verdict-ready comments** during the discussion phase. These are comments that other agents can directly cite in their own verdicts because they are short, specific, grounded, and decision-relevant. The best comments are not generic opinions, but concrete contributions such as evidence-backed corrections, novelty calibration, or identification of a severe overlooked issue.

Second, the agent should explicitly exploit areas where other agents are likely to be systematically weak. In particular, the system assumes that many agents may **overestimate novelty** and may **fail to search for severe issues unless prompted separately**. The Novelty Checker and Severe Issue Scanner are therefore not optional extras but competitive advantages. The Novelty Checker helps correct inflated claims of novelty by grounding the discussion in related work retrieved via the Lacuna API. The Severe Issue Scanner improves recall on fatal flaws that are disproportionately important for final accept/reject outcomes.

Third, the final verdict should be **decision-calibrated rather than rhetoric-calibrated**. This means the final 0–10 score should not come directly from an instruct-tuned model’s free-form opinion. Instead, the system should first produce structured reasoning over strengths, weaknesses, novelty, severe issues, and consensus, and then use a base model to map that reasoning into a numerical score. The goal is to improve score calibration and better approximate the true ICML decision boundary.

Paper selection is also critical because of the karma system. The first comment on a paper costs 1 karma, and subsequent comments on the same paper cost 0.1 karma, so the agent should not participate indiscriminately. It should prioritize papers with moderate participation—papers that are not so cold that they may fail to reach the 5-agent verdict threshold, but not so crowded that competition for citations becomes too intense. The ideal targets are papers with low-to-medium participation but sufficient discussion potential, where a single strong, evidence-grounded or novelty-calibrating comment can still stand out and be cited later.

## 7. Data Collection and Dataset Construction

A major secondary goal of this project is to systematically collect data during the competition in order to build a high-value AI review interaction dataset. We do not want to collect only extracted claims. Instead, we want to capture the full interaction structure around each paper: the paper itself, the comments, the threads, the verdicts, our own agent’s internal trajectory, and the additional signals introduced by the revised system.

At the paper level, we will store metadata, the PDF, parsed sections, figure/table indices, and any code repository links or supplementary metadata that become available. We will also store the **paper-level novelty artifacts** used by the Novelty Checker, including extracted contribution summaries, generated literature search queries, raw Lacuna API responses, selected related works, and the final novelty assessment. This is important because novelty judgment is one of the hardest and most contest-relevant dimensions, and preserving these intermediate artifacts enables later analysis of where novelty estimation succeeds or fails.

At the comment level, we will store all visible comments as raw text, along with the author/agent identity, timestamps, thread/reply structure, explicit citation relationships where available, and whether a comment was later cited by other comments or by verdicts. We will also store whether a comment was tagged internally as an evidence-check comment, a novelty-correction comment, a severe-issue comment, or a general synthesis comment.

At the claim level, the Evidence Checker will extract verifiable claims from discussion comments and store the resulting structured data: the claim text, the source comment, the claim type, the retrieved evidence, the verification outcome, and the confidence score. This is directly aligned with the existing GSR data model and can be extended naturally from our current claim extraction and verification pipeline.

At the severe-issue level, we will store the outputs of the Severe Issue Scanner, including which issue templates were triggered, which evidence supported the concern, the estimated severity, and whether the issue was later surfaced in a comment or used only internally for verdict calibration. This allows later study of which severe-issue patterns are most predictive of actual decisions.

At the trajectory level, we will store the complete behavior of our own agent: which papers and comments it saw, why it chose to participate or skip, why it did or did not post a comment, draft vs. final comments, draft vs. final verdicts, which external comments it selected for citation, the structured reasoning passed into the score calibration stage, the final numerical score emitted by the base model,and any API errors or retries. This is useful not only for later research and analysis, but also because the competition notes that top agents may need to submit their full trajectory logs.



