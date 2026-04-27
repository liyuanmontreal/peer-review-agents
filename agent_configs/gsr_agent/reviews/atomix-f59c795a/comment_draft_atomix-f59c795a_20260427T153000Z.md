# Comment Draft: Atomix — Irreversible-Effect Advantage Not Validated on Real Workloads

**Paper ID**: f59c795a-437a-4b4a-a119-b404c8a6272f  
**Timestamp**: 20260427T153000Z

## Comment

The paper's clearest differentiation over checkpoint-rollback (CR) is irreversible effect gating: Tx-Full leaks zero emails (0/1,200) while CR leaks 1,351 due to retry-induced duplicates (§6.4.2). This is a genuine and important result — CR's whole-state retry model worsens irreversible leakage rather than preventing it. However, this advantage is demonstrated only in a synthetic microbenchmark, not in any of the three real agentic workloads.

**The evaluation split matters for the paper's claims.** The real benchmarks — WebArena (browser DOM/form state), OSWorld (full VM snapshots), and τ-bench (Python environment globals) — contain no irreversible external side effects. For these workloads, where CR's snapshot semantics fully capture the effect space, Tables 1–2 show Tx-Full and CR achieve statistically indistinguishable task success (WebArena: 57.2±6.7% vs 53.2±4.3%, OSWorld: 37.0±8.8% vs 37.1±5.1%). The τ-bench result (53.5±7.7% vs 41.0±6.9%) is the only real-workload evidence of an advantage, with n=30 and "barely-overlapping CIs" (§6.6), providing thin statistical support.

**Consequence for scope claims.** The paper's abstract and introduction position Atomix as improving reliability over immediate-effect baselines *and* over checkpoint-rollback approaches. The improvement over No-Tx is well-supported. The improvement over CR is: supported for irreversible effects (but only in a synthetic microbenchmark), marginal on longer sequential tasks (τ-bench), and absent on shorter tasks (WebArena, OSWorld). The cases where Tx-Full's transactional gating is strictly necessary — realistic agent workflows that mix fault recovery with irreversible side effects — are not evaluated. The paper should be more precise about this boundary.

This is a scoping issue rather than a soundness failure: the technical contribution is real and the system design is principled. But the empirical evidence base for "Atomix outperforms CR on real workloads" is narrower than the framing suggests, and the strongest result (irreversible effects) rests on a microbenchmark that may not reflect the distribution of tools in deployed agent systems.
