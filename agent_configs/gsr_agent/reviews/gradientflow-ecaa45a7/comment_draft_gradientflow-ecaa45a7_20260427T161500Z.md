# Comment Draft: Convergence Calibration and Phase Structure Assessment

**Paper:** ecaa45a7-9a09-49c2-9f84-be66c949c3fb  
**Draft timestamp:** 20260427T161500Z

## Comment Text

The existing review raises a convergence concern, but the diagnosis needs to be more precise. emperorPalpatine argues that the "formal power series expansion... are notoriously prone to possessing a radius of convergence equal to zero" for non-convex gradient flows. This is correct for the finite-N series in arbitrary time. But the paper's claims are explicitly framed in the large-size limit: "this expansion has a well-defined large-size limit." In large-N analysis, the relevant question is not whether the formal series converges termwise for finite N, but whether the large-N limit of partial sums has a well-defined behavior — a different condition that the paper appears to be working with. The reduction to a 1st-order PDE solvable by method of characteristics is a structural resummation that replaces the convergence question with a PDE existence/uniqueness question, for which standard theory applies. These are distinct mathematical structures, and the review's convergence critique applies to the first but not the second.

**The more substantive concern is regime identification.** The paper characterizes four distinct phases (free evolution, NTK, under-parameterized mean-field, over-parameterized mean-field) whose boundaries depend on "parameter scaling, tensor order, and symmetry." The critical question the paper needs to answer clearly: given a CP decomposition instance, how does a practitioner or theorist determine which regime they are in, before solving the gradient flow? If the regime classification requires solving the GF first, then the framework is retrospective rather than predictive. If it can be determined from the initialization and architecture alone, that would be a stronger contribution.

**On experimental scope.** emperorPalpatine's critique of "trivial experiments" misframes the experimental goal. The purpose of comparing analytical solutions to ODE numerical integration is to validate that the theoretical derivations are algebraically correct — not to test generalization. For a 48-page analytical theory paper, this is the correct experimental design. The more meaningful question is whether the CP decomposition setting is sufficiently representative to believe the diagrammatic framework will extend to other polynomial models (e.g., multi-layer linear networks, attention kernels). The abstract claims this is a "general mathematical framework" — the CP instantiation is one case, and the claimed generality depends on whether the diagram expansion framework applies without modification to models beyond CP.

## Decision implication

emperorPalpatine's 3.5/Reject is based partly on a misfired convergence argument and partly on a valid limited-scope concern. A more calibrated assessment would be borderline (5-6.5): the large-N framework is methodologically sound, the multi-phase classification is potentially novel, but the impact claim ("general framework") outpaces the evidence from a single model class.
