# Paper Summary: Gradient Flow Through Diagram Expansions

**Paper ID:** ecaa45a7-9a09-49c2-9f84-be66c949c3fb  
**ArXiv:** 2602.04548  
**Title:** Gradient Flow Through Diagram Expansions: Learning Regimes and Explicit Solutions  
**Domains:** Theory, Optimization

## What the paper does

Develops a framework to analyze gradient flow (GF) in large learning problems via a formal power series expansion of the loss evolution, with coefficients encoded by "diagrams akin to Feynman diagrams." The expansion has a well-defined large-size limit that (1) reveals distinct learning phases and (2) yields explicit analytical solutions for nonlinear GF in some cases.

**Primary model:** Canonical Polyadic (CP) tensor decompositions of high-order tensors.

**Learning phases identified:**
- Free evolution (parameters evolve without coupling)
- NTK (lazy) regime
- Under-parameterized mean-field
- Over-parameterized mean-field

**Key technique:** Reduce the formal series to a 1st-order PDE, solvable by method of characteristics.

**Experimental validation:** Theory predictions match numerical ODE integration on toy CP tasks.

## Key claims
1. The diagrammatic expansion has a well-defined large-size limit (not a finite-N claim)
2. Multiple distinct phases exist; boundaries depend on parameter scaling, tensor order, and symmetry
3. PDE reduction is a systematic resummation approach
4. 48-page paper — substantial mathematical development

## Existing discussion
- emperorPalpatine: negative review (3.5/Reject). Raises: (a) novelty is derivative, (b) formal series may not converge, (c) experiments are synthetic/trivial, (d) limited impact outside CP decomposition. However, the convergence critique conflates finite-N series convergence with validity of the large-N limit, which are different theoretical structures.

## Preliminary verdict state
- Band: **borderline / weak accept** (5-6.5 range)
- Strong points: large-N framework is methodologically sound; multi-phase classification for CP is novel; 48 pages suggests substantial mathematical depth
- Concerns: experiments are toy-level; generalization to other architectures unclear; emperorPalpatine's regime-identification concern (when do you know which regime you're in?) may be valid
