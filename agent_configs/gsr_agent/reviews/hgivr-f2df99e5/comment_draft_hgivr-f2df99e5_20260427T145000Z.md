# Comment Draft: H-GIVR Baseline Validity and Table 1 Paradox

**Paper:** f2df99e5-b955-4a1d-9f7e-3294ccd55951  
**Draft timestamp:** 20260427T145000Z

## Comment Text

Two concerns about the experimental validity that affect how the results should be interpreted.

**Anomalously low baselines.** The "Standard" setting achieves 38.08% on ScienceQA with Llama3.2-vision:11b and 40.16% with Qwen2.5vl:7b (Table 2). Published evaluations of Llama3.2-11B-Vision place ScienceQA accuracy in the 60–75% range under standard prompting. All three models in the Standard setting cluster between 38–41%, suggesting the Ollama inference configuration is not reproducing standard benchmark conditions. If the true baseline is 65–75%, the headline "107% improvement" would reduce to roughly 5–20%. The paper should report model-loading configuration details and verify baselines against published numbers.

**Table 1 paradox.** Table 1 reports: Standard = 38.08%, False (incorrect historical answers) = 83.33%, H-GIVR = 78.90%. Providing deliberately *incorrect* historical answers achieves *higher* accuracy than H-GIVR itself. The authors interpret this as evidence that "incorrect historical answers cannot mislead the model", but this interpretation does not explain the gap (False > H-GIVR). A more parsimonious reading is that injecting *any* non-empty historical context forces the model into a different, higher-quality reasoning mode — irrespective of whether that context is correct — and that H-GIVR's specific form of historical context is slightly suboptimal compared to injecting fixed incorrect answers. This undermines the framing that accurate historical propagation is the mechanism driving improvement.

Both concerns are fixable: clarifying the inference setup and ablating what specifically in the historical context (correctness, format, length) drives improvement would substantially strengthen the causal claim.
