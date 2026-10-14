# AGENTS.md

This repository is an AIRAS research project. Its record is
`.research/record.json`, written only through the airas MCP tools
(`preregister_record`, `update_record`, `append_to_record`). Never edit it
by hand.

## Every claim names its verifier

A claim is one falsifiable sentence plus the verifier that will judge it.
The record rejects a claim without `verifier.kind`. Choose one:

| `verifier.kind` | use for | what it checks |
|---|---|---|
| `seyval` | machine-learning experiments run on the platform | metrics under `.research/results/<run_id>/` match the platform's stored run outputs |
| `lean` | mathematical statements | the theorem compiles under the declared toolchain with only the allowed axioms |
| `llm_judge` | claims neither of the above can judge | a dated model scores the evidence against a rubric file in this repository |

A claim no verifier can judge is not a claim: rephrase it until one can,
or drop it.

## Order

Claims, criteria and predicted intervals are committed before any pilot or
full run, on top of the experiment code. Results come back through
`import_run_outputs` with provenance. Nothing in the record is rewritten to
fit data later; a refuted claim is a negative result.

The step-by-step flow is in the `airas` plugin skills (`auto-research`
first). `get_prompts(step="hypothesis_and_design")` returns the claim shape.
