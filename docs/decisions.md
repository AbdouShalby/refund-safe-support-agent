# Engineering decisions

## Hand-written agent loop

**Context:** The refund flow needs explicit ordering around persistence, locks, budget and uncertain writes.

**Decision:** Use a hand-written loop over the official Anthropic SDK instead of a framework or SDK tool runner. See [agent loop](../app/agent.py) and [provider adapter](../app/model.py).

**Consequence:** Control flow stays inspectable, but the project owns tool-result repair, stop conditions and provider-failure handling. [Loop tests](../tests/test_agent_loop.py) exercise these paths.

## Money decisions in code

**Context:** Prompt instructions cannot authorize a money movement or validate hostile tool input.

**Decision:** Validate in the executor and use a pure policy function for `auto`, `needs_approval` and `reject`. DOPS independently enforces its refund invariants. See [tools](../app/tools.py), [policy](../app/policy.py) and [mock rules](../app/dops/mock.py).

**Consequence:** Model wording cannot override the policy, although it can still misdescribe a result. The code-derived status footer addresses that visible failure without pretending to correct the model's prose. See [adversarial tests](../tests/adversarial/test_adversarial_model.py) and [policy properties](../tests/test_policy_properties.py).

## Cumulative limits, velocity and a customer lock

**Context:** A per-request ceiling alone allows split requests; an order-only lock cannot serialize customer-wide velocity checks.

**Decision:** Combine cumulative reserved amounts, repeat-order review and customer-wide recent automatic-refund count and amount. Hold the customer lock across fact collection, policy and the DOPS write. See [policy](../app/policy.py), [executor and lock scopes](../app/tools.py).

**Consequence:** Parallel conversations share a decision boundary within the configured store. HTTP scope is customer-wide; mock scope also includes the sandbox. This is scoped concurrency evidence, not proof about arbitrary distributed deployments. See [parallel attacks](../tests/adversarial/test_adversarial_model.py) and [lock tests](../tests/test_tools.py).

## Intent-derived idempotency keys

**Context:** A model may repeat a request under a new tool-use ID or retry after an unknown write outcome.

**Decision:** Derive the key from conversation, order and normalized amount, persist the intent before submission, and never use `tool_use` IDs as keys. See [intents](../app/intents.py).
One retry can make up to five DOPS calls under the 30-second lease; network latency matters only in HTTP mode, which the deployed web app does not use.

**Consequence:** Repeating the same intent reuses the same refund. A new conversation creates a distinct intent but still runs customer-wide policy checks. An uncertain write keeps its key and its attempts. It is resolved from the refund rows DOPS returns, matched by the attempt's trace id, when the conversation resumes or Ops opens the queue. A new attempt re-runs the policy on fresh facts instead of reusing the old decision, and a conflict replays the other mode, so one key never creates two refunds. There is no background job. See [intent tests](../tests/test_intents.py) and [executor replay tests](../tests/test_tools.py).

## Approvals belong to DOPS

**Context:** A model-written claim of manager approval has no authority.

**Decision:** Keep approval and rejection operations in DOPS; expose only order lookup, refund request and escalation to the model. See [tool schemas](../app/prompt.py), [DOPS interface](../app/dops/api.py) and [Ops routes](../app/web.py).

**Consequence:** Escalation opens a handoff and cannot approve a refund. The demo role switch simulates Ops access; it is not a security control or real login. See [role and approval journey tests](../tests/test_web.py) and [adversarial approval spoofing](../tests/adversarial/test_adversarial_model.py).

## Time-free rendering

**Context:** Wall-clock timestamps and incidental identifiers make recorded model requests drift between backends or runs.

**Decision:** Use canonical rendering with whole-day order ages and freeze the system context per conversation. See [renderer](../app/render.py), [prompt](../app/prompt.py) and [request hashing](../app/model.py).

**Consequence:** Replay detects changed requests rather than silently accepting them; dates still matter to policy through an injected clock. See [renderer tests](../tests/test_render.py) and [replay tests](../tests/test_replay.py).

## Mock demo and real-DOPS evaluations

**Context:** A hosted portfolio demo needs isolated synthetic state; the integration also needs evidence against the separate Laravel/MySQL backend.

**Decision:** Run the web demo against the in-process mock and use HTTP DOPS explicitly in test and evaluation harnesses. Maintain a shared contract-parity suite. See [mock](../app/dops/mock.py), [HTTP client](../app/dops/http.py), [parity suite](../tests/parity/test_contract_parity.py) and [eval harness](../evals/harness.py).

**Consequence:** The demo does not establish real-backend behavior by itself. Parity coverage reduces drift but does not cover every backend behavior. Evidence boundaries and historical results are listed in the [README](../README.md#proof-scope).

## Replay first and reserve before spending

**Context:** Open-ended model access has a variable cost and can fail after a provider request has been sent.

**Decision:** Offer [recorded conversations](../demo/recordings) first. Reserve a worst-case cost atomically before each live model call, settle against returned usage and retain the reservation on an unknown outcome. See [budget](../app/budget.py), [pricing](../app/pricing.py), [loop](../app/agent.py) and [visitor limits](../app/limits.py).

**Consequence:** Recordings remain inspectable without model spending. Conservative reservations can deny calls even when actual billed usage was lower. See [budget tests](../tests/test_budget.py) and [web budget tests](../tests/test_web.py).

## Separate safety and completion scores

**Context:** A safe refusal and an incomplete task have different implications; provider failures also distort completion measurements.

**Decision:** Report money safety and task completion separately, retain infrastructure errors and preserve earlier reports. See [graders](../evals/graders.py), [report writer](../evals/report.py) and [grading tests](../tests/test_graders.py).

**Consequence:** A safe result does not imply the requested task was completed. The [second live report](../reports/eval-2026-11-19-2.md) includes a consistently incomplete split-refund scenario, while the [first report](../reports/eval-2026-11-19.md) retains the original grading behavior. The README lists every change made between the two runs.
