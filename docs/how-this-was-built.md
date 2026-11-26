# How this was built

Author: Abdelrhman Shalby.

I set the requirements, chose the architecture and approved each design trade-off. AI tools did most of the implementation and much of the review: design review, implementation, adversarial testing and code review. The work followed a written design and implementation plan, both reviewed by an independent model before approval.

The work was split into tasks. An AI coding agent implemented each task under a written contract, and a lead AI agent reviewed each diff and made the commits; the implementer never committed its own work. The individual contracts and review transcripts are not part of this repository.

The adversarial suite was written in a separate run that had not written the implementation. Its [scripted attacks](../tests/adversarial/test_adversarial_model.py) exercise the executor through the agent loop. Case 14 was planned to check what happens when a model claims a refund is done while it awaits review; the first implementation passed that false sentence through to the customer. The fix added the refund system's own status to replies in code ([agent](../app/agent.py)), and the test still contains the misleading model sentence.

Live evaluations were recorded, failures included, in [fixtures](../evals/fixtures), the [first report](../reports/eval-2026-11-19.md) and the [second report](../reports/eval-2026-11-19-2.md). Between the two runs, the grading was split into money safety and task completion ([graders](../evals/graders.py)), so that a safe refusal no longer counts as a failure. The README lists every change made between the runs. The original evidence remains available, and [replay regression tests](../tests/test_replay_regression.py) check the recorded outcomes.

The [README](../README.md) separates offline tests, real-backend checks, live-model evaluations and simulated demo behaviour, and lists the project's known limits. Model assistance and passing tests do not extend the scope of those measurements.
