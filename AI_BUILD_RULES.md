# AI_BUILD_RULES.md

## Portfolio AI Engineering — Master Build & Engineering Contract

### Purpose

These rules apply to all three portfolio-grade AI engineering projects.

AI tools may be used extensively to accelerate implementation. The objective is
not to prove that every line was manually written. The objective is to prove
that the engineer can understand, evaluate, debug, optimize, secure, observe,
scale, deploy, and operate an AI system.

AI builds the implementation quickly.

The engineer owns the engineering decisions.

The final project must demonstrate the ability to answer:

- Why was this architecture chosen?
- What alternatives were considered?
- How was quality measured?
- What failed?
- How was the failure detected?
- How was it debugged?
- What was changed?
- What metric improved?
- What tradeoff was introduced?
- What happens when traffic increases?
- What happens when an external service fails?
- What happens when the model produces an unsupported answer?

---

# 1. PRIMARY OBJECTIVE

Build a working AI system as quickly as reasonably possible without sacrificing
correctness, readability, testability, maintainability, reliability, or
security.

Do not spend unnecessary time manually implementing boilerplate that a capable
AI development tool can safely generate.

Prioritize engineering time toward:

- system design
- architecture decisions
- evaluation
- debugging
- reliability
- latency
- scalability
- cost
- observability
- security
- guardrails
- failure handling
- deployment
- production behavior

The system must be both functional and explainable from an engineering
perspective.

---

# 2. PRESERVE EXISTING WORK

Before creating, replacing, or deleting anything:

1. Inspect the existing repository.
2. Understand the current architecture.
3. Identify completed functionality.
4. Identify incomplete functionality.
5. Identify tests.
6. Identify known limitations.
7. Reuse working components whenever practical.

Do NOT discard working implementation simply because another approach exists.

Do NOT rewrite the entire project unnecessarily.

Replace an existing implementation only when:

- it is incorrect
- it causes measurable quality problems
- it creates a significant reliability problem
- it creates an important performance problem
- it prevents required functionality
- there is a clearly justified architectural reason

When replacing existing work:

- explain why
- identify what changes
- preserve useful tests
- avoid unrelated refactoring

Existing implementation and the engineering knowledge gained from it are
valuable assets.

---

# 3. AI-FIRST DEVELOPMENT

AI development tools may be used aggressively for implementation.

They may generate:

- application scaffolding
- UI
- APIs
- database models
- service layers
- RAG pipelines
- agent workflows
- integrations
- tests
- Docker configuration
- deployment configuration
- documentation
- CI/CD configuration
- monitoring integrations

AI-generated code must still be reviewed, tested, and understood.

Never assume generated code is correct because it runs.

---

# 4. CODE QUALITY

The goal is:

> Minimum clean code required to correctly solve the problem.

Prefer:

- simple code
- readable code
- explicit behavior
- focused functions
- clear names
- minimal dependencies
- straightforward control flow
- maintainable structure

Avoid:

- unnecessary abstractions
- unnecessary classes
- unnecessary helper functions
- excessive design patterns
- duplicated logic
- over-engineering
- premature optimization
- unnecessary frameworks
- unnecessary dependencies

Do not optimize for number of files or lines of code.

Optimize for clarity and correctness.

---

# 5. HUMAN-READABLE CODE

Code should look like code a strong engineer would reasonably maintain.

Prefer direct, understandable implementations over abstractions that obscure
simple behavior.

Use abstractions only when they provide a real benefit such as:

- separation of concerns
- testability
- configurability
- reuse
- provider substitution
- reliability
- maintainability

---

# 6. COMMENTS

Do not over-comment.

Comments should explain only:

- non-obvious reasoning
- important tradeoffs
- unusual constraints
- failure-handling decisions

Do not add comments that merely restate the code.

Keep comments concise and human-readable.

---

# 7. CONFIGURATION AND SECRETS

Never hard-code:

- API keys
- passwords
- secrets
- tokens
- deployment-specific values
- credentials

Use environment variables or appropriate configuration.

Important AI parameters should be configurable where practical, including:

- model
- embedding model
- chunk size
- chunk overlap
- top-k
- reranker
- similarity threshold
- temperature
- maximum tokens
- timeout
- retry count
- rate limits

---

# 8. AI/ML ENGINEERING

AI components must be treated as measurable system components.

Do not assume:

- a larger model is automatically better
- more retrieved documents are automatically better
- smaller chunks are automatically better
- larger chunks are automatically better
- semantic search is automatically better
- BM25 is automatically better
- reranking is automatically necessary
- longer prompts automatically produce better answers

When alternatives exist:

1. Establish a baseline.
2. Define evaluation metrics.
3. Run an experiment.
4. Compare results.
5. Consider latency.
6. Consider cost.
7. Select based on evidence.
8. Document important decisions.

---

# 9. RAG ENGINEERING

For RAG systems, evaluate where practical:

### Ingestion

- extraction quality
- malformed documents
- scanned documents
- tables
- figures
- unusual layouts

### Chunking

- chunk size
- chunk overlap
- sentence boundaries
- section boundaries
- metadata preservation

### Retrieval

- BM25
- vector retrieval
- hybrid retrieval
- top-k
- similarity thresholds

### Reranking

Measure whether reranking actually improves retrieval or answer quality
enough to justify its latency and cost.

### Generation

Evaluate:

- factual correctness
- groundedness
- citation correctness
- citation completeness
- hallucination rate
- answer relevance

Do not declare a retrieval strategy superior based on a few manually
observed examples.

---

# 10. EVALUATION

AI quality must be measured using representative evaluation datasets.

Where appropriate, measure:

- Recall@K
- Precision@K
- Hit@K
- MRR
- answer correctness
- faithfulness
- groundedness
- citation precision
- citation recall
- hallucination rate

System metrics may include:

- p50 latency
- p95 latency
- p99 latency
- throughput
- error rate
- timeout rate
- retry rate
- fallback rate
- token usage
- cost per request

Evaluation must be repeatable.

---

# 11. BASELINES AND EXPERIMENTS

Before optimizing a component:

1. Measure current behavior.
2. Record the baseline.
3. Identify the suspected problem.
4. Make one meaningful change.
5. Run evaluation again.
6. Compare results.
7. Document the tradeoff.

Do not optimize blindly.

Do not make multiple unrelated changes and claim that one change caused the
improvement.

For comparisons, record:

- baseline
- experiment
- metric
- result
- latency impact
- cost impact
- decision

Never fabricate results.

---

# 12. FAILURE-FIRST ENGINEERING

Intentionally test the system under failure.

Consider:

- invalid input
- empty input
- extremely large input
- malformed documents
- bad retrieval
- missing evidence
- hallucinated answers
- LLM timeout
- LLM rate limit
- LLM unavailable
- malformed LLM response
- embedding service failure
- vector database failure
- database failure
- network failure
- tool failure
- authentication failure
- dependency failure

The goal is to make failures:

- detectable
- bounded
- observable
- recoverable where appropriate
- graceful where recovery is impossible

---

# 13. RETRIES

Retries must be intentional.

Retry only errors that are plausibly transient.

Use:

- bounded retries
- exponential backoff where appropriate
- retry limits
- timeouts

Avoid:

- infinite retries
- retry storms
- retrying invalid requests
- retrying permanent failures

---

# 14. FALLBACKS AND GRACEFUL DEGRADATION

External dependencies should have appropriate fallback behavior where
practical.

Examples:

- primary LLM → fallback model
- vector retrieval → alternate retrieval
- unavailable service → graceful response

Fallbacks must be explicit.

Do not introduce fallbacks merely for architectural decoration.

If sufficient evidence is unavailable, a grounded system must not invent
evidence.

---

# 15. GUARDRAILS

Consider guardrails for:

- invalid input
- prompt injection
- malicious retrieved content
- unsupported questions
- missing evidence
- excessive context
- unsafe tool use
- malformed model output
- unauthorized actions
- secret leakage

Agent tool permissions must follow least-privilege principles.

---

# 16. OBSERVABILITY

Important requests should be traceable.

Where practical, capture:

- request ID
- timestamp
- component
- model
- latency
- retrieval statistics
- retrieved document/chunk identifiers
- model response status
- token usage
- errors
- retries
- fallback activation

Do not log:

- API keys
- passwords
- authentication tokens
- unnecessary sensitive information

Observability should allow an engineer to answer:

> What happened to this request?

---

# 17. LATENCY AND PERFORMANCE

Measure latency instead of guessing.

Break important requests into components such as:

Request
→ input processing
→ embedding
→ retrieval
→ reranking
→ prompt construction
→ LLM
→ post-processing
→ response

Measure component latency where practical.

Use:

- p50
- p95
- p99

When latency is too high:

1. Identify the bottleneck.
2. Measure it.
3. Optimize that component.
4. Measure again.

Do not optimize components that are not bottlenecks.

---

# 18. COST ENGINEERING

Track AI-related cost where applicable.

Consider:

- input tokens
- output tokens
- embedding usage
- reranking usage
- model pricing
- request frequency
- infrastructure cost

Useful metrics include:

- cost per request
- tokens per request
- cost per successful answer

Possible optimization strategies include:

- reducing unnecessary context
- reducing unnecessary model calls
- caching
- model routing
- reducing unnecessary reranking
- prompt optimization
- batching where appropriate

Do not reduce cost at the expense of unacceptable quality without measuring
the tradeoff.

---

# 19. SCALABILITY AND TRAFFIC

Document a scaling strategy.

Consider:

- concurrent requests
- horizontal scaling
- load balancing
- stateless services
- connection pooling
- caching
- queues
- rate limiting
- backpressure
- database limits
- vector database limits
- model provider limits
- API quotas
- autoscaling

Do not claim that a local deployment supports production traffic without
measurement.

When discussing higher traffic:

1. Identify the current bottleneck.
2. Estimate capacity.
3. Identify the next bottleneck.
4. Propose an architectural change.
5. Measure where possible.

---

# 20. SECURITY

Consider:

- secret management
- authentication
- authorization
- prompt injection
- tool permissions
- untrusted retrieved content
- input validation
- output validation
- rate limiting
- excessive requests
- secret leakage
- dependency vulnerabilities

Never allow an AI agent to execute sensitive actions without appropriate
authorization boundaries.

---

# 21. TESTING

Tests should cover important behavior.

Include where appropriate:

- unit tests
- integration tests
- API tests
- failure tests
- retrieval tests
- evaluation tests
- regression tests

Do not create tests merely to increase test count.

Tests should detect real regressions.

Always run relevant tests after meaningful changes.

---

# 22. REGRESSION PROTECTION

When changing:

- prompts
- models
- chunking
- embeddings
- retrieval
- reranking
- context construction
- generation parameters

run the relevant evaluation suite again.

A change that improves one metric but damages another must be identified and
documented.

AI systems require regression protection because behavior can change even when
the application code changes only slightly.

---

# 23. DEPENDENCIES

Keep dependencies minimal.

Before introducing a dependency, determine whether it provides a meaningful
benefit.

Avoid adding libraries for functionality that can be implemented simply
without them.

Use reproducible dependency management.

---

# 24. API DESIGN

APIs should:

- validate input
- return predictable responses
- handle errors explicitly
- use appropriate HTTP status codes
- avoid leaking internal exceptions
- support request tracing where appropriate

Do not expose unnecessary internal implementation details.

---

# 25. DOCUMENTATION

Every project must maintain a useful README containing:

- project purpose
- problem statement
- architecture
- data flow
- technology stack
- setup
- environment variables
- local execution
- API usage
- testing
- evaluation
- deployment
- observability
- known failure modes
- limitations
- important architecture decisions

Documentation should help another engineer understand and run the project.

Avoid unnecessary marketing language.

---

# 26. ARCHITECTURE DECISION RECORDS

For important technical choices, document:

- decision
- why it was needed
- alternatives considered
- chosen approach
- reason
- tradeoffs
- evidence

Examples:

- chunking strategy
- embedding model
- vector database
- retrieval strategy
- reranking
- LLM selection
- caching
- fallback
- deployment architecture

Do not document every trivial implementation choice.

---

# 27. GIT WORKFLOW

GitHub is the source of truth.

Development should happen incrementally.

After every meaningful logical milestone:

1. Inspect changes.
2. Run tests.
3. Verify the application.
4. Update documentation if necessary.
5. Commit the change.
6. Push to GitHub.

Prefer small logical commits over one enormous final commit.

Example commits:

- feat: add document ingestion pipeline
- feat: add sentence-aware chunking
- feat: add hybrid retrieval
- feat: add citation generation
- test: add retrieval evaluation dataset
- feat: add request tracing
- fix: handle LLM timeout fallback
- perf: reduce retrieval latency
- docs: document retrieval architecture

Do not mix unrelated changes into one commit.

---

# 28. GIT SAFETY

Before modifying an existing repository:

- inspect the current branch
- inspect current changes
- avoid overwriting user work
- avoid deleting files without justification

Do not reset, revert, force-push, or destroy existing work unless explicitly
requested.

Do not commit secrets.

---

# 29. INCREMENTAL AI DEVELOPMENT

AI builders should work incrementally:

Inspect
→ Plan
→ Implement
→ Test
→ Fix
→ Report

Do not generate an enormous rewrite when a small change is sufficient.

If a task is large, divide it into logical milestones.

The developer must have regular opportunities to create GitHub checkpoints.

---

# 30. MULTIPLE AI TOOLS

Different AI tools may be used for different purposes.

Do not allow multiple AI tools to independently rewrite the same project
without a controlled Git workflow.

Recommended pattern:

Rapid builder
→ initial implementation
→ GitHub checkpoint
→ Codex
→ engineering cleanup / implementation / tests
→ GitHub checkpoint
→ Claude Code
→ independent review / debugging
→ GitHub checkpoint
→ evaluation
→ optimization

GitHub remains the source of truth.

---

# 31. AI TOOL SELECTION

Use the tool best suited to the task.

Rapid application builders may handle:

- initial application generation
- UI
- scaffolding
- basic integrations

Coding agents may handle:

- repository-level implementation
- debugging
- refactoring
- testing
- integration
- code review

Research tools may handle:

- current documentation
- current model information
- current pricing
- technical comparisons
- unfamiliar technologies

Workflow automation tools may handle:

- event-driven workflows
- integrations
- notifications
- multi-step business processes

Do not use a tool simply because it is available.

---

# 32. HUMAN ENGINEERING OWNERSHIP

The developer must understand:

- system architecture
- data flow
- dependencies
- model behavior
- retrieval behavior
- evaluation methodology
- failure modes
- observability
- latency
- cost
- scaling
- security
- deployment

The developer must be able to defend the system technically in an
interview.

AI assistance is acceptable.

Blind dependence on AI is not.

---

# 33. DO NOT HIDE FAILURES

Never:

- silently ignore exceptions
- fabricate successful results
- fake metrics
- invent evaluation results
- claim a benchmark was performed when it was not
- claim production readiness without evidence
- remove failing tests simply because they fail

A failure is useful engineering information.

Report it.

Investigate it.

Fix it or document it.

---

# 34. NO FABRICATED METRICS

Never invent:

- accuracy
- recall
- precision
- latency
- throughput
- cost
- token usage
- benchmark results

If a metric has not been measured, state:

> Not measured yet.

If a value is estimated, label it explicitly as an estimate.

---

# 35. PRODUCTION READINESS

Do not call the system production-ready simply because:

- the application runs locally
- the API responds
- the UI looks good
- tests pass once

Production readiness should consider:

- reliability
- observability
- security
- performance
- scalability
- failure handling
- dependency availability
- deployment
- configuration
- monitoring
- rollback
- evaluation
- cost

The level of production readiness must be proportional to the project's
actual deployment requirements.

---

# 36. EXPERIMENTAL DECISION MAKING

When comparing alternatives, use evidence.

Record:

Baseline
→ Experiment
→ Metric
→ Result
→ Latency impact
→ Cost impact
→ Decision

Example:

Baseline:
Vector retrieval, top-k=5

Experiment:
Hybrid BM25 + vector retrieval

Result:
Record the actual measured result.

Latency:
Record the actual measured result.

Decision:
Select the approach only after considering quality, latency, cost and
operational complexity.

Never replace placeholders with invented values.

---

# 37. SIMPLICITY OVER COMPLEXITY

If two approaches provide similar measurable results, prefer the approach
with:

- fewer dependencies
- fewer services
- simpler deployment
- lower operational burden
- lower cost
- easier debugging
- easier maintenance

Do not add complexity without a measurable or architectural reason.

---

# 38. SECURITY OF AI-GENERATED CODE

AI-generated code must be treated as untrusted until reviewed.

Check for:

- insecure defaults
- exposed credentials
- unsafe shell execution
- SQL injection
- command injection
- unrestricted file access
- unsafe deserialization
- excessive permissions
- insecure API endpoints
- missing authentication
- missing authorization

Never assume generated code is secure.

---

# 39. MISSING INFORMATION

If an implementation decision materially depends on missing information:

DO NOT GUESS.

Ask for the required information.

Examples:

- deployment target
- available GPU
- API provider
- expected traffic
- data format
- authentication requirement
- compliance requirement
- storage requirement

If missing information does not materially affect implementation, choose the
simplest reasonable default and document it.

---

# 40. FINAL REPORT AFTER EACH TASK

After completing a task, return:

### What changed
Short summary.

### Files changed
List files.

### Tests run
List commands/tests.

### Results
State actual results.

### Known limitations
State remaining issues.

### Architecture impact
Explain meaningful architectural changes.

### Git checkpoint
Suggest a logical commit message.

### Next action
State the smallest logical next step.

Keep the response concise.

---

# 41. FINAL PROJECT PRINCIPLE

The objective is NOT:

> "AI wrote the application."

The objective is:

> "AI accelerated implementation while the engineer demonstrated the ability
> to understand, evaluate, debug, optimize, secure, observe, scale and operate
> the resulting AI system."

Build quickly.

Measure honestly.

Break intentionally.

Debug systematically.

Optimize using evidence.

Document important decisions.

Deploy reproducibly.

Keep the system simple.

Never fake engineering evidence.

---

# 42. THREE-PROJECT PORTFOLIO GOAL

The three projects together should demonstrate practical capability across:

## Project 1 — RAG Research Assistant

Demonstrate:

- ingestion
- chunking
- embeddings
- vector retrieval
- BM25
- hybrid retrieval
- reranking
- grounded generation
- citations
- retrieval evaluation
- answer evaluation
- hallucination mitigation
- latency
- token usage
- cost
- observability
- failure handling
- deployment

## Project 2 — Agentic Support / Ticket Resolution System

Demonstrate:

- agent orchestration
- tool calling
- workflow design
- retrieval
- structured outputs
- guardrails
- authorization
- retries
- fallbacks
- human escalation
- tool failures
- latency
- cost
- observability
- evaluation
- production reliability

## Project 3 — MCP Personal Developer Server

Demonstrate:

- MCP architecture
- tool design
- resource access
- permissions
- secure tool execution
- GitHub integration
- project search
- test execution
- logs
- observability
- failure handling
- developer workflow automation

The three projects should complement one another rather than duplicate the
same technology.

---

# 43. OVERALL WORKING MODEL

The project workflow is:

AI builds quickly
→ GitHub
→ inspect
→ understand architecture
→ evaluate
→ intentionally break
→ observe
→ diagnose
→ fix
→ benchmark
→ optimize
→ secure
→ scale
→ deploy
→ document
→ defend in interview

The implementation may be AI-assisted.

The engineering ownership must remain human.

---

# 44. OPERATING PRINCIPLE FOR THIS PORTFOLIO

Do not spend the majority of available time learning how to manually build
every component that AI tools can reliably generate.

Spend the majority of time learning how to determine:

- whether the generated system is correct
- whether it is reliable
- where it fails
- why it fails
- how to detect failures
- how to debug failures
- how to improve quality
- how to reduce latency
- how to reduce cost
- how to handle increased traffic
- how to protect the system
- how to deploy it
- how to monitor it
- how to recover from failures
- how to prove improvements with metrics

The final portfolio should show engineering judgment, not merely code volume.
