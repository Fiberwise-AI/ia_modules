## The Enterprise Agent Reliability Framework (EARF)

**Version:** 1.0
**Status:** Gold Standard
**Purpose:** This document defines the gold standard for creating reliable, observable, secure, and cost-effective AI agents. It provides a holistic framework encompassing design patterns, metrics, operational modes, and organizational practices necessary for enterprise-grade agentic systems.

### **Part I: The Three Pillars of Agent Reliability**

The reliability of any agent system rests on three foundational pillars. These are not optional; they are the minimum requirement for deploying an agent into a production environment.

#### **1. Total Observability**

**Definition:** The ability to ask arbitrary, complex questions about an agent's internal state and decision-making process, both in real-time and post-hoc, without needing to ship new code.

**Key Requirements:**
*   **Decision Trail:** A complete, immutable log of every step an agent takes. It must contain:
    *   **`StepRecord`**: Captures the agent's reasoning, goal, and chosen action.
    *   **`Evidence`**: All data points used for the decision, each tagged with a confidence level (see Part IV).
    *   **`ToolCall`**: Every external tool execution, logged with exact parameters, results, latency, and status.
    *   **`StateDelta`**: The precise "before and after" of the agent's internal state for each step.
*   **Semantic Tracing:** Every workflow must have a unique `TraceID` that is propagated through all agent steps, tool calls, and API interactions. Each step is a `Span` containing rich, queryable metadata (`agent_mode`, `token_cost`, `confidence_score`).
*   **Target Metric (SLO): Mean Time to Explain (MTTE) ≤ 5 minutes.** Any decision, no matter how obscure, must be fully reconstructable and explainable by an engineer within five minutes.

#### **2. Absolute Reproducibility**

**Definition:** The ability to perfectly replay any agent workflow to debug failures, validate changes, and prevent regressions.

**Key Requirements:**
*   **Replayer Engine:** A service capable of replaying a `DecisionTrail` in multiple modes:
    *   **`STRICT`**: Exact replay using captured tool results. This is for validating the agent's logic.
    *   **`SIMULATED`**: Replay with mocked tool calls. This is for fast, isolated unit testing.
    *   **`LIVE`**: Re-execute the workflow against live systems. This is for validating behavior after a system change.
*   **Environment Snapshotting:** For every workflow, capture a snapshot of its execution environment, including library versions (e.g., `poetry.lock`) and API versions. Replays must occur in a containerized environment matching this snapshot to guarantee fidelity.
*   **Golden Datasets:** Maintain a curated collection of `DecisionTrails` representing critical success cases, important failures, and security edge cases. This dataset serves as the core regression suite for the agent.
*   **Target Metric (SLO): Replay Success Rate (RSR) ≥ 99.9%.** The system must be able to successfully replay at least 99.9% of all historical workflows in `STRICT` mode.

#### **3. Formal Safety & Verification**

**Definition:** Proactive and provable guarantees that an agent cannot take destructive or unintended actions, enforced programmatically.

**Key Requirements:**
*   **Agent State Machine:** Formally define the agent's possible operational modes (see Part II) and the legal transitions between them. An agent in `explore` mode, for example, is formally prohibited from transitioning to a state that executes a write operation.
*   **Invariant-Based Guardrails:** A set of rules (invariants) that must hold true before any step is executed. These are non-negotiable safety checks.
    *   *Example Invariant 1:* "A tool call that modifies a production resource must be preceded by a `verified` evidence item of type `human_approval` in the decision trail."
    *   *Example Invariant 2:* "The cumulative token cost of a workflow cannot exceed the predefined budget in its context."
*   **ModeEnforcer Engine:** A critical runtime component that validates every proposed agent step against the State Machine and Invariant Guardrails. Any violation must trigger an immediate halt and escalation.

---

### **Part II: Agent Operational Modes**

To manage risk and ensure predictability, agents must operate in one of three explicit, enforced modes. The agent must declare its mode for each step, and the `ModeEnforcer` validates its actions against the declared mode.

1.  **Explore Mode (Read-Only)**
    *   **Purpose:** Information gathering, analysis, and planning without causing side effects.
    *   **Allowed Actions:** Calling read-only tools (e.g., search APIs, database reads), analyzing data, forming plans.
    *   **Forbidden Actions:** Modifying state, calling write APIs, executing code.
    *   **Core Principle:** The agent is building its `Evidence` base and must explicitly communicate its level of uncertainty.

2.  **Execute Mode (Write-Enabled)**
    *   **Purpose:** Taking concrete actions based on a high-confidence, validated plan.
    *   **Allowed Actions:** Modifying state, calling write APIs, deploying resources.
    *   **Prerequisites:** The `DecisionTrail` must contain sufficient `verified` or `claimed` evidence to justify the action. The proposed action must not violate any safety invariants.
    *   **Core Principle:** Actions should be idempotent and reversible where possible.

3.  **Escalate Mode (Human-in-the-Loop)**
    *   **Purpose:** Pausing execution to request human intervention for uncertain, high-stakes, or ambiguous decisions.
    *   **Process:**
        1.  Halt the workflow.
        2.  Present a structured "Request for Assistance" to a human operator, including: the goal, the ambiguity, options considered, and a recommended path.
        3.  Wait for explicit approval, correction, or denial.
        4.  Log the human interaction as a `verified` piece of `Evidence` and resume.
    *   **Core Principle:** The agent demonstrates it knows what it doesn't know.

---

### **Part III: The Hierarchy of Evidence**

Not all information is equal. To build trustworthy agents, every piece of data (`Evidence`) used in a decision must be categorized by its source and assigned a confidence level.

1.  **Verified (Highest Confidence)**
    *   **Source:** Direct output from deterministic external systems, or direct human input.
    *   **Examples:** API responses, database query results, file contents, explicit human approvals.
    *   **Trustworthiness:** Can be independently verified; considered a "fact" for the agent's reasoning.

2.  **Claimed (Medium Confidence)**
    *   **Source:** The agent's own LLM-generated outputs.
    *   **Examples:** The agent's reasoning, plans, analyses, summaries of other evidence.
    *   **Trustworthiness:** Requires validation. This is a claim made by the agent, not a fact.

3.  **Inferred (Lowest Confidence)**
    *   **Source:** A conclusion derived by the agent from a combination of other evidence.
    *   **Examples:** Predictions, assumptions, conclusions about causality.
    *   **Trustworthiness:** Highly suspect. Inferred evidence should primarily be used to guide the `Explore` mode, not to justify actions in `Execute` mode.

---

### **Part IV: The Ten Core Reliability & Performance Metrics**

These metrics provide a comprehensive dashboard of agent health, reliability, efficiency, and autonomy. They are the SLIs (Service Level Indicators) that feed our SLOs.

#### **Reliability & Stability**
1.  **Step Validity Rate (SVR):** `(successful_steps / total_steps)` — Measures fundamental step-level reliability. **Target: >99.5%**
2.  **Compensation Rate (CR):** `(compensated_steps / total_steps)` — Tracks how often agents must self-correct. High CR indicates flawed planning. **Target: <5%**
3.  **Plan Churn (PC):** `(total_retries / total_workflows)` — Measures planning stability. High PC indicates the agent is thrashing. **Target: <1.5**

#### **Autonomy & Trust**
4.  **Human Intervention Rate (HIR):** `(workflows_requiring_human / total_workflows)` — Measures true autonomy. **Target: <5% (varies by use case)**
5.  **Mode Adherence (MA):** `(1 - (mode_violations / total_steps))` — Measures predictability and conformance to safety rules. **Target: >99.9%**
6.  **Explanation Quality Score (EQS):** `(Average user rating of decision explanations on a 1-5 scale)` — Measures the quality of XAI and user trust. **Target: >4.5**

#### **Financial & Performance (FinOps)**
7.  **Tokens Per Workflow (TPW):** `(total_llm_tokens / total_workflows)` — Tracks the raw LLM cost of a workflow.
8.  **Cost Per Successful Workflow (CPSW):** `(total_operational_cost / successful_workflows)` — The true bottom-line metric for agent efficiency.
9.  **Tool Call Latency (TCL):** `(Average duration of tool executions)` — Tracks performance bottlenecks.
10. **Workflow Completion Time (WCT):** `(Average end-to-end time for a workflow)` — Measures overall user-facing performance.

---

### **Part V: Organizational Mandates**

This framework is not just a technical specification; it requires a cultural commitment to reliability.

1.  **Agent Red Teaming:** A dedicated, independent team is mandated to perform continuous adversarial testing against all production agents. Their goal is to break safety invariants, induce hallucinations, and bypass guardrails. Findings are treated as critical security vulnerabilities.
2.  **Mandatory Regression Testing:** No change to an agent's core logic, prompts, or tools can be deployed without successfully passing a full `STRICT` replay of the relevant "Golden Datasets." A drop in the Replay Success Rate (RSR) must automatically fail the build.
3.  **Reliability-Gated Deployments:** Agent features are rolled out progressively, gated by the core reliability metrics. A feature cannot proceed to a wider audience until it meets the SVR, HIR, and CPSW targets in a limited deployment.
4.  **Blameless Postmortems:** When an agent fails in production, the focus is on a blameless analysis of the systemic failure, not individual error. Every production failure must result in a new entry into the "Golden Dataset" to prevent regression.


### **Part VI - The Governance & Assurance Layer**

**Definition:** The centralized system that aggregates data from all EARF-compliant agents to provide automated compliance monitoring, risk dashboards, and audit-ready reports. This is the bridge between the technical framework and the business's governance, risk, and compliance (GRC) functions.

**Key Components:**

1.  **Compliance Policy Engine:** A system where auditors can define compliance rules in a structured way (e.g., "Any agent handling PII data must have an HIR of at least 10% for write operations"). This engine continuously scans `DecisionTrails` and metrics to detect violations.
2.  **Automated Evidence Collector:** A service that automatically packages `DecisionTrails`, metric reports, and replay results into immutable, audit-ready evidence packages. When an auditor asks, "Show me proof that agent X is compliant with SOX," this system provides it with one click.
3.  **Risk & Performance Dashboards:** High-level dashboards tailored for non-technical stakeholders (like Chief Risk Officers or internal auditors) that visualize key GRC metrics:
    *   Overall agent autonomy vs. human oversight (HIR trends).
    *   Frequency of safety guardrail activations.
    *   Cost-benefit analysis (CPSW vs. business value).
    *   Alerts on significant deviations from expected behavior.
