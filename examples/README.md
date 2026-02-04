# 🪁 Kite Framework Case Studies & Tutorials

Welcome to the **Kite Examples Codebase**. This collection demonstrates how to use Kite to build production-grade agents for various domains.

Each case study is designed to teach specific framework capabilities.

---

## 📚 Table of Contents

| Case | Domain | Key Concepts | Difficulty |
| :--- | :--- | :--- | :--- |
| **[Case 1: E-commerce](#case-1-e-commerce-support-rag--tools)** | Support | RAG, Semantic Routing, Session Memory | 🟢 Beginner |
| **[Case 2: Data Scientist](#case-2-enterprise-data-scientist-sql--python)** | Analytics | Hybrid Reasoning, Code Sandbox, File I/O | 🟡 Intermediate |
| **[Case 3: Deep Researcher](#case-3-deep-research-assistant-web--synthesis)** | Research | Web Search, Multi-step Synthesis, Citations | 🟡 Intermediate |
| **[Case 4: Complex Workflow](#case-4-complex-iterative-workflow-hitl)** | Process | Human-in-the-Loop, Checkpoints, State | 🔴 Advanced |
| **[Case 5: DevOps Automator](#case-5-devops-automation-safety--shells)** | DevOps | Shell Execution, Guardrails, Circuit Breakers | 🟡 Intermediate |
| **[Case 6: Architectures](#case-6-agent-architectures-react-vs-rewoo-vs-tot)** | Agent Design | ReWOO, Tree-of-Thoughts, ReAct Plan | 🔴 Advanced |

---

## Case 1: E-commerce Support (RAG + Tools)

**File:** `case1_ecommerce_support.py`

### 🎯 Scenario
An automated customer support agent that handles order inquiries, returns (checking policy), and escalations.

### 💡 What You'll Learn
1.  **Semantic Routing**: Automatically route "Where is my order?" to `order_support` and "Return policy?" to `policy_specialist`.
2.  **Tool Creation**: How to use the `@app.tool` decorator.
3.  **Lightweight RAG**: Searching JSON policies without a vector DB.

### 🚀 Usage
```bash
PYTHONPATH=. python3 examples/case1_ecommerce_support.py
```

### 🧠 Code Highlight
```python
# Routing based on user intent
@app.agent(routes=["Where is my order?", "Track package"])
def order_support(context):
    """You are an Order Support Specialist..."""
```
*The `@app.agent` decorator automatically registers the agent with the Semantic Router.*

---

## Case 2: Enterprise Data Scientist (SQL + Python)

**File:** `case2_enterprise_analytics.py`

### 🎯 Scenario
A data analyst that needs to query a database **AND** perform complex calculations/visualizations that SQL cannot handle well (e.g., forecasting, plotting).

### 💡 What You'll Learn
1.  **Hybrid Tool Use**: Giving an agent both `SQLTool` (structured data) and `PythonReplTool` (analysis).
2.  **Safe Code Execution**: Using restricted environments for Python execution.
3.  **File Generation**: Agent creates a `.png` file and returns the path.

### 🚀 Usage
```bash
# Requires pandas & matplotlib
PYTHONPATH=. /path/to/venv/bin/python3 examples/case2_enterprise_analytics.py
```

### 🧠 Code Highlight
```python
# Agent decides when to use SQL vs Python
tools = [
    SQLTool(db_path="analytics.db"),
    PythonReplTool() # "Sandbox" for pandas/matplotlib
]
```

---

## Case 3: Deep Research Assistant (Web + Synthesis)

**File:** `case3_research_assistant.py`

### 🎯 Scenario
A researcher tasked with investigating a broad topic (e.g., "AI Security Trends"). It must search multiple sources, filter noise, and compile a report.

### 💡 What You'll Learn
1.  **Web Search Integration**: Using search tools (Google/DuckDuckGo).
2.  **Content Synthesis**: Processing large amounts of unstructured text.
3.  **Citation Tracking**: Keeping track of where facts came from.

### 🚀 Usage
```bash
PYTHONPATH=. python3 examples/case3_research_assistant.py
```

---

## Case 4: Complex Iterative Workflow (HITL)

**File:** `case4_complex_iterative_workflow.py`

### 🎯 Scenario
A business process that requires *human approval* before proceeding. E.g., Generating a marketing email -> Human Reviews -> Send.

### 💡 What You'll Learn
1.  **Human-in-the-Loop (HITL)**: Pausing execution for user input.
2.  **State Management**: Persisting workflow state across pauses.
3.  **Checkpoints**: Defining critical gates in a workflow.

### 🚀 Usage
```bash
PYTHONPATH=. python3 examples/case4_complex_iterative_workflow.py
```
*Note: Follow the interactive prompts to "Approve" or "Reject" the draft.*

---

## Case 5: DevOps Automation (Safety & Shells)

**File:** `case5_devops_automation.py`

### 🎯 Scenario
A system administrator agent that checks server health and performs deployments. It must be **safe** (no `rm -rf`).

### 💡 What You'll Learn
1.  **ShellTool with Guardrails**: Whitelisting allowed commands (`df`, `uptime`) and blocking dangerous ones.
2.  **Circuit Breakers**: Stopping the agent if commands fail repeatedly.
3.  **System Interaction**: Interacting with the underlying OS.

### 🚀 Usage
```bash
PYTHONPATH=. python3 examples/case5_devops_automation.py
```

### 🧠 Code Highlight
```python
# Guardrails in action
system_tools = ShellTool(
    whitelist=["df", "uptime", "free", "git"], 
    blocklist=["rm", "sudo", "shutdown"]
)
```

---

## Case 6: Agent Architectures (ReAct, ReWOO, ToT)

**File:** `case6_reasoning_architectures.py`

### 🎯 Scenario
Comparing how different "brains" solve the same complex problem: "Design a scalable analytics system".

### 💡 What You'll Learn
1.  **ReAct (Reason+Act)**: The standard interactive loop. Good for dynamic tasks.
2.  **ReWOO (Plan->Execute)**: Generates a full plan first, then executes tools in parallel. **Fastest** for known tasks.
3.  **Tree-of-Thoughts (ToT)**: Explores multiple possibilities (Branch A, Branch B) and backtracks. Best for complex reasoning.

### 🚀 Usage
```bash
PYTHONPATH=. python3 examples/case6_reasoning_architectures.py
```
*Watch the logs to see the different "Thinking" patterns!*

---

## 🛠️ Modifying the Examples

To build your own agent, copy one of these files as a template:

1.  **Copy**: `cp examples/case1_ecommerce_support.py my_agent.py`
2.  **Edit Tools**: Define your own `@app.tool` functions.
3.  **Edit Agent**: Change the `system_prompt` and `routes`.
4.  **Run**: `python3 my_agent.py`
