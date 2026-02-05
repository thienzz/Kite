# 🎯 Kite Examples - Production Case Studies

**Learn by building real-world AI agents** 

Each example is a complete, production-ready system showcasing different Kite capabilities. Copy, modify, and ship.

---

## 📚 Quick Navigation

| Case | What You'll Build | Tech Highlights | Difficulty |
|:----:|-------------------|----------------|:----------:|
| **[1](#-case-1-e-commerce-support-system)** | Multi-Agent Customer Support | LLM Routing • RAG • Parallel Processing | 🟢 Beginner |
| **[2](#-case-2-enterprise-data-analyst)** | SQL + Python Data Scientist | Code Execution • Hybrid Reasoning • Charts | 🟡 Intermediate |
| **[3](#-case-3-deep-research-assistant)** | Web Research Automation | Multi-Step Planning • CheckpoI nting • Synthesis | 🟡 Intermediate |
| **[4](#-case-4-multi-agent-collaboration)** | Supervisor-Worker Pattern | HITL • Quality Gates • Iterative Refinement | 🔴 Advanced |
| **[5](#-case-5-devops-automation)** | Safe System Administrator | Shell Tools • Whitelisting • Circuit Breakers | 🟡 Intermediate |
| **[6](#-case-6-agent-reasoning-architectures)** | ReAct vs ReWOO vs ToT Comparison | Cognitive Patterns • Performance Analysis | 🔴 Advanced |

**🎓 Recommended Path:** Follow 1→2→3→5→6→4 for progressive learning.

---

## 🟢 Case 1: E-Commerce Support System

**File:** [`case1_ecommerce_support.py`](case1_ecommerce_support.py)

### 🎯 The Challenge

You're running an online store. Customers ask about orders, refunds, and product availability. One generic chatbot can't handle all these well. **You need specialized agents for each domain** - but customers shouldn't have to know which agent to talk to.

### 💡 The Solution

Build a **semantic router** that automatically directs queries to specialized agents:

- 👨‍💼 **OrderSpecialist** - Handles tracking & shipping
- 💰 **RefundSpecialist** - Processes returns & refunds  
- 📦 **ProductSpecialist** - Checks inventory & pricing

```python
# The router intelligently routes without hardcoded rules
query = "Where is my order ORD-001?"
result = await ai.llm_router.route(query)
# Automatically routed to OrderSpecialist ✅
```

### 🔑 Key Concepts

**1. Multiple Specialized Agents**

Instead of one bloated agent, create focused specialists:

```python
order_agent = ai.create_agent(
    name="OrderSpecialist",
    system_prompt="You are an order support specialist...",
    tools=[search_tool],  # Only needs order lookup
    agent_type="react"
)

refund_agent = ai.create_agent(
    name="RefundSpecialist",
    system_prompt="You are a refund specialist...",
    tools=[search_tool, refund_tool, cancel_tool],  # More powerful toolset
    agent_type="react"
)
```

**2. LLM-Based Routing**

The router uses the LLM itself to understand intent:

```python
from kite.routing.llm_router import LLMRouter

ai.llm_router = LLMRouter(llm=ai.llm)

# Register routes with descriptions
ai.llm_router.add_route(
    name="order_support",
    description="Handle order tracking, delivery status, and shipping updates.",
    handler=lambda q, c=None: order_agent.run(q, context=c)
)
```

**3. Parallel Processing**

Process multiple customer queries simultaneously:

```python
queries = [
    "Check order ORD-003",
    "Is the phone available?",
    "Process refund for ORD-001"
]

results = await asyncio.gather(*[ai.llm_router.route(q) for q in queries])
# All 3 run concurrently! ⚡
```

### 🚀 Run It

```bash
PYTHONPATH=. python3 examples/case1_ecommerce_support.py
```

**Expected Output:**
```
[Query 1/5] Where is my order ORD-001?
   Route: order_support
   Confidence: 95.00%
   Response: Your order ORD-001 is Shipped with 1x Laptop ($1200.00)...
   Time: 1.23s
```

### 📖 Learning Outcomes

✅ Build multi-agent systems with role specialization  
✅ Implement semantic routing for automatic intent detection  
✅ Handle concurrent requests for better throughput  
✅ Structure production customer support automation  

---

## 🟡 Case 2: Enterprise Data Analyst

**File:** [`case2_enterprise_analytics.py`](case2_enterprise_analytics.py)

### 🎯 The Challenge

Your business team asks: *"Show me sales by region with a chart."* 

A pure SQL agent can query data but can't create visualizations. A pure Python agent doesn't know your database schema. **You need hybrid reasoning**.

### 💡 The Solution

Build an agent that combines **SQL for data retrieval** + **Python for analysis**:

```python
# Agent decides which tool to use
analyst = ai.create_agent(
    name="DataScientist",
    tools=[SQLTool(db_path), PythonReplTool()],
    system_prompt="""You are a Lead Data Scientist.
    
    Process:
    1. Use 'query_sql' to fetch data from database
    2. Use 'python' to analyze, calculate metrics, and visualize
    3. Output summary + save chart as 'sales_chart.png'
    """
)

result = await analyst.run("Analyze sales performance by Region with a chart.")
```

### 🔑 Key Concepts

**1. Safe Code Execution**

The `PythonReplTool` runs arbitrary Python in a restricted environment:

```python
from kite.tools.code_execution import PythonReplTool

python_tool = PythonReplTool()
# Supports pandas, matplotlib, numpy
# Blocks dangerous operations (file system writes outside cwd, network calls)
```

**2. Hybrid Tool Usage**

The agent chains tools intelligently:

```
User: "Analyze sales by region"
  ↓
Agent: (Reasoning) "I need data first"
  ↓
Tool: query_sql("SELECT region, SUM(amount) FROM sales GROUP BY region")
  ↓
Agent: (Reasoning) "Now I can visualize"
  ↓
Tool: python("import matplotlib; df.plot(kind='bar'); plt.savefig('chart.png')")
  ↓
Response: "Sales analysis complete! North leads with $1,225. Chart saved."
```

**3. Smart Model Selection**

Use powerful models for complex reasoning:

```python
from kite.optimization.resource_router import ResourceAwareRouter

router = ResourceAwareRouter(ai.config)

# Use the "smart" model (e.g., GPT-4) for analyst work
analyst = ai.create_agent(
    model=router.smart_model,  # Not the cheap fast model!
    ...
)
```

### 🚀 Run It

```bash
PYTHONPATH=. python3 examples/case2_enterprise_analytics.py
```

**Expected Output:**
```
📊 CASE 2: ENTERPRISE DATA SCIENTIST AGENT
...
Final Analysis Report
================================================================================
Sales Performance by Region:
- North: $1,225.00 (3 transactions)
- South: $415.00 (2 transactions)
- East: $140.00 (2 transactions)
- West: $2,500.00 (1 transaction)

North region has the highest transaction volume...

[System] Chart generated successfully: sales_chart.png 📈
```

### 📖 Learning Outcomes

✅ Combine SQL and Python for powerful data analysis  
✅ Execute code safely with sandboxed environments  
✅ Generate visual outputs (charts, graphs)  
✅ Build "analyst" agents that reason across multiple tools  

---

## 🟡 Case 3: Deep Research Assistant

**File:** [`case3_research_assistant.py`](case3_research_assistant.py)

### 🎯 The Challenge

You need to research: *"The Impact of Solid State Batteries on EV Industry by 2030"*

This requires:
1. Breaking the topic into sub-questions
2. Searching multiple sources
3. Synthesizing findings into a report

**No single LLM call can do this.** You need a multi-step pipeline with checkpointing.

### 💡 The Solution

Build a **3-agent research system** with state persistence:

1. **Planner** (Smart Model) → Decomposes topic into 4 sub-questions
2. **Researcher** (Fast Model) → Searches web for each question
3. **Analyst** (Smart Model) → Synthesizes final markdown report

```python
# Planning Phase
planner = ai.create_agent(
    model=router.smart_model,  # Need good reasoning
    system_prompt="Break topic into 4 searchable sub-questions..."
)

# Research Phase  
researcher = ai.create_agent(
    model=router.fast_model,  # High volume, cheap model
    tools=[WebSearchTool()],
    system_prompt="Find concrete facts for each question..."
)

# Synthesis Phase
analyst = ai.create_agent(
    model=router.smart_model,  # Need quality writing
    system_prompt="Write professional markdown report from notes..."
)
```

### 🔑 Key Concepts

**1. Multi-Step Workflow with Checkpointing**

Save progress at each phase so you can resume if interrupted:

```python
from kite.persistence import JSONCheckpointer

checkpointer = JSONCheckpointer("research_state.json")

state = checkpointer.load() or {
    "topic": "...",
    "plan": [],
    "research_notes": {},
    "final_report": "",
    "status": "planning"  # → researching → reporting → completed
}

# After each phase:
checkpointer.save(state)
```

**2. Resource-Aware Model Selection**

Don't waste money on expensive models for simple tasks:

```python
# SMART for planning (complex reasoning)
planner = ai.create_agent(model=router.smart_model, ...)

# FAST for web searching (high volume, simple task)
researcher = ai.create_agent(model=router.fast_model, tools=[search], ...)

# SMART for final synthesis (quality writing)
analyst = ai.create_agent(model=router.smart_model, ...)
```

**3. Web Search Integration**

Use real web search tools:

```python
from kite.tools import WebSearchTool

search_tool = WebSearchTool()  # DuckDuckGo scraper

# Agent can now Google things!
researcher = ai.create_agent(tools=[search_tool], ...)
```

### 🚀 Run It

```bash
PYTHONPATH=. python3 examples/case3_research_assistant.py
```

**Expected Output:**
```
[Phase 1] Planning Research Strategy...
   [Planner] Generated 4 sub-questions:
      1. What is the current state of solid-state battery technology?
      2. Which companies are leading solid-state battery development?
      3. What are the main challenges preventing mass production?
      4. When is commercial deployment expected?

[Phase 2] Executing Deep Research...
   👉 Researching Q1: What is the current state...
   [System] Cooling down for 2s...
   ...

[Phase 3] Synthesizing Final Report...

RESEARCH COMPLETED. Report saved to: research_report.md
```

### 📖 Learning Outcomes

✅ Build multi-phase research pipelines  
✅ Use checkpointing for fault-tolerant workflows  
✅ Optimize costs with smart/fast model selection  
✅ Integrate real web search into agents  
✅ Synthesize structured reports from unstructured data  

---

## 🔴 Case 4: Multi-Agent Collaboration

**File:** [`case4_multi_agent_collab.py`](case4_multi_agent_collab.py)

### 🎯 The Challenge

You're building a system that researches a topic, writes a report, but **requires quality approval before finalizing**. If the report is low quality, the writer should revise based on feedback.

This is a classic **supervisor-worker pattern with iterative refinement**.

### 💡 The Solution

Build a 3-agent loop with quality gates:

1. **Researcher** (Fast) → Gathers facts from web
2. **Analyst** (Adaptive) → Writes draft report
3. **Critic** (Smart) → Reviews quality, approves or requests revision

```python
# Supervisor loop
while not state['approved'] and iteration < max_revisions:
    # Research
    facts = await researcher.run("Find info on {topic}")
    
    # Draft
    draft = await analyst.run("Write report", context=facts)
    
    # Review
    review = await critic.run("Review this draft...")
    
    if review['approved']:
        state['approved'] = True
    else:
        state['feedback_history'].append(review['feedback'])
        # Loop again with feedback context
```

### 🔑 Key Concepts

**1. Supervisor Pattern**

One orchestrator coordinates multiple specialist agents:

```python
# Research Agent: Gathers data
researcher = ai.create_agent(
    model=router.fast_model,
    tools=[WebSearchTool()],
    ...
)

# Analyst Agent: Writes drafts
analyst = ai.create_agent(
    system_prompt="Write comprehensive executive summary..."
)

# Critic Agent: Quality control
critic = ai.create_agent(
    system_prompt="""Review for clarity and strategic value.
    Output JSON: {"score": int, "feedback": str, "approved": bool}"""
)
```

**2. Iterative Refinement with Feedback**

Feed previous feedback into next iteration:

```python
context = f"""
Previous Feedback: {state['feedback_history']}
Current Facts: {state['facts']}
"""

draft = await analyst.run("Improve the report", context=context)
```

**3. Output Guardrails**

Validate agent outputs match expected formats:

```python
from kite.safety.guardrails import OutputGuardrail, StandardEvaluation

quality_guard = OutputGuardrail(StandardEvaluation)

# Ensures critic returns valid JSON
validated = quality_guard.validate(critic_response)
```

### 🚀 Run It

```bash
PYTHONPATH=. python3 examples/case4_multi_agent_collab.py
```

**Expected Output:**
```
CASE 4: PRODUCTION-GRADE MULTI-AGENT COLLABORATION

--- Iteration 1/3 ---
[Supervisor] Tasking Researcher (Real Web Search)...
   [Researcher] Data gathered.
[Supervisor] Tasking Analyst to Draft...
   [Analyst] 1,247 chars written.
[Supervisor] Tasking Critic to Review...
   [Critic] Score: 6/10 | Approved: False
   [Feedback] Lacks concrete examples and data points...

--- Iteration 2/3 ---
...
   [Critic] Score: 9/10 | Approved: True

PROJECT COMPLETED SUCCESSFULLY
   Saved to: final_report.md
```

### 📖 Learning Outcomes

✅ Build supervisor-worker multi-agent systems  
✅ Implement iterative refinement loops  
✅ Use output guardrails for validation  
✅ Handle state persistence across iterations  
✅ Create quality gates with approval logic  

---

## 🟡 Case 5: DevOps Automation

**File:** [`case5_devops_automation.py`](case5_devops_automation.py)

### 🎯 The Challenge

You want an AI agent that can check server health and deploy services. But giving an agent full shell access is **extremely dangerous** - imagine it running `rm -rf /` by mistake.

**You need safe, constrained system access.**

### 💡 The Solution

Use `ShellTool` with strict whitelisting:

```python
from kite.tools.system_tools import ShellTool

# Only allow safe commands
shell_tool = ShellTool(allowed_commands=[
    "ls", "pwd", "git", "df", "echo", "uptime", "grep"
])

# Agent can now run ONLY these commands
sysops = ai.create_agent(
    name="SysOps",
    tools=[shell_tool, deploy_tool],
    system_prompt="""You are a Senior DevOps Engineer.
    ALWAYS check system resources before deployment.
    """
)

await sysops.run("Check server health and deploy 'payment-service' to production")
```

**What happens if agent tries dangerous command?**
```python
# Agent attempts: "shell_execute('rm -rf /')"
# → BLOCKED by whitelist ✋
# → Returns error: "Command 'rm' not allowed"
```

### 🔑 Key Concepts

**1. Shell Whitelisting**

Prevent catastrophic mistakes:

```python
# Safe setup
safe_shell = ShellTool(
    allowed_commands=["ls", "git", "df"],  # Whitelist
)

# Dangerous setup (DON'T DO THIS)
dangerous_shell = ShellTool(
    allowed_commands=["rm", "sudo"],  # ❌ BAD
)
```

**2. Custom Deployment Tool**

Wrap risky operations in controlled tools:

```python
class DeploymentTool(Tool):
    async def execute(self, service_name: str, environment: str):
        # Your controlled deployment logic
        print(f"Deploying {service_name} to {environment}...")
        # Can include approval checks, rollback logic, etc.
        return "SUCCESS: Deployed"
```

**3. Pre-Deployment Health Checks**

Agent automatically validates safety:

```python
system_prompt = """
Rules:
1. ALWAYS check disk space with 'df' before deployment
2. ALWAYS check git status
3. If disk >90% full, abort deployment
"""
```

### 🚀 Run It

```bash
PYTHONPATH=. python3 examples/case5_devops_automation.py
```

**Expected Output:**
```
[SysOps] Checking server health...
   [Tool: shell_execute] Running: df -h /home
   Filesystem      Size  Used Avail Use%
   /dev/sda1       100G   45G   55G  45%

   [Tool: shell_execute] Running: uptime
   14:23:15 up 30 days, load average: 0.52, 0.48, 0.51

[SysOps] System healthy. Proceeding with deployment...
   [Cloud] Initiating deployment for 'payment-service' to 'production'...
   [Cloud] Health check passed.
   
SUCCESS: payment-service deployed to production (Build #124)
```

### 📖 Learning Outcomes

✅ Build safe system automation agents  
✅ Use shell whitelisting to prevent dangerous commands  
✅ Wrap critical operations in controlled tools  
✅ Implement pre-flight validation checks  

---

## 🔴 Case 6: Agent Reasoning Architectures

**File:** [`case6_reasoning_architectures.py`](case6_reasoning_architectures.py)

### 🎯 The Challenge

**Not all agent "brains" are created equal.** Different reasoning patterns excel at different tasks:

- **ReAct** = Linear, interactive (best for dynamic environments)
- **ReWOO** = Pre-planned, parallel (best for speed when tasks are known)
- **Tree-of-Thoughts** = Exploratory, multi-path (best for creative/complex problems)

**Which should you use?** This case study shows you the differences.

### 💡 The Solution

Run the **same problem** through all 3 architectures and compare:

**Problem:** *"Design a scalable real-time analytics system for 10M daily users"*

```python
# 1. ReAct: Think → Act → Observe → Repeat
react = ReActAgent(name="Engineer_ReAct", ...)
result = await react.run("Design analytics system...")

# 2. ReWOO: Plan everything first → Execute in parallel
rewoo = ReWOOAgent(name="Engineer_ReWOO", ...)
result = await rewoo.run("Design analytics system...")

# 3. ToT: Explore multiple design branches
tot = TreeOfThoughtsAgent(name="Engineer_ToT", ...)
result = await tot.run("Design analytics system...")
```

### 🔑 Conceptual Breakdown

**1. ReAct (Reasoning + Acting)**

Classic interactive loop:

```
User: "Design analytics system"
  ↓
Thought: "I should research streaming platforms"
  ↓
Action: search("kafka vs flink comparison")
  ↓
Observation: "Kafka is better for event streaming..."
  ↓
Thought: "Now I need database options"
  ↓
Action: search("time-series databases")
  ↓
...continues until final answer
```

**Best for:** Problems requiring dynamic research, where next steps depend on previous findings.

**2. ReWOO (Reasoning WithOut Observation)**

Plans everything upfront, executes in parallel:

```
User: "Design analytics system"
  ↓
Planning Phase:
  Plan Step 1: Search for streaming platforms
  Plan Step 2: Search for time-series databases  
  Plan Step 3: Search for visualization tools
  ↓
Execution Phase (PARALLEL):
  ├─ search("kafka flink") → Result A
  ├─ search("influxdb timescaledb") → Result B
  └─ search("grafana kibana") → Result C
  ↓
Synthesis: "Based on A, B, C... here's the architecture"
```

**Best for:** Problems where you know what info you need upfront. **Much faster** than ReAct.

**3. Tree-of-Thoughts (ToT)**

Explores multiple solution branches:

```
User: "Design system"
  ↓
Generate 3 different approaches:
  Branch A: Lambda Architecture (batch + stream)
  Branch B: Kappa Architecture (stream-only)
  Branch C: Microservices with event sourcing
  ↓
Evaluate each branch:
  - Complexity score
  - Scalability score
  - Cost score
  ↓
Pick best branch OR combine ideas
  ↓
Final answer
```

**Best for:** Creative/complex problems with multiple valid solutions.

### 🚀 Run It

```bash
PYTHONPATH=. python3 examples/case6_reasoning_architectures.py
```

**Expected Output:**
```
----------------------------------------
1. ReAct Agent (Standard)
----------------------------------------
   [Engineer_ReAct] Thinking: I should research streaming platforms...
   [Engineer_ReAct] Action: search("kafka vs flink")
   [Engineer_ReAct] Observation: Kafka is better for...
   ...
 Time: 8.3s

----------------------------------------
2. ReWOO (Reasoning Without Observation)
----------------------------------------
   [Engineer_ReWOO] Planning: Step 1: search streaming...
   [Engineer_ReWOO] Executing all steps in parallel...
Time: 3.1s  ⚡ (2.7x faster!)

----------------------------------------
3. Tree of Thoughts (ToT)
----------------------------------------
   [Engineer_ToT] Generating 3 architecture branches...
   [Engineer_ToT] Evaluating Branch A (Lambda)...
   [Engineer_ToT] Evaluating Branch B (Kappa)...
 Time: 12.7s

REASONING COMPARISON COMPLETE
```

### 📖 Learning Outcomes

✅ **Understand** different agent reasoning patterns  
✅ **Know when** to use ReAct vs ReWOO vs ToT  
✅ **Optimize** agent performance for your use case  
✅ **Implement** all 3 architectures with Kite  

---

## 🛠️ How to Modify Examples for Your Use Case

### 1. Start with the Closest Example

```bash
# Copy the relevant example
cp examples/case1_ecommerce_support.py my_agent.py
```

### 2. Define Your Custom Tools

```python
# Replace mock tools with your real business logic
def check_inventory(product_id: str) -> dict:
    # Call your real inventory system API
    response = requests.get(f"https://api.yourcompany.com/inventory/{product_id}")
    return response.json()

inventory_tool = ai.create_tool("check_inventory", check_inventory, 
                                "Check real-time inventory levels")
```

### 3. Update Agent Prompts

```python
agent = ai.create_agent(
    name="InventoryBot",
    system_prompt="""You are an inventory specialist for ACME Corp.
    
    Your job:
    - Check product availability using check_inventory tool
    - Provide accurate restock dates
    - Suggest alternative products if out of stock
    
    CRITICAL: Always check inventory before making promises to customers.
    """,
    tools=[inventory_tool],
    agent_type="react"
)
```

### 4. Test & Iterate

```bash
PYTHONPATH=. python3 my_agent.py
```

---

## 🎓 Learning Path

**For Beginners:**
1. **Case 1** - Learn agent basics and routing
2. **Case 2** - Understand tool composition
3. **Case 5** - See safety patterns in action

**For Intermediate:**
3. **Case 3** - Multi-step workflows
4. **Case 2** - Hybrid reasoning

**For Advanced:**
5. **Case 6** - Reasoning architecture comparison
6. **Case 4** - Multi-agent orchestration

---

## 📊 Comparison Matrix

| Feature | Case 1 | Case 2 | Case 3 | Case 4 | Case 5 | Case 6 |
|---------|:------:|:------:|:------:|:------:|:------:|:------:|
| Multi-Agent | ✅ | ❌ | ✅ | ✅✅ | ❌ | ✅ |
| Web Search | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| Code Execution | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Shell Access | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| Checkpointing | ❌ | ❌ | ✅ | ✅ | ❌ | ❌ |
| HITL | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ |
| Guardrails | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ |
| Routing | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| Smart/Fast Models | ❌ | ✅ | ✅ | ✅ | ❌ | ❌ |

---

## 🤝 Contributing Examples

Have a cool use case? Contribute it!

**Requirements:**
- Production-ready code (with error handling)
- Detailed inline comments
- README section explaining the concept
- Test data included (mock DBs, fixtures)

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines.

---

## 💬 Need Help?

- **💡 Ask Questions:** [GitHub Discussions](https://github.com/thienzz/Kite/discussions)
- **🐛 Report Issues:** [GitHub Issues](https://github.com/thienzz/Kite/issues)

---

<p align="center">
  <strong>These aren't toy demos - they're production blueprints.</strong><br>
  Copy, customize, ship. 🚀
</p>
