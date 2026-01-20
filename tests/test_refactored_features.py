import pytest
import asyncio
from agentic_framework import AgenticAI
from agentic_framework.safety.kill_switch import KillSwitch
from agentic_framework.agents.react_agent import ReActAgent

def test_kill_switch_iterations():
    ks = KillSwitch(max_iterations=5)
    state = {'steps': 5, 'total_cost': 0.1, 'actions': [], 'completed': False}
    should_stop, reason = ks.check(state)
    assert should_stop is True
    assert "iterations" in reason

def test_kill_switch_cost():
    ks = KillSwitch(max_cost=0.5)
    state = {'steps': 1, 'total_cost': 0.6, 'actions': [], 'completed': False}
    should_stop, reason = ks.check(state)
    assert should_stop is True
    assert "Budget" in reason

def test_kill_switch_loop():
    ks = KillSwitch(max_same_action=2)
    state = {
        'steps': 3, 
        'total_cost': 0.1, 
        'actions': [{'type': 'search'}, {'type': 'search'}], 
        'completed': False
    }
    should_stop, reason = ks.check(state)
    assert should_stop is True
    assert "Stuck in loop" in reason

def test_contrib_tools_registry():
    ai = AgenticAI()
    assert ai.tools.get("web_search") is not None
    assert ai.tools.get("calculator") is not None
    assert ai.tools.get("get_datetime") is not None

def test_calculator_tool():
    ai = AgenticAI()
    calc = ai.tools.get("calculator")
    result = calc.execute(expression="2 + 3 * 4")
    assert result["result"] == 14

def test_react_agent_creation():
    ai = AgenticAI()
    calc_tool = ai.tools.get("calculator")
    agent = ai.create_react_agent(
        name="TestBot",
        system_prompt="Test prompt",
        tools=[calc_tool]
    )
    assert isinstance(agent, ReActAgent)
    assert agent.name == "TestBot"
    assert "calculator" in agent.tools

if __name__ == "__main__":
    # Quick manual run
    test_kill_switch_iterations()
    test_kill_switch_cost()
    test_kill_switch_loop()
    test_contrib_tools_registry()
    test_calculator_tool()
    asyncio.run(test_react_agent_creation()) # Note: wrap in asyncio.run if needed, but here we just need to test creation which is likely sync
    print("All basic tests passed!")
