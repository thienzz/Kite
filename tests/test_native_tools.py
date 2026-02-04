import asyncio

from kite.agent import Agent
from kite.tool import Tool

# 1. Mock Framework
class MockFramework:
    def __init__(self):
        class MockBus:
            def emit(self, *args): pass
        self.event_bus = MockBus()
        class MockMetrics:
            def record_request(self, *args): pass
        self.metrics = MockMetrics()

# 2. Mock LLM with Native Support
class MockNativeLLM:
    def __init__(self):
        self.model = "mock-native"
        self.calls = []
        
    async def chat_async(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        
        # Scenario: First call -> Call 'add' tool
        if len(self.calls) == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "add",
                            "arguments": '{"a": 5, "b": 3}'
                        }
                    }
                ]
            }
        
        # Scenario: Second call -> Receive tool result and finish
        if len(self.calls) == 2:
            # Verify tool output is in history
            last_msg = messages[-1]
            if last_msg.get("role") != "tool" or last_msg.get("content") != "8":
                return "Error: Tool result not found"
            
            return "Final Answer: The sum is 8"

    def complete(self, prompt): return "Final Answer: Error"

# 3. Define Tool
async def add_func(framework, a: int, b: int):
    return a + b

def test_native_tool_flow():
    # Setup
    framework = MockFramework()
    llm = MockNativeLLM()
    tool = Tool(name="add", func=add_func, description="Adds two numbers")
    
    agent = Agent(
        name="TestAgent",
        system_prompt="You are a helper.",
        llm=llm,
        tools=[tool],
        framework=framework
    )
    
    # Run
    # usage of nest_asyncio needed if running in existing loop, but pytest-asyncio handles it usually
    # or just run sync wrapper
    result = agent.run_sync("What is 5 + 3?")
    
    # Verify
    assert result["success"] is True
    assert result["response"] == "The sum is 8"
    assert "add" in result["data"]
    assert result["data"]["add"] == 8
    
    # Verify LLM interaction
    assert len(llm.calls) == 2
    assert llm.calls[0]["tools"] is not None # Tools passed in first call
    # Check schema generation
    schema = llm.calls[0]["tools"][0]
    assert schema["function"]["name"] == "add"
    assert "a" in schema["function"]["parameters"]["properties"]

if __name__ == "__main__":
    test_native_tool_flow()
    print("Test Passed!")
