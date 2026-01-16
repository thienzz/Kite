from agentic_framework import AgenticAI

def test_config():
    ai = AgenticAI()
    
    print("\n--- Testing Per-Agent Config ---")
    
    # Agent 1: Default (from AI core)
    agent_default = ai.create_agent("DefaultAgent", "Prompt")
    print(f"Agent 1 ({agent_default.name}):")
    print(f"  LLM: {agent_default.metadata['llm']}")
    
    # Agent 2: Overridden with Mock
    agent_custom = ai.create_agent(
        "CustomAgent", 
        "Prompt",
        llm_provider="mock",
        slm_provider="mock"
    )
    print(f"Agent 2 ({agent_custom.name}):")
    print(f"  LLM: {agent_custom.metadata['llm']}")
    print(f"  SLM: {agent_custom.metadata['slm']}")
    
    if agent_custom.metadata['llm'] == "Mock/LLM":
        print("\n✅ SUCCESS: Per-agent configuration verified!")
    else:
        print("\n❌ FAILED: Per-agent configuration mismatch.")

if __name__ == "__main__":
    test_config()
