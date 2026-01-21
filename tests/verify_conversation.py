import asyncio
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite

async def test_conversation():
    ai = Kite()
    
    # 1. Create Agents
    researcher = ai.create_agent(
        name="Researcher",
        system_prompt="You are a meticulous researcher. Provide facts and evidence about AI agents. If the Critic agrees, conclude with 'Final Conclusion: [summary]'."
    )
    
    critic = ai.create_agent(
        name="Critic",
        system_prompt="You are a skeptical critic. Challenge the Researcher's points with potential risks or counter-arguments. If the Researcher addresses your points, express agreement and conclude with 'I agree, let's reach a consensus'."
    )
    
    # 2. Create Conversation
    print("\n--- Testing Conversational Agents ---")
    conv = ai.create_conversation(
        agents=[researcher, critic],
        max_turns=6,
        termination_condition="consensus"
    )
    
    # 3. Run Conversation
    res = await conv.run("Discuss the future of autonomous AI agents in healthcare.")
    
    print(f"\n[OK] Conversation finished in {res['turns']} turns.")
    print(f"Termination: {res['termination']}")
    
    assert res['success'] == True
    assert res['turns'] > 1
    assert len(res['history']) == res['turns']
    
    print("\n--- History Summary ---")
    for turn in res['history']:
        print(f"[{turn['turn']}] {turn['agent']}: {turn['content'][:100]}...")

    print("\n[OK] CONVERSATIONAL AGENTS VERIFIED")

if __name__ == "__main__":
    asyncio.run(test_conversation())
