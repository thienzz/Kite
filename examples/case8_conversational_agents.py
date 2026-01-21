"""
Case Study 8: Conversational Agents
Demonstrates multi-agent collaboration with debate and consensus patterns.
"""

import asyncio
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite.core import Kite

async def main():
    print("="*80)
    print("CASE STUDY 8: MULTI-AGENT DEBATE ON AI ETHICS")
    print("="*80)

    ai = Kite()
    
    # Define Agents
    researcher = ai.create_agent(
        name="Researcher",
        system_prompt="""You are an AI Ethics Researcher. 
        Provide objective data and technological trends regarding AI alignment.
        Focus on safety and performance.
        If a consensus is reached, clearly state 'CONSENSUS REACHED'."""
    )
    
    critic = ai.create_agent(
        name="Critic",
        system_prompt="""You are a Skeptical Ethicist. 
        Challenge the Researcher's points with potential societal risks, bias, and equity concerns.
        Look for flaws in the 'objective' data.
        If your concerns are addressed, express agreement and mention 'CONSENSUS REACHED'."""
    )
    
    synthesizer = ai.create_agent(
        name="Synthesizer",
        system_prompt="""You are a Mediator and Synthesizer. 
        Your goal is to find a middle ground between the Researcher and the Critic.
        Summarize the key points and propose a balanced path forward.
        If both sides seem to agree, conclude with 'FINAL CONSENSUS REACHED'."""
    )

    # Note: In a real scenario, we might want a custom orchestration logic,
    # but for this demo, round-robin between [R, C, S] works.
    
    print("\nStarting multi-turn debate...")
    
    conversation = ai.create_conversation(
        agents=[researcher, critic, synthesizer],
        max_turns=6,
        min_turns=3,
        termination_condition="consensus"
    )
    
    res = await conversation.run("Should autonomous AI agents be allowed to make financial investment decisions without human oversight?")
    
    print(f"\n[DEBATE SUMMARY]")
    print(f"Turns: {res['turns']}")
    print(f"Termination: {res['termination']}")
    
    # We could also show the full history here
    # for turn in res['history']:
    #     print(f"\n--- {turn['agent']} ---")
    #     print(turn['content'])

    print("\n" + "="*80)
    print("[OK] CASE STUDY 8 COMPLETE")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())
