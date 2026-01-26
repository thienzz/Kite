import asyncio
import os
import time
from kite.core import Kite

async def run_linkedin_scraper():
    ai = Kite()

    # Enable Real-time Dashboard Monitoring
    ai.event_bus.add_relay("http://localhost:8000/events")
    ai.enable_verbose_monitoring()

    # 1. EXPLICIT KNOWLEDGE DECLARATION
    ai.add_knowledge_source(
        source_type="local_json", 
        path="knowledge/linkedin_queries.json", 
        name="linkedin_expertise"
    )

    # 2. DEFINE SPECIALIZED AGENTS
    # Note: search_linkedin and other tools are registered NATIVELY by the framework
    search_agent = ai.create_agent(
        name="Searcher",
        system_prompt=(
            "You are a lead generation expert. "
            "MISSION: Find high-quality B2B leads on LinkedIn. "
            "Consult your AUTHORIZED EXPERTISE to get high-performance Boolean search strings. "
            "Use those strings in 'search_linkedin'. "
            "RETURN the raw JSON results with post_link and profile_link preserved."
        ),
        tools=[ai.tools.get("search_linkedin")],
        agent_type="react",
        knowledge_sources=["linkedin_expertise"]
    )

    research_agent = ai.create_agent(
        name="Researcher",
        system_prompt=(
            "Given raw LinkedIn data, visit profiles and companies to verify seniority. "
            "STRICT RULE: Only research people present in the input. Do NOT invent people. "
            "Preserve all links and EXPOSE all content for the Aggregator."
        ),
        tools=[ai.tools.get("get_profile"), ai.tools.get("get_company")],
        agent_type="react"
    )

    aggregator_agent = ai.create_agent(
        name="Aggregator",
        system_prompt=(
            "Analyze lead data and produce a deep, transparent Final Report. "
            "Include Name, Links, FULL POST CONTENT, and detailed seniority insights. "
            "If links or content are missing, report 'Data Missing'. NEVER invent data."
        ),
        agent_type="base"
    )

    # 3. DEFINE WORKFLOW STEPS
    async def step_search(input_query):
        print("\n[Step 1] Searching for leads via framework-native tools...")
        res = await search_agent.run(f"Find 3 contacts for: {input_query}")
        return res['response']

    async def step_research(raw_data):
        print("\n[Step 2] Deep Research & Verification...")
        if not raw_data or "No leads found" in str(raw_data) or "No final answer" in str(raw_data):
            return "No leads to research."
        res = await research_agent.run(f"Verify and enrich these leads. Preserve all links: {raw_data}")
        return res['response']

    async def step_aggregate(enriched_data):
        print("\n[Step 3] Final Report Generation...")
        if "No leads" in str(enriched_data):
            return "Final Report: No verifiable leads found in this session."
        res = await aggregator_agent.run(f"Format this data into a professional report: {enriched_data}")
        return res['response']

    # 4. ORCHESTRATE PIPELINE
    pipeline = ai.create_workflow("LinkedIn-Lead-Scraper")
    pipeline.add_step("Search", step_search)
    pipeline.add_step("Research", step_research)
    pipeline.add_step("Aggregate", step_aggregate)

    print("\n🚀 STARTING NATIVE LINKEDIN LEAD SCRAPER WORKFLOW")
    start_time = time.time()
    
    initial_query = "Healthtech startup looking for developer"
    final_state = await pipeline.execute_async(initial_query)

    duration = time.time() - start_time
    
    if final_state.status.value == "completed":
        print("\n" + "="*50)
        print("🏆 FINAL SCRAPER REPORT")
        print("="*50)
        print(final_state.results["Aggregate"])
        print(f"\n✅ Workflow completed in {duration:.2f}s")
    else:
        print(f"\n❌ Pipeline failed: {final_state.errors}")

if __name__ == "__main__":
    asyncio.run(run_linkedin_scraper())
