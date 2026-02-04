import json
import os
import asyncio
import time
import random
from datetime import datetime
from kite.core import Kite

async def run_linkedin_scraper_reactive():
    # DEBUG
    import kite
    print(f"\n[DEBUG] Kite package path: {kite.__file__}\n")
    
    ai = Kite()
    
    # 1. NEW: Native Monitoring & State Tracking
    ai.enable_tracing("process_trace.json")
    state_tracker = ai.enable_state_tracking("scraper_session.json", {
        "pipeline:lead_result": "leads"
    })
    
    ai.add_event_relay("http://127.0.0.1:8000/events")
    ai.enable_verbose_monitoring()
    ai.config['max_iterations'] = 15

    ai.add_knowledge_source(
        source_type="local_json", 
        path="knowledge/linkedin_queries.json", 
        name="linkedin_expertise"
    )

    # Define Agents
    search_agent = ai.create_agent(
        name="Searcher",
        system_prompt=(
            "You are a LinkedIn Search Specialist. Your ONLY job is to execute the specific query provided to you.\n\n"
            "## 🔍 SEARCH RULES:\n"
            "1. **Use the precise query string** provided in the instruction.\n"
            "2. **Call search_linkedin** immediately with that query.\n"
            "3. **Return once complete**. Do not try to be creative or explore other topics.\n"
        ),
        tools=[ai.tools.get("search_linkedin")],
        agent_type="react",
        knowledge_sources=["linkedin_expertise"]
    )
    search_agent.max_iterations = 3 
    
    classifier_agent = ai.create_agent(
        name="Classifier",
        system_prompt=(
            "You are a STRATEGIC SALES FILTER. Your goal is to identify authors who need development help or represent significant market signals.\n\n"
            "## ✅ TARGET LEADS (ACCEPT):\n"
            "1. **Direct Dev Needs**: Founders/execs looking for developers, agencies, or MVP builders.\n"
            "2. **Specialized Tech Hiring**: Posts looking for 'Lead Engineer', 'iOS Engineer', 'Data Engineer', 'AI Engineer', or 'Integration Engineer'.\n"
            "3. **Technical Leadership**: Hiring for 'CTO', 'Technical Co-founder', 'Founding Engineer', or 'Head of Tech/Labs'.\n"
            "4. **Product Leadership (FUNDED ONLY)**: Hiring for 'Product Manager', 'Product Owner', 'Product Lead', or 'Senior Product Designer' ONLY IF the company is mentioned as 'Seed', 'Series A/B', 'Funded', or 'Backed'.\n"
            "5. **Market Signals (NEW)**: Founders or leaders discussing regional growth, regulatory changes (e.g., Vietnam Crypto Law), or ecosystem shifts that imply upcoming dev demand.\n"
            "6. **Urgent Problems**: Mentioning 'launching stealth', 'ASAP', 'immediately', 'freelance or tenured', or 'remote'.\n\n"
            "## 🕵️‍♂️ REJECT CRITERIA (REJECT IF ANY ARE TRUE):\n"
            "1. **Junior/Student**: Is it for 'Interns', 'Sophomores', or 'Students'? REJECT.\n"
            "2. **Selling Services**: Is the author offering their OWN dev services? (Note: Founders and RECRUITERS hiring for a specific client are NOT sellers). \n"
            "   - **STRICT RULE**: Do NOT reject a lead just because they are a 'Recruiter' or 'Talent Agent' IF they are hiring for high-value roles listed above.\n"
            "3. **Generic Advice**: Is the post just generic motivation or advice without any context of hiring or partnership? (Note: Contextual advice about specific industries like Web3/Crypto IS accepted as a signal).\n\n"
            "OUTPUT FORMAT: You MUST start your response with 'Final Answer: LEAD [reason]' or 'Final Answer: REJECT [reason]'."
        ),
        agent_type="base"
    )

    research_agent = ai.create_agent(
        name="Researcher",
        system_prompt=(
            "You are a Background Investigator. Your goal is to confirm if a person is a REAL BUYER, HIRER, or STRATEGIC SIGNAL with intent or influence in software development.\n\n"
            "## 🎯 HIGH-PRIORITY LEAD TYPES:\n"
            "- 'Partner', 'Lead', or 'Head of' roles in 'Labs', 'Innovation', or 'Product' units.\n"
            "- Founders hiring 'Founding Engineers', 'iOS/Android Devs', or 'Data/AI Engineers'.\n"
            "- **RECRUITERS/TALENT AGENTS** hiring for 'Series A/B' or 'Funded' startups.\n"
            "- Ecosystem leaders or influencers discussing scalable tech shifts (AI, Crypto, Web3).\n\n"
            "## 🔍 STEP-BY-STEP PROCESS:\n"
            "1. Use 'get_profile' to check seniority. **CRITICAL**: If the 'name' is 'Unknown', focus on the 'headline' and 'about' section to infer authority.\n"
            "2. If the author is a Recruiter, try to identify the **Target Company** or its funding stage from the post and profile.\n"
            "3. If they are hiring for specialized tech or product leadership roles at a funded startup, treat it as a SERVICE SALES OPPORTUNITY.\n"
            "4. Use 'get_company' ONLY if you see a specific target company name. \n\n"
            "## 🛑 AUTO-REJECT:\n"
            "- Role is 'Intern' or 'Junior developer'.\n"
            "- Profile headline is a 'Software Consultant' seeking their OWN clients for dev work (Recruiters are NOT auto-rejected).\n\n"
            "OUTPUT RULES:\n"
            "- You MUST start your final answer with EXACTLY the word 'LEAD' or 'REJECT' (e.g., 'Final Answer: LEAD - [Reasoning]').\n"
            "- NEVER start the final answer with the person's name or a sentence like 'I have found...'.\n"
            "- This is CRITICAL for our automated filters. If you find seniority, high-value hiring, or strategic signals, start with 'LEAD'.\n\n"
            "CRITICAL: Always use the 'Thought:' and 'Action:' format."
        ),
        tools=[ai.tools.get("get_profile"), ai.tools.get("get_company")],
        agent_type="react"
    )

    aggregator_agent = ai.create_agent(
        name="Aggregator",
        system_prompt=(
            "You are a Senior Sales Analyst. Create a high-quality Markdown report summarizing the found leads. "
            "Organize them into 'Qualified Leads' and 'Rejected/Low Interest'. "
            "For each lead, include name, profile link, and a brief summary of why they were included/rejected. "
            "CRITICAL: You MUST start your response strictly with 'Final Answer:' followed by your Markdown report."
        ),
        agent_type="base"
    )

    # 1. Pipeline Definition
    pipeline = ai.create_reactive_workflow("Reactive-LinkedIn-Scraper")

    # STAGE 1: Discovery (Streaming Producer)
    async def stage_discovery(query):
        print(f"\n[Discovery] Starting systematic streaming search...")
        queue = asyncio.Queue()
        
        async def on_post_discovered(event, data):
            post = data.get("post")
            if post:
                await queue.put(post)

        ai.event_bus.subscribe("scraper:post_discovered", on_post_discovered)
        
        # Load all categories from knowledge to ensure exhaustion
        try:
            with open("knowledge/linkedin_queries.json", "r") as f:
                queries_config = json.load(f)
            categories = list(queries_config.keys())
            print(f"   [Discovery] Found {len(categories)} categories for exhaustion.")
        except Exception as e:
            print(f"   [Discovery] Error loading categories: {e}")
            categories = ["high_value_hiring"]

        # Track processed categories to ensure we hit them all
        for category in categories:
            queries = queries_config.get(category, [])
            print(f"   [Discovery] 🎯 Mission: Exhausting category '{category}' ({len(queries)} queries)")
            
            for i, query_string in enumerate(queries):
                print(f"   [Discovery]   - Query {i+1}/{len(queries)}: {query_string}")
                
                # Run the search agent for this SPECIFIC query string
                # We constrain it to focus only on this query to avoid "agent wandering"
                search_mission = f"Search LinkedIn for this exact query: {query_string}. Limit 40 results."
                
                search_task = asyncio.create_task(search_agent.run(search_mission))
                
                # Stream items from queue while this specific query is being searched
                while not (search_task.done() and queue.empty()):
                    try:
                        post = await asyncio.wait_for(queue.get(), timeout=1.0)
                        print(f"   [Discovery] Streaming post to Analysis: {post.get('author', {}).get('name')}")
                        yield post
                        queue.task_done()
                    except asyncio.TimeoutError:
                        if search_task.done(): break
                        continue
                
                res = await search_task
                if not res.get("success"):
                    print(f"   [Discovery] ⚠️ Query failed: {query_string}")

        # Final cleanup sweep of the queue
        while not queue.empty():
            post = await queue.get()
            yield post
            queue.task_done()
        
        # Cleanup
        ai.event_bus.unsubscribe("scraper:post_discovered", on_post_discovered)
        print("   [Discovery] Discovery stage completed all categories.")

    # STAGE 2: Analysis (Background Workers) - TEMPORARILY DISABLED LLM FOR COLLECTION-ONLY MODE
    async def stage_analysis(post):
        lead_name = post.get('author', {}).get('name', 'Unknown')
        profile_link = post.get('author', {}).get('profile_link')
        post_content = post.get('content', '')
        
        if not post_content or len(post_content) < 20:
            return None

        # Log to classifier_results.jsonl without calling LLM (Search Step Only)
        print(f"   [Analysis] 📥 Recorded Search Result: {lead_name}")
        log_agent_result("classifier_results.jsonl", "Searcher", lead_name, profile_link, post_content, "RAW_DATA_LOG")
        
        # EMIT FOR TRACE ONLY
        ai.event_bus.emit("pipeline:lead_result", {
            "name": lead_name, 
            "status": "recorded", 
            "reason": "LLM Analysis Disabled",
            "url": profile_link
        })
        return {"name": lead_name, "status": "recorded"}

    # Add stages to pipeline
    pipeline.add_stage("Discovery", stage_discovery, workers=1)
    pipeline.add_stage("Analysis", stage_analysis, workers=1) # Minimal workers as it's just logging

    print("\n🚀 STARTING REACTIVE LINKEDIN LEAD SCRAPER")
    start_time = time.time()
    query = "Founders or early-stage startups searching for developers to build an MVP"
    
    try:
        # Run the pipeline
        task_id = await pipeline.execute(query)
        
        # Wait for all workers to finish
        await pipeline.wait_until_complete()
        
        duration = time.time() - start_time
        print(f"\n✅ All background workers finished in {duration:.2f}s")

        # 2. Final Aggregation (Manual after-pipeline step)
        print("\n[Final] Generating aggregate report...")
        leads_data = json.dumps(state_tracker.data["leads"])
        res = await aggregator_agent.run(f"Create a professional report from this data: {leads_data}")
        
        if res.get('success'):
            print("\n" + "="*50 + "\n🏆 FINAL REPORT\n" + "="*50)
            print(res.get('response', 'No report content.'))
            print("="*50)
        else:
            print(f"\n[Aggregator ERROR] {res.get('error')}")
            print("\nPartial Results (Raw Leads):")
            for lead in state_tracker.data["leads"]:
                print(f"- {lead.get('name')} ({lead.get('status')})")
        
        ai.print_summary()
        
    finally:
        pipeline.stop()

if __name__ == "__main__":
    asyncio.run(run_linkedin_scraper_reactive())
