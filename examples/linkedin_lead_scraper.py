import json
import os
import asyncio
import time
from datetime import datetime
from kite.core import Kite

class TraceSink:
    def __init__(self, trace_file="process_trace.json"):
        if not os.path.isabs(trace_file):
            trace_file = os.path.join(os.getcwd(), trace_file)
        self.trace_file = trace_file
        # Initialize file as an empty list
        with open(self.trace_file, "w") as f:
            f.write("[\n]")
    
    def on_event(self, event, data):
        """Append event to the JSON list file."""
        try:
            # Simple append logic for a JSON list
            with open(self.trace_file, "rb+") as f:
                f.seek(-2, os.SEEK_END)
                pos = f.tell()
                f.truncate()
                if pos > 2: f.write(b",\n")
                
                log_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "event": event,
                    "data": self._sanitize(data)
                }
                json_str = json.dumps(log_entry, indent=4)
                f.write(json_str.encode('utf-8'))
                f.write(b"\n]")
        except Exception as e:
            pass
    
    def _sanitize(self, data):
        """Ensure data is JSON serializable."""
        if isinstance(data, dict):
            return {k: self._sanitize(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._sanitize(v) for v in data]
        elif isinstance(data, (str, int, float, bool, type(None))):
            return data
        else:
            return str(data)

class LeadTracker:
    def __init__(self, session_file="scraper_session.json"):
        # Ensure we use an absolute path for reliability
        if not os.path.isabs(session_file):
            session_file = os.path.join(os.getcwd(), session_file)
        self.session_file = session_file
        
        # Move datetime import up if needed, but it's handled at module level now
        from datetime import datetime
        self.data = {
            "query": "",
            "start_time": datetime.now().isoformat(),
            "leads": [],
            "status": "starting"
        }
    
    def update_leads(self, raw_posts):
        for post in raw_posts:
            name = post.get("author", {}).get("name", "Unknown")
            # Check for duplicates by name
            if not any(l["name"] == name for l in self.data["leads"]):
                self.data["leads"].append({
                    "name": name,
                    "title": post.get("author", {}).get("title", "LinkedIn User"),
                    "profile_link": post.get("author", {}).get("profile_link", ""),
                    "post_link": post.get("post_link", ""),
                    "post_content": post.get("content", ""),
                    "status": "found",
                    "research_analysis": ""
                })
        self.save()

    def update_research(self, name, status, analysis):
        for lead in self.data["leads"]:
            if lead["name"] == name:
                lead["status"] = status
                lead["research_analysis"] = analysis
                break
        self.save()

    def save(self):
        try:
            with open(self.session_file, "w") as f:
                json.dump(self.data, f, indent=4)
            print(f"Session log updated: {self.session_file}")
        except Exception as e:
            print(f"Error saving session log: {e}")

class LiveLeadReporter:
    def __init__(self, output_file="live_leads.md"):
        if not os.path.isabs(output_file):
            output_file = os.path.join(os.getcwd(), output_file)
        self.output_file = output_file
        # Initialize the file with headers
        with open(self.output_file, "w") as f:
            f.write("# 🚀 LIVE LEAD REPORT\n")
            # Use current time
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"*Last updated: {now}*\n\n")
            f.write("This file is updated in real-time as leads are verified.\n\n")
            f.write("---\n\n")

    def report_lead(self, lead_name, status, analysis, profile_link, post_link, post_content):
        emoji = "🔥 [HOT]" if status == "match" else "🟡 [WARM]"
        with open(self.output_file, "a") as f:
            f.write(f"## {emoji} {lead_name}\n")
            f.write(f"- **Profile**: {profile_link}\n")
            f.write(f"- **Post Link**: {post_link}\n")
            f.write(f"- **Analysis**: {analysis}\n")
            f.write(f"- **Full Content**:\n```text\n{post_content}\n```\n")
            f.write("\n---\n\n")



async def run_linkedin_scraper():
    ai = Kite()
    tracker = LeadTracker()
    reporter = LiveLeadReporter()
    trace = TraceSink()

    # Capture EVERY event for the process trace (Agent responses, tool responses, thoughts)
    ai.event_bus.subscribe("*", trace.on_event)

    # Immediacy: Background Research Queue
    research_queue = asyncio.Queue()
    research_tasks = set()

    async def background_worker():
        """Continuously process leads from the research_queue."""
        print("[System] Background Research Worker Started.")
        processed_count = 0
        while True:
            lead_data = await research_queue.get()
            if lead_data is None: # Shutdown signal
                break
            
            lead_name = lead_data.get('author', {}).get('name', 'Unknown')
            profile_link = lead_data.get('author', {}).get('profile_link')
            
            if profile_link and lead_name != "Unknown":
                print(f"   [Worker] Evaluating post for: {lead_name}")
                ai.event_bus.emit("scraper:research_start", {"name": lead_name})
                
                try:
                    # 1. CLASSIFY (Fast, no tools)
                    post_content = lead_data.get('content', '')
                    classification_res = await classifier_agent.run(f"Evaluate this post for sales potential (IT Consulting/MVP): {post_content}")
                    
                    
                    res_text = classification_res['response']
                    print(f"   [Classifier] Decision for {lead_name}: {res_text}")
                    is_lead = "LEAD" in res_text.upper()
                    
                    if not is_lead:
                        print(f"   [Worker] Rejected (Low Interest): {lead_name}")
                        tracker.update_research(lead_name, "rejected", "Post classified as low interest.")
                    else:
                        print(f"   [Worker] Potential Lead! Starting deep research for: {lead_name}")
                        # 2. RESEARCH (Deep discovery)
                        research_res = await research_agent.run(f"Identify if this person is a founder/leader and evaluate if they need IT consulting or MVP development services based on their post: {json.dumps(lead_data)}")
                        
                        analysis = research_res['response']
                        # Look for lead detection signal in analysis
                        is_verified = any(k in analysis.lower() for k in ["lead detected", "high potential", "strong match", "verified"])
                        has_keywords = any(k in analysis.lower() for k in ["consult", "mvp", "developer", "hire", "build", "tech"])
                        
                        status = "match" if (is_verified or has_keywords) else "rejected"
                        research_agent.record_outcome("lead" if status == "match" else "reject")
                        
                        # HIGH VISIBILITY DASHBOARD & TRACE HIGHLIGHT
                        tracker.update_research(lead_name, status, analysis)
                        ai.event_bus.emit("scraper:research_update", {
                            "name": lead_name,
                            "status": status,
                            "analysis": analysis[:200]
                        })

                        if status == "match":
                            # Emit verified lead event for dashboard
                            ai.event_bus.emit("pipeline:lead_verified", {
                                "name": lead_name,
                                "url": profile_link,
                                "post_url": lead_data.get('post_link'),
                                "analysis": analysis
                            })
                            # LOUD TRACE LOG for verified lead
                            ai.event_bus.emit("tool:log", {
                                "agent": "Pipeline",
                                "message": f"🔥 [HOT LEAD DETECTED] 🔥\nName: {lead_name}\nURL: {profile_link}\nNeed: {analysis[:150]}..."
                            })
                            # Report to live file
                            reporter.report_lead(lead_name, status, analysis, profile_link, lead_data.get('post_link'), post_content)
                        elif status == "rejected":
                             # Detail for false leads to help Aggregator later
                             ai.event_bus.emit("tool:log", {
                                "agent": "Pipeline",
                                "message": f"🚫 [REJECTED: SELLER/INFLUENCER] 🚫\nName: {lead_name}\nReason: {analysis[:100]}..."
                            })

                        print(f"   [Worker] Deep Research completed for: {lead_name} (Status: {status})")
                except Exception as e:
                    print(f"   [Worker] Error processing {lead_name}: {e}")
            
            research_queue.task_done()
            processed_count += 1

    # Start the worker
    worker_task = asyncio.create_task(background_worker())

    # Real-time lead extraction from events
    def on_lead_discovered(event, data):
        post = data.get("post")
        if post:
            lead_name = post.get("author", {}).get("name", "Unknown")
            print(f"   [Tracker] Real-time Discovery: {lead_name}")
            tracker.update_leads([post])
            
            # Update Dashboard Immediately
            ai.event_bus.emit("scraper:search_data", {"posts": [post]})
            
            # Immediately queue for research
            research_queue.put_nowait(post)
    
    ai.event_bus.subscribe("scraper:post_discovered", on_lead_discovered)

    # Enable Real-time Dashboard Monitoring
    ai.event_bus.add_relay("http://localhost:8000/events")
    ai.enable_verbose_monitoring()

    ai.config['max_iterations'] = 15 # Set for all agents in this workflow

    # 1. EXPLICIT KNOWLEDGE DECLARATION
    ai.add_knowledge_source(
        source_type="local_json", 
        path="knowledge/linkedin_queries.json", 
        name="linkedin_expertise"
    )

    # 2. DEFINE SPECIALIZED AGENTS
    search_agent = ai.create_agent(
        name="Searcher",
        system_prompt=(
            "You are a lead generation expert. MISSION: Find technical founders and startup leaders seeking MVPs. "
            "STRICT SEARCH POLICY: "
            "1. ALWAYS consult your AUTHORIZED EXPERTISE first. "
            "2. IMPORTANT: You MUST provide a 'query' argument to the 'search_linkedin' tool (e.g., {'query': '...'}) or it will fail. "
            "3. **FILTERING**: You MUST ALWAYS include 'date_posted': 'past-week' in your tool call to return only recent results. "
            "4. If search results are limited, just return the posts you found. "
            "5. RETURN the raw list of posts including content and author details."
        ),
        tools=[ai.tools.get("search_linkedin")],
        agent_type="react",
        knowledge_sources=["linkedin_expertise"]
    )
    
    classifier_agent = ai.create_agent(
        name="Classifier",
        system_prompt=(
            "You are a CYNICAL SALES FILTER. Your job is to TRASH any post that is selling services. "
            "Analyze the post content and determine the author's INTENT. "
            
            "⛔ FATAL REJECTION CRITERIA (Mark as REJECT immediately): "
            "1. SELLER PATTERNS: 'I help founders...', 'We build MVPs...', 'Stop wasting time...', 'I am a developer...'. "
            "2. AGENCY LANGUAGE: 'Our team', 'We offer', 'DM me for a quote', 'Launch in 2 weeks'. "
            "3. EDUCATIONAL/ADVICE: 'How to build...', 'Why startups fail...', 'Tips for...'. "
            "4. JOB SEEKERS: 'Open for work', 'Looking for roles'. "
            
            "✅ ACCEPTANCE CRITERIA (Only mark as LEAD if): "
            "1. EXPLICIT HIRING: 'Looking for a developer', 'Hiring a CTO', 'Need someone to build', 'Who can help me make this app?'. "
            "2. FOUNDER STRUGGLE: 'Struggling to find tech co-founder', 'My dev left, need help'. "
            
            "*** IMPLICIT THOUGHT PROCESS ***"
            "1. WHO is writing? (Agency owner? Freelancer? Or Startup Founder?)"
            "2. WHAT do they want? (To sell me dev hours? Or to pay someone for code?)"
            
            "OUTPUT FORMAT: Start strictly with 'LEAD' or 'REJECT' followed by a 5-word reason."
            "Example: 'REJECT: Author is selling web services.'"
            "Example: 'LEAD: Founder explicitly asking for CTO.'"
        ),
        agent_type="base"
    )

    research_agent = ai.create_agent(
        name="Researcher",
        system_prompt=(
            "You are a Background Investigator. Validate if this person is a REAL BUYER. "
            "Use the provided JSON data (profile & post). "
            
            "🕵️ INVESTIGATION STEPS: "
            "1. CHECK HEADLINE: Does their headline say 'CEO of [Dev Agency]' or 'Helping Founders Build'? -> REJECT (Competitor). "
            "2. CHECK CONTENT AGAIN: specifically look for 'I help', 'I offer'. If present -> REJECT. "
            "3. CONFIRM NEED: Do they specifically say they need to HIRE or PARTNER? "
            
            "OUTPUT RULES: "
            "- If they are a Service Provider/Agency/Coach: output 'REJECTED: Service Provider'. "
            "- If they are a Job Seeker: output 'REJECTED: Job Seeker'. "
            "- If they are a Founder needing tech help: output 'LEAD DETECTED' followed by specific needs. "
        ),
        tools=[ai.tools.get("get_profile"), ai.tools.get("get_company")],
        agent_type="react"
    )

    aggregator_agent = ai.create_agent(
        name="Aggregator",
        system_prompt=(
            "You are a Senior Sales Coordinator. Produce a clean Markdown report. "
            "ONLY include leads marked as 'match' or 'LEAD DETECTED' by previous agents. "
            "Ignore rejected entries. "
            "Format: Name | Profile URL | Need Summary."
            "Start response with 'Final Answer:'"
        ),
        agent_type="base"
    )

    # 3. DEFINE WORKFLOW STEPS
    async def step_search(input_query):
        tracker.data["query"] = input_query
        print("\n[Step 1] Searching for leads via framework-native tools...")
        res = await search_agent.run(f"Find high-quality contacts for: {input_query}. Scan at least 30 posts.")
        
        # Wait for any remaining background research to finish
        print("\n[System] Waiting for background research to complete...")
        await research_queue.join()
        
        return "Search and background research complete."

    async def step_research(raw_data):
        # This step is now largely handled by the background worker, 
        # but we can use it to verify any outliers or summarize state.
        print("\n[Step 2] Syncing Research Results...")
        total_leads = len(tracker.data["leads"])
        researched = len([l for l in tracker.data["leads"] if l["status"] != "found"])
        print(f"   [Sync] {researched}/{total_leads} leads processed via immediate background worker.")
        return json.dumps(tracker.data["leads"])

    async def step_aggregate(enriched_data):
        print("\n[Step 3] Final Report Generation...")
        if not tracker.data["leads"]: return "No leads verified."
        res = await aggregator_agent.run(f"Create a professional report based on these leads: {enriched_data}")
        
        tracker.data["status"] = "completed"
        tracker.save()
        ai.event_bus.emit("scraper:final_report", {"report": res['response']})
        return res['response']

    # 4. ORCHESTRATE PIPELINE
    pipeline = ai.create_workflow("LinkedIn-Lead-Scraper")
    pipeline.add_step("Search", step_search)
    pipeline.add_step("Research", step_research)
    pipeline.add_step("Aggregate", step_aggregate)

    print("\n🚀 STARTING NATIVE LINKEDIN LEAD SCRAPER WORKFLOW")
    start_time = time.time()
    initial_query = "Founders or early-stage startups searching for developers to build an MVP"
    
    try:
        final_state = await pipeline.execute_async(initial_query)
        duration = time.time() - start_time
        
        if final_state.status.value == "completed":
            print("\n" + "="*50 + "\n🏆 FINAL SCRAPER REPORT\n" + "="*50)
            print(final_state.results["Aggregate"])
            ai.print_summary()
            print(f"\n✅ Workflow completed in {duration:.2f}s")
        else:
            print(f"\n❌ Pipeline failed: {final_state.errors}")
    finally:
        # Cleanup worker
        research_queue.put_nowait(None)
        await worker_task

if __name__ == "__main__":
    asyncio.run(run_linkedin_scraper())
