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
        except Exception:
            pass
    
    def _sanitize(self, data):
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
        if not os.path.isabs(session_file):
            session_file = os.path.join(os.getcwd(), session_file)
        self.session_file = session_file
        self.data = {
            "query": "",
            "start_time": datetime.now().isoformat(),
            "leads": [],
            "status": "starting"
        }
    
    def log_lead(self, post, status="found", analysis=""):
        """Dynamically add a single lead to the session."""
        name = post.get("author", {}).get("name", "Unknown")
        # Avoid duplicates
        if not any(l["name"] == name for l in self.data["leads"]):
            self.data["leads"].append({
                "name": name,
                "title": post.get("author", {}).get("title", "LinkedIn User"),
                "profile_link": post.get("author", {}).get("profile_link", ""),
                "post_link": post.get("post_link", ""),
                "post_content": post.get("content", ""),
                "status": status,
                "research_analysis": analysis
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
        except Exception as e:
            print(f"Error saving session log: {e}")

class LiveLeadReporter:
    def __init__(self, output_file="live_leads.md"):
        if not os.path.isabs(output_file):
            output_file = os.path.join(os.getcwd(), output_file)
        self.output_file = output_file
        with open(self.output_file, "w") as f:
            f.write("# 🚀 LIVE LEAD REPORT (Reactive)\n")
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"*Last updated: {now}*\n\n")
            f.write("This file is updated in real-time via framework-native background workers.\n\n")
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

async def run_linkedin_scraper_reactive():
    ai = Kite()
    tracker = LeadTracker()
    reporter = LiveLeadReporter()
    trace = TraceSink()

    ai.event_bus.subscribe("*", trace.on_event)
    ai.event_bus.add_relay("http://localhost:8000/events")
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
            "You are a lead generation expert. Find technical founders and startup leaders seeking MVPs. "
            "Use 'date_posted': 'past-week' in your tool call. RETURN a list of posts."
        ),
        tools=[ai.tools.get("search_linkedin")],
        agent_type="react",
        knowledge_sources=["linkedin_expertise"]
    )
    
    classifier_agent = ai.create_agent(
        name="Classifier",
        system_prompt=(
            "You are a CYNICAL SALES FILTER. Determine if the author is HIRING or SELLING. "
            "OUTPUT FORMAT: You MUST start your response with 'Final Answer: LEAD [reason]' or 'Final Answer: REJECT [reason]'."
        ),
        agent_type="base"
    )

    research_agent = ai.create_agent(
        name="Researcher",
        system_prompt=(
            "You are a Background Investigator. Validate if this person is a REAL BUYER. "
            "OUTPUT RULES: 'REJECTED: ...' or 'LEAD DETECTED: ...'"
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

    # STAGE 1: Discovery (Producer-like)
    async def stage_discovery(query):
        tracker.data["query"] = query
        print(f"\n[Discovery] Searching for: {query}")
        res = await search_agent.run(f"Find leads for: {query}. Scan 30 posts.")
        # If search_agent returns a list in its response data, we need to extract it
        # ReactivePipeline will flatten this list if it's the return value
        posts = res.get('data', {}).get('search_linkedin', [])
        # tracker.update_leads(posts)  <-- REMOVED: Don't track everything immediately
        
        # Emit discovery events for dashboard (Still useful for visualization)
        if posts:
            ai.event_bus.emit("scraper:search_data", {"posts": posts})
        return posts

    # STAGE 2: Analysis (Background Workers)
    async def stage_analysis(post):
        lead_name = post.get('author', {}).get('name', 'Unknown')
        profile_link = post.get('author', {}).get('profile_link')
        print(f"   [Analysis] Processing: {lead_name}")
        
        try:
            # Classify
            post_content = post.get('content', '')
            class_res = await classifier_agent.run(f"Evaluate: {post_content}")
            if "REJECT" in class_res['response'].upper():
                tracker.update_research(lead_name, "rejected", "Post classified as low interest.")
                return None
            
            # Research
            research_res = await research_agent.run(f"Investigate buyer intent: {json.dumps(post)}")
            analysis = research_res['response']
            
            status = "match" if "LEAD DETECTED" in analysis.upper() else "rejected"
            
            # TRACK ONLY NOW: After we have analysis and status
            tracker.log_lead(post, status, analysis)
            
            # Real-time Reporting
            if status == "match":
                reporter.report_lead(lead_name, status, analysis, profile_link, post.get('post_link'), post_content)
                ai.event_bus.emit("pipeline:lead_verified", {
                    "name": lead_name, "url": profile_link, "analysis": analysis
                })
            
            return {"name": lead_name, "status": status, "analysis": analysis}
        except Exception as e:
            print(f"      [Error] Stage Analysis for {lead_name}: {e}")
            return None

    # Add stages to pipeline
    pipeline.add_stage("Discovery", stage_discovery, workers=1)
    pipeline.add_stage("Analysis", stage_analysis, workers=5) # 5 concurrent researchers!

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
        leads_data = json.dumps(tracker.data["leads"])
        res = await aggregator_agent.run(f"Create a professional report from this data: {leads_data}")
        
        if res.get('success'):
            print("\n" + "="*50 + "\n🏆 FINAL REPORT\n" + "="*50)
            print(res.get('response', 'No report content.'))
            print("="*50)
        else:
            print(f"\n[Aggregator ERROR] {res.get('error')}")
            print("\nPartial Results (Raw Leads):")
            for lead in tracker.data["leads"]:
                print(f"- {lead.get('name')} ({lead.get('status')})")
        
        ai.print_summary()
        
    finally:
        pipeline.stop()

if __name__ == "__main__":
    asyncio.run(run_linkedin_scraper_reactive())
