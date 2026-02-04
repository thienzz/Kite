"""
CASE 5: DEVOPS AUTOMATION AGENT
===============================
A system administrator agent capable of managing infrastructure.
Features:
- Shell Interaction (Safe Mode)
- Guardrails (Blocks 'rm -rf')
- Human-in-the-Loop for critical actions (Deploy)

Scenario:
1. User asks to deploy 'service-api' to production.
2. Agent checks system health (local disk space).
3. Agent checks git status.
4. Agent requests approval.
5. Agent triggers deployment.
"""

import os
import sys
import asyncio
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite
from kite.tools.system_tools import ShellTool
from kite.tool import Tool

# ============================================================================
# MOCK DEPLOYMENT TOOL
# ============================================================================

class DeploymentTool(Tool):
    def __init__(self):
        super().__init__(
            name="cloud_deploy",
            func=self.execute,
            description="Deploy a service to the cloud cluster. Requires 'service_name' and 'environment'."
        )
        
    async def execute(self, service_name: str, environment: str, **kwargs) -> str:
        # Mocking a long running deployment process
        print(f"   [Cloud] Initiating deployment for '{service_name}' to '{environment}'...")
        time.sleep(1)
        print("   [Cloud] Building container...")
        time.sleep(1)
        print("   [Cloud] Pushing to registry...")
        time.sleep(1)
        print("   [Cloud] Rolling update 20%...")
        time.sleep(1)
        print("   [Cloud] Health check passed.")
        return f"SUCCESS: {service_name} deployed to {environment} (Build #124)."

# ============================================================================
# MAIN
# ============================================================================

async def main():
    print("\n" + "=" * 80)
    print("CASE 5: DEVOPS AUTOMATION AGENT")
    print("=" * 80)
    
    ai = Kite()
    
    # 1. Initialize Tools
    # Strict whitelist for safety
    shell_tool = ShellTool(allowed_commands=["ls", "pwd", "git", "df", "echo", "uptime", "grep"])
    deploy_tool = DeploymentTool()
    
    # 2. Create DevOps Agent
    sysops = ai.create_agent(
        name="SysOps",
        model="groq/llama-3.3-70b-versatile",
        tools=[shell_tool, deploy_tool],
        system_prompt="""You are a Senior DevOps Engineer.
        Your goal is to maintain system health and deploy services.
        
        Rules:
        1. ALWAYS check system resources (disk/uptime) before deployment.
        2. Use 'shell_execute' to run commands.
        3. Use 'cloud_deploy' to ship code.
        4. If a command fails, investigate why.
        """,
        verbose=True
    )
    
    print("\n[User Request] 'Check the server health and if everything looks good, deploy the 'payment-service' to production.'")
    
    # Run the agent
    await sysops.run("Check server health (check disk space of /home) and if healthy, deploy 'payment-service' to production.")
    
    print("\n" + "=" * 80)
    print("CASE 5 COMPLETE - DevOps Workflow")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
