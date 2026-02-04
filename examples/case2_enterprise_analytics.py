"""
CASE 2: ENTERPRISE DATA SCIENTIST AGENT
=======================================
A production-grade Analyst that combines SQL querying with Python code execution.
1. Queries database for raw data.
2. Uses Python (pandas/matplotlib) to analyze and visualize data.
3. Generates insights report.

Features:
- Safe Python REPL (Sandboxed)
- SQL Database Tool
- Multi-step Reasoning
"""

import os
import sys
import asyncio
import sqlite3
import pandas as pd # Ensure pandas is available or mock it? 
# We assume env has pandas. If not, we should install it or mock.

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite
from kite.optimization.resource_router import ResourceAwareRouter
from kite.tools.code_execution import PythonReplTool
from kite.tool import Tool

# ============================================================================
# 1. SETUP DATABASE
# ============================================================================

def setup_db():
    db_path = "sales_data.db"
    conn = sqlite3.connect(db_path)
    
    # Create Tables
    conn.execute("DROP TABLE IF EXISTS sales")
    conn.execute("""
        CREATE TABLE sales (
            id INTEGER PRIMARY KEY,
            date DATE,
            product TEXT,
            category TEXT,
            amount REAL,
            region TEXT
        )
    """)
    
    # Seed Data
    data = [
        (1, '2024-01-01', 'Laptop X1', 'Electronics', 1200.0, 'North'),
        (2, '2024-01-02', 'Mouse M1', 'Accessories', 25.0, 'North'),
        (3, '2024-01-03', 'Monitor 4K', 'Electronics', 400.0, 'South'),
        (4, '2024-01-04', 'HDD 1TB', 'Storage', 80.0, 'East'),
        (5, '2024-01-05', 'Laptop Pro', 'Electronics', 2500.0, 'West'),
        (6, '2024-01-06', 'USB-C Cable', 'Accessories', 15.0, 'South'),
        (7, '2024-01-07', 'Headphones', 'Audio', 150.0, 'North'),
        (8, '2024-01-08', 'Webcam', 'Electronics', 60.0, 'East'),
    ]
    conn.executemany("INSERT INTO sales VALUES (?,?,?,?,?,?)", data)
    conn.commit()
    conn.close()
    return db_path

# ============================================================================
# 2. DEFINE SQL TOOL
# ============================================================================

class SQLTool(Tool):
    def __init__(self, db_path):
        super().__init__(
            name="query_sql",
            func=self.execute,
            description="Execute SQL query on 'sales_data.db'. Tables: sales(id, date, product, category, amount, region)."
        )
        self.db_path = db_path
        
    async def execute(self, query: str, **kwargs) -> str:
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df.to_markdown(index=False)
        except Exception as e:
            return f"SQL Error: {e}"

# ============================================================================
# 3. MAIN WORKFLOW
# ============================================================================

async def main():
    print("\n" + "=" * 80)
    print("📊 CASE 2: ENTERPRISE DATA SCIENTIST AGENT")
    print("=" * 80)
    
    # Setup
    db_path = setup_db()
    ai = Kite()
    router = ResourceAwareRouter(ai.config)
    
    # Tools
    sql_tool = SQLTool(db_path)
    python_tool = PythonReplTool()
    
    # Agent
    # We use SMART model because Python coding + Analytics requires reasoning
    analyst = ai.create_agent(
        name="DataCientist",
        model=router.smart_model,
        tools=[sql_tool, python_tool],
        system_prompt="""You are a Lead Data Scientist.
        Goal: Analyze sales data and provide actionable insights.
        
        Capabilities:
        1. 'query_sql': To fetch data frame from database.
        2. 'python_repl': To process data, calculate stats, or create ASCII charts.
        
        Process:
        1. Explore data using SQL.
        2. Use Python to calculate advanced metrics (e.g. % share, growth) and visualize.
        3. Output text summary of findings.
        
        Visualization:
        If you make a plot, save it as 'sales_chart.png' using matplotlib.
        """,
        verbose=True
    )
    
    # Execution
    print("\n[User Request] 'Analyze the sales performance by Region. I want to know who is performing best.'")
    
    result = await analyst.run("Analyze sales performance by Region. Provide a breakdown and a chart.")
    
    print("\n" + "=" * 80)
    print("Final Analysis Report")
    print("=" * 80)
    print(result['response'])
    
    if os.path.exists("sales_chart.png"):
        print("\n[System] Chart generated successfully: sales_chart.png 📈")
    
    # Cleanup
    if os.path.exists(db_path):
        os.remove(db_path)

if __name__ == "__main__":
    asyncio.run(main())
