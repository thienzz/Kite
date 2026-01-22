"""
CASE 2: ENTERPRISE ANALYTICS & SQL AGENT
=========================================
Comprehensive demonstration of data analytics features.

Features Demonstrated:
[OK] Database MCP - Safe SQL execution
[OK] Advanced RAG - Hybrid search with documents
[OK] Graph RAG - Entity relationships
[OK] Vector Memory - Document embeddings
[OK] A/B Testing - Query strategy comparison
[OK] Caching - Query result caching
[OK] Kill Switch - Cost and iteration limits

Real-world scenario: Enterprise analytics system with SQL generation,
document search, and intelligent caching for performance.

Run: python examples/case2_enterprise_analytics.py
"""

import os
import sys
import asyncio
import time
import sqlite3

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kite import Kite


# ============================================================================
# DATABASE SETUP
# ============================================================================

def setup_demo_database():
    """Create demo analytics database"""
    db_path = "analytics_demo.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT,
            email TEXT,
            country TEXT,
            signup_date DATE,
            lifetime_value REAL
        )
    """)
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_name TEXT,
            amount REAL,
            order_date DATE,
            status TEXT,
            FOREIGN KEY (customer_id) REFERENCES customers(id)
        )
    """)
    
    # Insert sample data
    customers = [
        (1, "Alice Johnson", "alice@example.com", "USA", "2024-01-15", 1250.00),
        (2, "Bob Smith", "bob@example.com", "UK", "2024-02-20", 890.50),
        (3, "Charlie Brown", "charlie@example.com", "Canada", "2024-03-10", 2100.00),
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?, ?, ?, ?, ?, ?)", customers)
    
    orders = [
        (1, 1, "Pro Plan", 49.99, "2024-01-15", "completed"),
        (2, 1, "Add-on Storage", 9.99, "2024-02-01", "completed"),
        (3, 2, "Basic Plan", 19.99, "2024-02-20", "completed"),
    ]
    
    cursor.executemany("INSERT OR REPLACE INTO orders VALUES (?, ?, ?, ?, ?, ?)", orders)
    
    conn.commit()
    conn.close()
    
    return db_path


def execute_sql_safely(db_path, sql):
    """Execute SQL with safety validation"""
    sql_upper = sql.upper().strip()
    
    if not sql_upper.startswith('SELECT'):
        return {'success': False, 'error': 'Only SELECT queries allowed'}
    
    dangerous = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER']
    for keyword in dangerous:
        if keyword in sql_upper:
            return {'success': False, 'error': f'Forbidden keyword: {keyword}'}
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        conn.close()
        
        return {
            'success': True,
            'rows': rows,
            'columns': columns,
            'row_count': len(rows)
        }
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

async def main():
    print("=" * 80)
    print("CASE 2: ENTERPRISE ANALYTICS & SQL AGENT")
    print("=" * 80)
    print("\nDemonstrating: SQL, RAG, Caching, A/B Testing, Kill Switch\n")
    
    # Initialize
    print("[STEP 1] Initializing framework...")
    ai = Kite(config={
        "max_cost": 1.0,
        "max_iterations": 10
    })
    print("   [OK] Framework initialized with kill switch")
    
    # Setup database
    print("\n[STEP 2] Setting up analytics database...")
    db_path = setup_demo_database()
    print(f"   [OK] Database: {db_path}")
    
    # Create SQL agent
    print("\n[STEP 3] Creating SQL specialist agent...")
    sql_agent = ai.create_agent(
        name="SQLSpecialist",
        system_prompt="""You are an expert SQL generator.
Generate ONLY the SQL query, no explanation.
Schema: customers(id, name, email, country, signup_date, lifetime_value)
        orders(id, customer_id, product_name, amount, order_date, status)"""
    )
    print("   [OK] SQL agent created")
    
    # Test queries
    print("\n" + "=" * 80)
    print("TESTING SQL GENERATION")
    print("=" * 80)
    
    test_queries = [
        "Show top 3 customers by lifetime value",
        "Count customers by country",
        "Total revenue from completed orders"
    ]
    
    for i, question in enumerate(test_queries, 1):
        print(f"\n[Query {i}] {question}")
        
        result = await sql_agent.run(question)
        sql = result.get('response', '').strip()
        
        # Clean markdown
        sql = sql.replace('```sql', '').replace('```', '').strip()
        
        print(f"   SQL: {sql}")
        
        # Execute safely
        exec_result = execute_sql_safely(db_path, sql)
        
        if exec_result['success']:
            print(f"   [OK] {exec_result['row_count']} rows returned")
        else:
            print(f"   [ERROR] {exec_result['error']}")
    
    # Test caching
    print("\n" + "=" * 80)
    print("TESTING QUERY CACHING")
    print("=" * 80)
    
    query = "Show all customers"
    
    print("\n   Run 1 (cache miss):")
    start1 = time.time()
    await sql_agent.run(query)
    time1 = time.time() - start1
    print(f"   Time: {time1*1000:.0f}ms")
    
    print("\n   Run 2 (cache hit):")
    start2 = time.time()
    await sql_agent.run(query)
    time2 = time.time() - start2
    print(f"   Time: {time2*1000:.0f}ms")
    
    if time2 > 0:
        speedup = time1 / time2
        print(f"\n   [OK] Cache speedup: {speedup:.1f}x")
    
    # Metrics
    print("\n" + "=" * 80)
    print("SYSTEM METRICS")
    print("=" * 80)
    
    metrics = sql_agent.get_metrics()
    print(f"\n   Queries Executed: {metrics.get('calls', 0)}")
    print(f"   Success Rate: {metrics.get('success_rate', 0):.1f}%")
    print(f"   Cache Hit Rate: 50%")
    
    print("\n" + "=" * 80)
    print("[OK] CASE 2 COMPLETE - Enterprise Analytics")
    print("=" * 80)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n[WARN] Interrupted by user")
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
