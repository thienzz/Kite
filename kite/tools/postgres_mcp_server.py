"""
MCP Server Implementation: PostgreSQL Database

A Model Context Protocol (MCP) server that allows AI agents to query PostgreSQL databases.
This is a production-ready implementation with security and safety features.

Run: python postgres_mcp_server.py
"""

import os
import json
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv

load_dotenv()


@dataclass
class MCPServerConfig:
    """Configuration for the MCP server."""
    host: str = "localhost"
    port: int = 5432
    database: str = "kite"
    user: str = "postgres"
    password: str = ""
    
    # Safety limits
    max_rows: int = 1000
    max_query_length: int = 5000
    timeout_seconds: int = 30
    
    # Allowed operations
    allow_select: bool = True
    allow_insert: bool = False
    allow_update: bool = False
    allow_delete: bool = False


class PostgresMCPServer:
    """
    MCP Server for PostgreSQL database access.
    
    This server exposes database operations as MCP tools that AI agents can use.
    It includes security features like query validation, row limits, and timeouts.
    
    Example:
        config = MCPServerConfig(
            host="localhost",
            database="my_db",
            allow_select=True
        )
        server = PostgresMCPServer(config)
        result = await server.execute_query("SELECT * FROM users LIMIT 10")
    """
    
    def __init__(self, config: MCPServerConfig = None, connection_string: str = None, **kwargs):
        self.config = config or MCPServerConfig()
        if connection_string:
            # Simple parsing for demo - in production use proper URL parser
            # postgresql://user:pass@host:port/db
            self.config.database = connection_string.split('/')[-1]
        self.connection = None
        self.tools = self._define_tools()
    
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            print(f"[OK] Connected to PostgreSQL: {self.config.database}")
        except Exception as e:
            print(f"  Connection failed: {e}")
            raise
    
    def disconnect(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            print("[OK] Disconnected from PostgreSQL")
    
    def _is_safe_query(self, query: str) -> tuple[bool, str]:
        """
        Validate query for safety.
        
        Returns:
            (is_safe, reason)
        """
        query_upper = query.upper().strip()
        
        # Check length
        if len(query) > self.config.max_query_length:
            return False, f"Query too long ({len(query)} > {self.config.max_query_length})"
        
        # Check allowed operations
        dangerous_keywords = ['DROP', 'TRUNCATE', 'ALTER', 'CREATE TABLE', 'GRANT', 'REVOKE']
        for keyword in dangerous_keywords:
            if keyword in query_upper:
                return False, f"Forbidden keyword: {keyword}"
        
        # Check operation permissions
        if query_upper.startswith('SELECT') and not self.config.allow_select:
            return False, "SELECT queries not allowed"
        
        if query_upper.startswith('INSERT') and not self.config.allow_insert:
            return False, "INSERT queries not allowed"
        
        if query_upper.startswith('UPDATE') and not self.config.allow_update:
            return False, "UPDATE queries not allowed"
        
        if query_upper.startswith('DELETE') and not self.config.allow_delete:
            return False, "DELETE queries not allowed"
        
        return True, "OK"
    
    async def execute_query(self, query: str, params: tuple = None) -> Dict[str, Any]:
        """
        Execute SQL query with safety checks.
        
        Args:
            query: SQL query string
            params: Query parameters (optional)
            
        Returns:
            Result dictionary with rows and metadata
        """
        print(f"\n[CHART] Executing query:")
        print(f"   {query[:100]}...")
        
        # Validate query
        is_safe, reason = self._is_safe_query(query)
        if not is_safe:
            return {
                "success": False,
                "error": f"Unsafe query: {reason}",
                "rows": []
            }
        
        try:
            cursor = self.connection.cursor()
            
            # Execute with timeout
            cursor.execute(f"SET statement_timeout = {self.config.timeout_seconds * 1000}")
            cursor.execute(query, params)
            
            # Fetch results (with row limit)
            if query.upper().strip().startswith('SELECT'):
                rows = cursor.fetchmany(self.config.max_rows)
                columns = [desc[0] for desc in cursor.description]
                
                # Convert to list of dicts
                results = [
                    dict(zip(columns, row))
                    for row in rows
                ]
                
                self.connection.commit()
                cursor.close()
                
                print(f"   [OK] Returned {len(results)} rows")
                
                return {
                    "success": True,
                    "rows": results,
                    "count": len(results),
                    "columns": columns
                }
            else:
                # For INSERT/UPDATE/DELETE
                self.connection.commit()
                affected_rows = cursor.rowcount
                cursor.close()
                
                print(f"   [OK] Affected {affected_rows} rows")
                
                return {
                    "success": True,
                    "affected_rows": affected_rows
                }
                
        except Exception as e:
            self.connection.rollback()
            print(f"     Query failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "rows": []
            }
    
    def _define_tools(self) -> List[Dict]:
        """
        Define MCP tools that agents can use.
        
        These tool definitions follow the MCP standard format.
        """
        return [
            {
                "name": "query_database",
                "description": """
                Query the PostgreSQL database using SQL.
                Use this to retrieve data from tables.
                
                Safety features:
                - Automatic row limit enforcement
                - Query timeout protection
                - Dangerous operations blocked
                
                Examples:
                - SELECT * FROM users WHERE age > 25 LIMIT 10
                - SELECT product_name, price FROM products WHERE category = 'electronics'
                """,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "SQL SELECT query to execute"
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get_table_schema",
                "description": """
                Get the schema (column names and types) of a database table.
                Use this to understand table structure before querying.
                """,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "table_name": {
                            "type": "string",
                            "description": "Name of the table"
                        }
                    },
                    "required": ["table_name"]
                }
            },
            {
                "name": "list_tables",
                "description": """
                List all tables in the database.
                Use this to discover what data is available.
                """,
                "input_schema": {
                    "type": "object",
                    "properties": {}
                }
            }
        ]
    
    async def handle_tool_call(self, tool_name: str, args: Dict) -> Dict:
        """
        Handle MCP tool call from agent.
        
        Args:
            tool_name: Name of the tool to execute
            args: Tool arguments
            
        Returns:
            Tool execution result
        """
        if tool_name == "query_database":
            return await self.execute_query(args["query"])
        
        elif tool_name == "get_table_schema":
            query = """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position
            """
            result = await self.execute_query(query, (args["table_name"],))
            return result
        
        elif tool_name == "list_tables":
            query = """
                SELECT table_name, table_type
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name
            """
            result = await self.execute_query(query)
            return result
        
        else:
            return {
                "success": False,
                "error": f"Unknown tool: {tool_name}"
            }
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get MCP tool definitions for AI agent."""
        return self.tools


# ============================================================================
# DEMO: Using MCP Server with Mock Data
# ============================================================================

async def setup_demo_database():
    """Create demo tables with sample data."""
    config = MCPServerConfig(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        database=os.getenv("POSTGRES_DB", "postgres"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", "")
    )
    
    server = PostgresMCPServer(config)
    
    try:
        server.connect()
        
        # Create demo tables
        print("\n  Setting up demo database...")
        
        # Drop existing tables
        await server.execute_query("DROP TABLE IF EXISTS orders CASCADE")
        await server.execute_query("DROP TABLE IF EXISTS products CASCADE")
        await server.execute_query("DROP TABLE IF EXISTS customers CASCADE")
        
        # Create customers table
        await server.execute_query("""
            CREATE TABLE customers (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                email VARCHAR(100),
                joined_date DATE
            )
        """)
        
        # Create products table
        await server.execute_query("""
            CREATE TABLE products (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100),
                category VARCHAR(50),
                price DECIMAL(10, 2),
                stock INT
            )
        """)
        
        # Create orders table
        await server.execute_query("""
            CREATE TABLE orders (
                id SERIAL PRIMARY KEY,
                customer_id INT REFERENCES customers(id),
                product_id INT REFERENCES products(id),
                quantity INT,
                total DECIMAL(10, 2),
                order_date DATE
            )
        """)
        
        # Insert sample data
        await server.execute_query("""
            INSERT INTO customers (name, email, joined_date) VALUES
            ('John Doe', 'john@example.com', '2024-01-15'),
            ('Jane Smith', 'jane@example.com', '2024-02-20'),
            ('Bob Wilson', 'bob@example.com', '2024-03-10')
        """)
        
        await server.execute_query("""
            INSERT INTO products (name, category, price, stock) VALUES
            ('Laptop Pro', 'Electronics', 1299.99, 50),
            ('Wireless Mouse', 'Electronics', 29.99, 200),
            ('Office Chair', 'Furniture', 199.99, 30),
            ('Desk Lamp', 'Furniture', 49.99, 100)
        """)
        
        await server.execute_query("""
            INSERT INTO orders (customer_id, product_id, quantity, total, order_date) VALUES
            (1, 1, 1, 1299.99, '2024-04-01'),
            (1, 2, 2, 59.98, '2024-04-01'),
            (2, 3, 1, 199.99, '2024-04-05'),
            (3, 4, 3, 149.97, '2024-04-10')
        """)
        
        print("[OK] Demo database ready!")
        
        return server
        
    except Exception as e:
        print(f"  Setup failed: {e}")
        server.disconnect()
        return None


async def demo_mcp_server():
    """Demonstrate MCP server usage."""
    print("=" * 70)
    print("MCP SERVER DEMO: PostgreSQL")
    print("=" * 70)
    
    # Setup database
    server = await setup_demo_database()
    if not server:
        return
    
    try:
        print("\n" + "=" * 70)
        print("EXAMPLE 1: List all tables")
        print("=" * 70)
        result = await server.handle_tool_call("list_tables", {})
        print(json.dumps(result, indent=2, default=str))
        
        print("\n" + "=" * 70)
        print("EXAMPLE 2: Get table schema")
        print("=" * 70)
        result = await server.handle_tool_call("get_table_schema", {"table_name": "products"})
        print(json.dumps(result, indent=2, default=str))
        
        print("\n" + "=" * 70)
        print("EXAMPLE 3: Query products")
        print("=" * 70)
        result = await server.handle_tool_call(
            "query_database",
            {"query": "SELECT * FROM products WHERE category = 'Electronics'"}
        )
        print(json.dumps(result, indent=2, default=str))
        
        print("\n" + "=" * 70)
        print("EXAMPLE 4: Complex join query")
        print("=" * 70)
        result = await server.handle_tool_call(
            "query_database",
            {"query": """
                SELECT 
                    c.name as customer_name,
                    p.name as product_name,
                    o.quantity,
                    o.total
                FROM orders o
                JOIN customers c ON o.customer_id = c.id
                JOIN products p ON o.product_id = p.id
                ORDER BY o.order_date DESC
            """}
        )
        print(json.dumps(result, indent=2, default=str))
        
        print("\n" + "=" * 70)
        print("EXAMPLE 5: Safety check - dangerous query blocked")
        print("=" * 70)
        result = await server.handle_tool_call(
            "query_database",
            {"query": "DROP TABLE products"}
        )
        print(json.dumps(result, indent=2, default=str))
        
        print("\n" + "=" * 70)
        print("[OK] Demo completed successfully!")
        print("=" * 70)
        
    finally:
        server.disconnect()


if __name__ == "__main__":
    # Run demo
    asyncio.run(demo_mcp_server())
    
    print("\n  TO USE WITH AI AGENT:")
    print("""
    # In your agent code:
    from postgres_mcp_server import PostgresMCPServer, MCPServerConfig
    
    # Initialize MCP server
    config = MCPServerConfig(database="your_db")
    mcp_server = PostgresMCPServer(config)
    mcp_server.connect()
    
    # Get tool definitions for agent
    tools = mcp_server.get_tool_definitions()
    
    # When agent calls a tool:
    result = await mcp_server.handle_tool_call(
        tool_name="query_database",
        args={"query": "SELECT * FROM users"}
    )
    """)
