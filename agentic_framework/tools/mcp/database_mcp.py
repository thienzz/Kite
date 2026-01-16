"""
MCP Database Connectors
Provides Model Context Protocol (MCP) inspired tools for database interaction.
"""

import os
import json
from typing import Dict, List, Any, Optional
import sqlite3
import psycopg2
import mysql.connector
import redis
import asyncio
from functools import wraps

# New Production Connectors
try:
    from neo4j import GraphDatabase
except ImportError:
    GraphDatabase = None
try:
    from cassandra.cluster import Cluster
    from cassandra.auth import PlainTextAuthProvider
except Exception:
    Cluster = None
try:
    import boto3
except ImportError:
    boto3 = None
try:
    from google.cloud import firestore
except ImportError:
    firestore = None

def safe_execute(timeout_seconds: int = 30):
    """Decorator for safe tool execution with timeouts and error handling."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                # Handle both sync and async functions
                if asyncio.iscoroutinefunction(func):
                    return await asyncio.wait_for(func(*args, **kwargs), timeout=timeout_seconds)
                else:
                    # For sync functions in an async environment
                    loop = asyncio.get_event_loop()
                    return await asyncio.wait_for(
                        loop.run_in_executor(None, lambda: func(*args, **kwargs)),
                        timeout=timeout_seconds
                    )
            except asyncio.TimeoutError:
                return {"error": f"Database operation timed out after {timeout_seconds}s"}
            except Exception as e:
                return {"error": f"Database error: {str(e)}"}
        return wrapper
    return decorator

class DatabaseMCP:
    """
    MCP-style interface for various databases.
    Implements tools for query and schema discovery.
    """
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.conns = {}

    def _get_pg_conn(self):
        if 'postgres' not in self.conns:
            self.conns['postgres'] = psycopg2.connect(
                dsn=self.config.get('postgres_dsn', os.getenv('POSTGRES_DSN'))
            )
        return self.conns['postgres']

    def _get_mysql_conn(self):
        if 'mysql' not in self.conns:
            self.conns['mysql'] = mysql.connector.connect(
                host=self.config.get('mysql_host', os.getenv('MYSQL_HOST')),
                user=self.config.get('mysql_user', os.getenv('MYSQL_USER')),
                password=self.config.get('mysql_password', os.getenv('MYSQL_PASSWORD')),
                database=self.config.get('mysql_db', os.getenv('MYSQL_DB'))
            )
        return self.conns['mysql']

    def _get_sqlite_conn(self):
        if 'sqlite' not in self.conns:
            self.conns['sqlite'] = sqlite3.connect(
                self.config.get('sqlite_db', os.getenv('SQLITE_DB', 'agentic_ai.db')),
                check_same_thread=False
            )
        return self.conns['sqlite']

    def _get_redis_conn(self):
        if 'redis' not in self.conns:
            self.conns['redis'] = redis.Redis(
                host=self.config.get('redis_host', os.getenv('REDIS_HOST', 'localhost')),
                port=self.config.get('redis_port', os.getenv('REDIS_PORT', 6379)),
                db=self.config.get('redis_db', 0),
                decode_responses=True,
                socket_timeout=5,
                retry_on_timeout=True
            )
        return self.conns['redis']

    def _get_neo4j_conn(self):
        if 'neo4j' not in self.conns:
            if GraphDatabase is None: raise ImportError("neo4j not installed")
            self.conns['neo4j'] = GraphDatabase.driver(
                self.config.get('neo4j_uri', os.getenv('NEO4J_URI')),
                auth=(self.config.get('neo4j_user', os.getenv('NEO4J_USER')), 
                      self.config.get('neo4j_password', os.getenv('NEO4J_PASSWORD')))
            )
        return self.conns['neo4j']

    def _get_cassandra_conn(self):
        if 'cassandra' not in self.conns:
            if Cluster is None: raise ImportError("cassandra-driver not installed")
            auth_provider = PlainTextAuthProvider(
                username=self.config.get('cassandra_user', os.getenv('CASSANDRA_USER')),
                password=self.config.get('cassandra_password', os.getenv('CASSANDRA_PASSWORD'))
            )
            self.conns['cassandra'] = Cluster(
                [self.config.get('cassandra_host', os.getenv('CASSANDRA_HOST', 'localhost'))],
                auth_provider=auth_provider
            ).connect()
        return self.conns['cassandra']

    def _get_dynamodb_resource(self):
        if 'dynamodb' not in self.conns:
            if boto3 is None: raise ImportError("boto3 not installed")
            self.conns['dynamodb'] = boto3.resource(
                'dynamodb',
                region_name=self.config.get('aws_region', os.getenv('AWS_REGION', 'us-east-1'))
            )
        return self.conns['dynamodb']

    def _get_firestore_client(self):
        if 'firestore' not in self.conns:
            if firestore is None: raise ImportError("google-cloud-firestore not installed")
            self.conns['firestore'] = firestore.Client()
        return self.conns['firestore']


    @safe_execute(timeout_seconds=30)
    def query_postgres(self, sql: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a PostgreSQL query and return results."""
        conn = self._get_pg_conn()
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]

    @safe_execute(timeout_seconds=30)
    def query_sqlite(self, sql: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a SQLite query and return results."""
        conn = self._get_sqlite_conn()
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(sql, params or ())
        return [dict(row) for row in cur.fetchall()]

    @safe_execute(timeout_seconds=30)
    def query_mysql(self, sql: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a MySQL query and return results."""
        conn = self._get_mysql_conn()
        cur = conn.cursor(dictionary=True)
        cur.execute(sql, params)
        return cur.fetchall()

    @safe_execute(timeout_seconds=30)
    async def query_mongo(self, collection: str, query: Dict, limit: int = 10) -> List[Dict]:
        """Query a MongoDB collection."""
        from motor.motor_asyncio import AsyncIOMotorClient
        client = AsyncIOMotorClient(self.config.get('mongo_uri', os.getenv('MONGO_URI')))
        db_name = self.config.get('mongo_db', os.getenv('MONGO_DB'))
        results = await client[db_name][collection].find(query).to_list(length=limit)
        return results

    @safe_execute(timeout_seconds=30)
    def query_neo4j(self, cypher: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a Cypher query on Neo4j."""
        driver = self._get_neo4j_conn()
        with driver.session() as session:
            result = session.run(cypher, params or {})
            return [dict(record) for record in result]

    @safe_execute(timeout_seconds=30)
    def query_cassandra(self, cql: str, params: Optional[tuple] = None) -> List[Dict]:
        """Execute a CQL query on Cassandra."""
        session = self._get_cassandra_conn()
        rows = session.execute(cql, params or ())
        return [dict(row._asdict()) for row in rows]

    @safe_execute(timeout_seconds=30)
    def query_dynamodb(self, table_name: str, key: Dict) -> Dict:
        """Get an item from DynamoDB."""
        table = self._get_dynamodb_resource().Table(table_name)
        response = table.get_item(Key=key)
        return response.get('Item', {})

    @safe_execute(timeout_seconds=30)
    def query_firestore(self, collection: str, document: str) -> Dict:
        """Get a document from Firestore."""
        client = self._get_firestore_client()
        doc_ref = client.collection(collection).document(document)
        doc = doc_ref.get()
        return doc.to_dict() if doc.exists else {}

    def redis_get(self, key: str) -> Optional[str]:
        """Get value from Redis."""
        return self._get_redis_conn().get(key)

    def redis_set(self, key: str, value: str, ex: int = 3600):
        """Set value in Redis with TTL."""
        self._get_redis_conn().set(key, value, ex=ex)

    def get_tools(self) -> List[Dict]:
        """Expose methods as AgenticAI tools."""
        return [
            {
                "name": "query_postgres",
                "description": "Execute Read-only SQL queries on PostgreSQL",
                "parameters": {"sql": "string"},
                "func": self.query_postgres
            },
            {
                "name": "query_mysql",
                "description": "Execute Read-only SQL queries on MySQL",
                "parameters": {"sql": "string"},
                "func": self.query_mysql
            },
            {
                "name": "redis_get",
                "description": "Retrieve data from Redis operational cache",
                "parameters": {"key": "string"},
                "func": self.redis_get
            },
            {
                "name": "query_sqlite",
                "description": "Execute Read-only SQL queries on local SQLite database",
                "parameters": {"sql": "string"},
                "func": self.query_sqlite
            },
            {
                "name": "query_neo4j",
                "description": "Execute Cypher queries on Neo4j graph database",
                "parameters": {"cypher": "string"},
                "func": self.query_neo4j
            },
            {
                "name": "query_cassandra",
                "description": "Execute CQL queries on Cassandra database",
                "parameters": {"cql": "string"},
                "func": self.query_cassandra
            },
            {
                "name": "query_dynamodb",
                "description": "Get item from DynamoDB table",
                "parameters": {"table_name": "string", "key": "object"},
                "func": self.query_dynamodb
            },
            {
                "name": "query_firestore",
                "description": "Get document from Firestore collection",
                "parameters": {"collection": "string", "document": "string"},
                "func": self.query_firestore
            }
        ]
