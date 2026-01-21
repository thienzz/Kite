"""
SQL Generator SLM (Small Language Model)
Based on Chapter 1.3: Sub-Agents (The Specialists & SLMs)

A specialized small model for SQL generation.
"""

import os
import json
import time
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import re
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# CONFIGURATION
# ============================================================================

class SQLDialect(Enum):
    """Supported SQL dialects."""
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLITE = "sqlite"


@dataclass
class SLMConfig:
    """Configuration for SQL Generator SLM."""
    model_name: str = "llama-3.1-8b-instruct"
    dialect: SQLDialect = SQLDialect.POSTGRESQL
    max_tokens: int = 500
    temperature: float = 0.1
    
    # Safety
    allow_write_operations: bool = False
    max_joins: int = 5
    timeout_seconds: int = 5


# ============================================================================
# SQL GENERATOR SLM
# ============================================================================

class SQLGeneratorSLM:
    """
    Small Language Model specialized for SQL generation.
    
    In production, this would use a fine-tuned model.
    This implementation provides a base for pattern-based SQL generation.
    """
    
    def __init__(self, config: SLMConfig = None):
        self.config = config or SLMConfig()
        
        # SQL keywords for validation
        self.write_keywords = ["INSERT", "UPDATE", "DELETE", "DROP", "TRUNCATE", "ALTER"]
        
        print(f"[OK] SQL Generator SLM initialized")
        print(f"  Model: {self.config.model_name}")
        print(f"  Dialect: {self.config.dialect.value}")
    
    def _parse_intent(self, prompt: str) -> Dict:
        """Parse user intent from natural language (pattern-based fallback)."""
        prompt_lower = prompt.lower()
        
        intent = {
            "operation": "select",
            "tables": [],
            "columns": ["*"],
            "conditions": [],
            "joins": [],
            "group_by": None,
            "order_by": None,
            "limit": None
        }
        
        # Extract table names
        table_patterns = [r"from (\w+)", r"in (\w+)", r"(\w+) table"]
        for pattern in table_patterns:
            matches = re.findall(pattern, prompt_lower)
            if matches:
                intent["tables"].extend(matches)
        
        if any(word in prompt_lower for word in ["count", "how many"]):
            intent["operation"] = "count"
            intent["columns"] = ["COUNT(*)"]
        
        return intent
    
    def _validate_sql(self, sql: str) -> tuple[bool, str]:
        """Validate generated SQL for safety."""
        sql_upper = sql.upper()
        
        if not self.config.allow_write_operations:
            for keyword in self.write_keywords:
                if keyword in sql_upper:
                    return False, f"Write operation not allowed: {keyword}"
        
        if sql_upper.count("JOIN") > self.config.max_joins:
            return False, f"Too many JOINs"
        
        dangerous_patterns = [r";.*DROP", r";.*DELETE", r"--.*", r"/\*.*\*/"]
        for pattern in dangerous_patterns:
            if re.search(pattern, sql_upper):
                return False, f"Potentially dangerous pattern detected"
        
        return True, "OK"
    
    def generate(
        self,
        prompt: str,
        schema: Optional[Dict[str, List[str]]] = None,
        dialect: Optional[SQLDialect] = None
    ) -> Dict:
        """Generate SQL from natural language."""
        start_time = time.time()
        target_dialect = dialect or self.config.dialect
        
        # In production, this would call the SLM
        # Here we simulate with intent parsing
        intent = self._parse_intent(prompt)
        sql = self._generate_from_intent(intent, schema, target_dialect)
        
        is_valid, error = self._validate_sql(sql)
        elapsed = (time.time() - start_time) * 1000
        
        if not is_valid:
            return {"success": False, "error": error, "latency_ms": elapsed}
        
        return {
            "success": True,
            "sql": sql,
            "intent": intent,
            "dialect": target_dialect.value,
            "latency_ms": elapsed,
            "model": self.config.model_name
        }
    
    def _generate_from_intent(self, intent: Dict, schema: Optional[Dict[str, List[str]]], dialect: SQLDialect) -> str:
        """Generate SQL string from intent (simulate SLM)."""
        table = intent["tables"][0] if intent["tables"] else "unknown_table"
        cols = ", ".join(intent["columns"])
        
        if intent["operation"] == "count":
            return f"SELECT COUNT(*) FROM {table}"
        return f"SELECT {cols} FROM {table}"
    
    def explain(self, sql: str) -> str:
        """Explain what a SQL query does in natural language."""
        return f"This query performs an operation on the database: {sql}"
    
    def get_stats(self) -> Dict:
        """Get generator statistics."""
        return {
            "model": self.config.model_name,
            "dialect": self.config.dialect.value
        }
