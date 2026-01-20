"""
Stripe MCP Server Implementation
Based on Chapter 4: Model Context Protocol

Lessons from $15K Refund Loop (Chapter 1):
- Idempotency keys MANDATORY
- Circuit breaker REQUIRED
- Confirmation prompts for all write operations
- Detailed audit logging
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from dotenv import load_dotenv

load_dotenv(".kite.env")


# ============================================================================
# SAFETY COMPONENTS
# ============================================================================

class RefundStatus(Enum):
    """Refund processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DUPLICATE = "duplicate"


@dataclass
class IdempotencyKey:
    """Idempotency key for preventing duplicates."""
    key: str
    created_at: datetime
    result: Optional[Dict] = None


class IdempotencyManager:
    """Prevents duplicate operations using idempotency keys."""
    
    def __init__(self, ttl_seconds: int = 86400):
        self.cache: Dict[str, IdempotencyKey] = {}
        self.ttl_seconds = ttl_seconds
    
    def generate_key(self, operation: str, **params) -> str:
        data = json.dumps({"operation": operation, **params}, sort_keys=True)
        key = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"idem_{key}"
    
    def get_cached(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            idem = self.cache[key]
            if (datetime.now() - idem.created_at).seconds < self.ttl_seconds:
                return idem.result
            del self.cache[key]
        return None
    
    def store(self, key: str, result: Dict):
        self.cache[key] = IdempotencyKey(key=key, created_at=datetime.now(), result=result)


class RefundCircuitBreaker:
    """Circuit breaker to stop repeated refund failures."""
    
    def __init__(self, failure_threshold: int = 3, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"
    
    def can_attempt(self) -> tuple[bool, str]:
        if self.state == "closed": return True, "OK"
        if self.state == "open":
            if self.last_failure_time and (datetime.now() - self.last_failure_time).seconds >= self.timeout_seconds:
                self.state = "half_open"
                return True, "Testing"
            return False, "Circuit open"
        return True, "Testing"
    
    def record_success(self):
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class StripeConfig:
    """Configuration for Stripe MCP server."""
    api_key: str = ""
    max_refund_amount: float = 1000.00
    require_confirmation: bool = True
    max_refunds_per_hour: int = 10
    rate_limit_per_minute: int = 30
    enable_read: bool = True
    enable_refund: bool = False


# ============================================================================
# STRIPE MCP SERVER
# ============================================================================

class StripeMCPServer:
    """
    MCP Server for Stripe integration with safety defaults.
    """
    
    def __init__(self, config: StripeConfig = None, client = None):
        self.config = config or StripeConfig()
        self.stripe = client # Should be a real stripe client in production
        
        # SAFETY COMPONENTS
        self.idempotency = IdempotencyManager()
        self.circuit_breaker = RefundCircuitBreaker()
        
        # Rate limiting
        self.request_count = 0
        self.refund_count_hour = 0
        self.window_start = datetime.now()
        self.hour_start = datetime.now()
        
        # Audit log
        self.audit_log: List[Dict] = []
        
        print(f"[OK] Stripe MCP Server initialized (SAFETY MODE: EXTREME)")
    
    def _check_rate_limit(self) -> bool:
        now = datetime.now()
        if (now - self.window_start).seconds >= 60:
            self.request_count = 0
            self.window_start = now
        if self.request_count >= self.config.rate_limit_per_minute:
            return False
        self.request_count += 1
        return True
    
    def _check_refund_limit(self) -> bool:
        now = datetime.now()
        if (now - self.hour_start).seconds >= 3600:
            self.refund_count_hour = 0
            self.hour_start = now
        return self.refund_count_hour < self.config.max_refunds_per_hour
    
    def _log_audit(self, operation: str, params: Dict, result: Dict):
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "params": params,
            "result": result
        })
    
    def get_payment(self, charge_id: str) -> Dict[str, Any]:
        """Get payment details (READ-ONLY)."""
        if not self.config.enable_read or not self._check_rate_limit():
            return {"success": False, "error": "Access denied or rate limited"}
        
        try:
            # Placeholder for real stripe call
            if not self.stripe: return {"success": False, "error": "Stripe client not configured"}
            charge = self.stripe.charges_retrieve(charge_id)
            return {"success": True, "charge_id": charge_id, "amount": charge["amount"]/100}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def process_refund(self, charge_id: str, amount: Optional[float] = None) -> Dict[str, Any]:
        """Process refund with safety checks."""
        if not self.config.enable_refund: return {"success": False, "error": "Refunds disabled"}
        if not self._check_rate_limit() or not self._check_refund_limit():
            return {"success": False, "error": "Limit exceeded"}
        
        idem_key = self.idempotency.generate_key("refund", charge_id=charge_id, amount=amount)
        cached = self.idempotency.get_cached(idem_key)
        if cached: return {**cached, "duplicate_prevented": True}
        
        can_attempt, _ = self.circuit_breaker.can_attempt()
        if not can_attempt: return {"success": False, "error": "Circuit breaker open"}
        
        if amount and amount > self.config.max_refund_amount and self.config.require_confirmation:
            return {"success": False, "error": "Approval required", "requires_approval": True}
        
        try:
            if not self.stripe: raise Exception("Stripe client not configured")
            refund = self.stripe.refunds_create(charge=charge_id, amount=int(amount*100) if amount else None)
            self.circuit_breaker.record_success()
            self.refund_count_hour += 1
            result = {"success": True, "refund_id": refund["id"], "charge_id": charge_id}
            self.idempotency.store(idem_key, result)
            self._log_audit("process_refund", {"charge_id": charge_id}, result)
            return result
        except Exception as e:
            self.circuit_breaker.record_failure()
            return {"success": False, "error": str(e)}

    def get_tool_definitions(self) -> List[Dict]:
        """Get MCP tool definitions."""
        return [
            {"name": "stripe_get_payment", "description": "Get payment details (READ)"},
            {"name": "stripe_process_refund", "description": "Process refund (WRITE)"}
        ]
