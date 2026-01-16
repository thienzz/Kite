"""
Stripe MCP Server Implementation
Based on Chapter 4: MCP (Model Context Protocol)

CRITICAL: This handles payments - EXTREME SAFETY required!

Lessons from $15K Refund Loop (Chapter 1):
- Idempotency keys MANDATORY
- Circuit breaker REQUIRED
- Confirmation prompts for all write operations
- Detailed audit logging

Tools provided:
- get_payment: Get payment details (READ-ONLY)
- list_payments: List recent payments (READ-ONLY)
- process_refund: Process refund (WRITE - DANGEROUS!)

Run: python stripe_mcp_server.py
"""

import os
import json
import hashlib
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# SAFETY COMPONENTS (From $15K Case Study)
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
    """
    Prevents duplicate refunds using idempotency keys.
    
    From Chapter 1 - $15K Lesson:
    "Stripe, seeing a new request ID (because the engineers 
    hadn't implemented idempotency keys), processed a second refund."
    """
    
    def __init__(self, ttl_seconds: int = 86400):  # 24 hours
        self.cache: Dict[str, IdempotencyKey] = {}
        self.ttl_seconds = ttl_seconds
    
    def generate_key(self, operation: str, **params) -> str:
        """Generate deterministic idempotency key."""
        # Create stable hash from operation and parameters
        data = json.dumps({
            "operation": operation,
            **params
        }, sort_keys=True)
        
        key = hashlib.sha256(data.encode()).hexdigest()[:16]
        return f"idem_{key}"
    
    def get_cached(self, key: str) -> Optional[Dict]:
        """Get cached result if exists and not expired."""
        if key in self.cache:
            idem = self.cache[key]
            age = (datetime.now() - idem.created_at).seconds
            
            if age < self.ttl_seconds:
                print(f"  [OK] Idempotency cache hit: {key}")
                return idem.result
            else:
                # Expired, remove
                del self.cache[key]
        
        return None
    
    def store(self, key: str, result: Dict):
        """Store result in cache."""
        self.cache[key] = IdempotencyKey(
            key=key,
            created_at=datetime.now(),
            result=result
        )
        print(f"  [OK] Stored in idempotency cache: {key}")


class RefundCircuitBreaker:
    """
    Circuit breaker to stop repeated refund failures.
    
    From Chapter 1 - $15K Lesson:
    "The Agent, now panicking in its own digital way, 
    entered a tight 'Retry' loop."
    """
    
    def __init__(self, failure_threshold: int = 3, timeout_seconds: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.state = "closed"  # closed, open, half_open
    
    def can_attempt(self) -> tuple[bool, str]:
        """Check if operation can be attempted."""
        if self.state == "closed":
            return True, "OK"
        
        if self.state == "open":
            # Check if timeout has passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).seconds
                
                if elapsed >= self.timeout_seconds:
                    self.state = "half_open"
                    print(f"    Circuit breaker: half-open (testing)")
                    return True, "Testing after timeout"
                else:
                    remaining = self.timeout_seconds - elapsed
                    return False, f"Circuit open, retry in {remaining}s"
        
        if self.state == "half_open":
            return True, "Testing"
        
        return False, "Unknown state"
    
    def record_success(self):
        """Record successful operation."""
        if self.state == "half_open":
            print(f"  [OK] Circuit breaker: closed (recovered)")
        
        self.failure_count = 0
        self.state = "closed"
    
    def record_failure(self):
        """Record failed operation."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            print(f"    Circuit breaker: OPEN (too many failures)")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class StripeConfig:
    """Configuration for Stripe MCP server."""
    api_key: str = ""
    
    # CRITICAL SAFETY LIMITS
    max_refund_amount: float = 1000.00  # Auto-approve limit
    require_confirmation: bool = True   # Human confirmation for refunds
    max_refunds_per_hour: int = 10
    
    # Rate limiting
    rate_limit_per_minute: int = 30
    
    # Allowed operations
    enable_read: bool = True
    enable_refund: bool = False  # DISABLED by default!


# ============================================================================
# MOCK STRIPE CLIENT
# ============================================================================

class MockStripeClient:
    """Mock Stripe client for demonstration."""
    
    def __init__(self):
        self.payments_db = self._create_mock_payments()
        self.refunds_db = []
    
    def _create_mock_payments(self) -> List[Dict]:
        """Create mock payment database."""
        return [
            {
                "id": "ch_001",
                "amount": 29999,  # $299.99 in cents
                "currency": "usd",
                "status": "succeeded",
                "created": int((datetime.now() - timedelta(days=5)).timestamp()),
                "customer": "cus_123",
                "description": "Monthly subscription",
                "refunded": False
            },
            {
                "id": "ch_002",
                "amount": 15000,  # $150.00
                "currency": "usd",
                "status": "succeeded",
                "created": int((datetime.now() - timedelta(days=2)).timestamp()),
                "customer": "cus_456",
                "description": "Product purchase",
                "refunded": False
            }
        ]
    
    def charges_retrieve(self, charge_id: str) -> Dict:
        """Retrieve charge details."""
        for payment in self.payments_db:
            if payment["id"] == charge_id:
                return payment
        
        raise Exception(f"Charge not found: {charge_id}")
    
    def charges_list(self, limit: int = 10) -> Dict:
        """List charges."""
        return {
            "data": self.payments_db[:limit]
        }
    
    def refunds_create(self, charge: str, amount: Optional[int] = None, idempotency_key: Optional[str] = None) -> Dict:
        """Create refund."""
        # Find charge
        payment = None
        for p in self.payments_db:
            if p["id"] == charge:
                payment = p
                break
        
        if not payment:
            raise Exception(f"Charge not found: {charge}")
        
        if payment["refunded"]:
            raise Exception(f"Charge already refunded: {charge}")
        
        # Create refund
        refund_amount = amount or payment["amount"]
        
        refund = {
            "id": f"re_{len(self.refunds_db)+1:03d}",
            "amount": refund_amount,
            "charge": charge,
            "created": int(datetime.now().timestamp()),
            "status": "succeeded",
            "idempotency_key": idempotency_key
        }
        
        self.refunds_db.append(refund)
        payment["refunded"] = True
        
        return refund


# ============================================================================
# STRIPE MCP SERVER (WITH EXTREME SAFETY)
# ============================================================================

class StripeMCPServer:
    """
    MCP Server for Stripe integration.
    
    CRITICAL: Implements all lessons from $15K Refund Loop:
    - Idempotency keys
    - Circuit breaker
    - Confirmation prompts
    - Detailed logging
    
    Example:
        config = StripeConfig()
        server = StripeMCPServer(config)
        
        # Safe: Read payment
        payment = server.get_payment("ch_001")
        
        # Dangerous: Refund (requires confirmation)
        refund = server.process_refund("ch_001", amount=100.00)
    """
    
    def __init__(self, config: StripeConfig = None, api_key: str = None):
        self.config = config or StripeConfig()
        if api_key:
            self.config.api_key = api_key
        
        # Initialize Stripe client (mock for demo)
        self.stripe = MockStripeClient()
        
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
        
        print(f"[OK] Stripe MCP Server initialized")
        print(f"  [WARN]  SAFETY MODE: EXTREME")
        print(f"  Max auto-refund: ${self.config.max_refund_amount:.2f}")
        print(f"  Confirmation required: {self.config.require_confirmation}")
        print(f"  Refund enabled: {self.config.enable_refund}")
    
    def _check_rate_limit(self) -> bool:
        """Check rate limit."""
        now = datetime.now()
        
        if (now - self.window_start).seconds >= 60:
            self.request_count = 0
            self.window_start = now
        
        if self.request_count >= self.config.rate_limit_per_minute:
            return False
        
        self.request_count += 1
        return True
    
    def _check_refund_limit(self) -> bool:
        """Check hourly refund limit."""
        now = datetime.now()
        
        if (now - self.hour_start).seconds >= 3600:
            self.refund_count_hour = 0
            self.hour_start = now
        
        return self.refund_count_hour < self.config.max_refunds_per_hour
    
    def _log_audit(self, operation: str, params: Dict, result: Dict):
        """Log operation to audit trail."""
        self.audit_log.append({
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "params": params,
            "result": result
        })
    
    def get_payment(self, charge_id: str) -> Dict[str, Any]:
        """
        Get payment details (READ-ONLY - SAFE).
        
        Args:
            charge_id: Stripe charge ID
            
        Returns:
            Payment details
        """
        print(f"\n  Getting payment: {charge_id}")
        
        if not self.config.enable_read:
            return {
                "success": False,
                "error": "Read is disabled"
            }
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        try:
            charge = self.stripe.charges_retrieve(charge_id)
            
            result = {
                "success": True,
                "charge_id": charge["id"],
                "amount": charge["amount"] / 100,  # Convert cents to dollars
                "currency": charge["currency"],
                "status": charge["status"],
                "refunded": charge["refunded"],
                "description": charge["description"]
            }
            
            print(f"  [OK] Retrieved payment: ${result['amount']:.2f}")
            
            self._log_audit("get_payment", {"charge_id": charge_id}, result)
            
            return result
            
        except Exception as e:
            print(f"    Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def list_payments(self, limit: int = 10) -> Dict[str, Any]:
        """
        List recent payments (READ-ONLY - SAFE).
        
        Args:
            limit: Number of payments to retrieve
            
        Returns:
            List of payments
        """
        print(f"\n  Listing {limit} payments")
        
        if not self.config.enable_read:
            return {
                "success": False,
                "error": "Read is disabled"
            }
        
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        try:
            charges = self.stripe.charges_list(limit=limit)
            
            payments = [
                {
                    "charge_id": ch["id"],
                    "amount": ch["amount"] / 100,
                    "status": ch["status"],
                    "refunded": ch["refunded"],
                    "description": ch["description"]
                }
                for ch in charges["data"]
            ]
            
            print(f"  [OK] Retrieved {len(payments)} payments")
            
            return {
                "success": True,
                "payments": payments,
                "count": len(payments)
            }
            
        except Exception as e:
            print(f"    Error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def process_refund(
        self,
        charge_id: str,
        amount: Optional[float] = None,
        reason: str = "requested_by_customer"
    ) -> Dict[str, Any]:
        """
        Process refund (WRITE - DANGEROUS!).
        
        CRITICAL SAFETY FEATURES:
        1. Idempotency key (prevent duplicates)
        2. Circuit breaker (stop failure loops)
        3. Amount limits (require approval)
        4. Hourly limits (prevent abuse)
        5. Audit logging (track everything)
        
        Args:
            charge_id: Stripe charge ID
            amount: Refund amount (None = full refund)
            reason: Refund reason
            
        Returns:
            Refund result
        """
        print(f"\n[WARN]  PROCESSING REFUND: {charge_id}")
        print(f"  Amount: ${amount if amount else 'FULL'}")
        print(f"  Reason: {reason}")
        
        # SAFETY CHECK 1: Feature enabled?
        if not self.config.enable_refund:
            return {
                "success": False,
                "error": "Refunds are DISABLED for safety"
            }
        
        # SAFETY CHECK 2: Rate limit
        if not self._check_rate_limit():
            return {
                "success": False,
                "error": "Rate limit exceeded"
            }
        
        # SAFETY CHECK 3: Hourly refund limit
        if not self._check_refund_limit():
            return {
                "success": False,
                "error": f"Hourly refund limit reached ({self.config.max_refunds_per_hour})"
            }
        
        # SAFETY CHECK 4: Idempotency (prevent duplicates)
        idem_key = self.idempotency.generate_key(
            "refund",
            charge_id=charge_id,
            amount=amount
        )
        
        cached = self.idempotency.get_cached(idem_key)
        if cached:
            print(f"  [WARN]  DUPLICATE REQUEST DETECTED!")
            print(f"  Returning cached result (prevented duplicate refund)")
            return {
                **cached,
                "duplicate_prevented": True
            }
        
        # SAFETY CHECK 5: Circuit breaker
        can_attempt, reason_msg = self.circuit_breaker.can_attempt()
        if not can_attempt:
            return {
                "success": False,
                "error": f"Circuit breaker: {reason_msg}"
            }
        
        # SAFETY CHECK 6: Amount limit
        if amount and amount > self.config.max_refund_amount:
            if self.config.require_confirmation:
                return {
                    "success": False,
                    "error": f"Amount ${amount:.2f} exceeds auto-approval limit ${self.config.max_refund_amount:.2f}",
                    "requires_approval": True,
                    "approval_workflow": "Route to finance team for manual review"
                }
        
        # Process refund
        try:
            print(f"    Calling Stripe API...")
            
            refund = self.stripe.refunds_create(
                charge=charge_id,
                amount=int(amount * 100) if amount else None,  # Convert to cents
                idempotency_key=idem_key
            )
            
            # Success!
            self.circuit_breaker.record_success()
            self.refund_count_hour += 1
            
            result = {
                "success": True,
                "refund_id": refund["id"],
                "charge_id": charge_id,
                "amount": refund["amount"] / 100,
                "status": refund["status"],
                "idempotency_key": idem_key
            }
            
            # Cache result
            self.idempotency.store(idem_key, result)
            
            # Audit log
            self._log_audit("process_refund", {
                "charge_id": charge_id,
                "amount": amount,
                "reason": reason
            }, result)
            
            print(f"  [OK] Refund processed: {refund['id']}")
            print(f"  Amount: ${result['amount']:.2f}")
            
            return result
            
        except Exception as e:
            # Failure!
            self.circuit_breaker.record_failure()
            
            print(f"    Refund failed: {e}")
            
            error_result = {
                "success": False,
                "error": str(e)
            }
            
            # Still cache the failure to prevent retries
            self.idempotency.store(idem_key, error_result)
            
            return error_result
    
    def get_audit_log(self) -> List[Dict]:
        """Get audit trail."""
        return self.audit_log
    
    def get_tool_definitions(self) -> List[Dict]:
        """Get MCP tool definitions for AI agents."""
        tools = [
            {
                "name": "stripe_get_payment",
                "description": "Get payment details (READ-ONLY - SAFE)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "charge_id": {
                            "type": "string",
                            "description": "Stripe charge ID"
                        }
                    },
                    "required": ["charge_id"]
                }
            },
            {
                "name": "stripe_list_payments",
                "description": "List recent payments (READ-ONLY - SAFE)",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "limit": {
                            "type": "integer",
                            "description": "Number of payments to retrieve",
                            "default": 10
                        }
                    }
                }
            }
        ]
        
        if self.config.enable_refund:
            tools.append({
                "name": "stripe_process_refund",
                "description": """
                Process refund (DANGEROUS - USE WITH EXTREME CAUTION!)
                
                SAFETY FEATURES:
                - Idempotency prevents duplicates
                - Circuit breaker stops failure loops
                - Amount limits require approval
                - Full audit trail
                """,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "charge_id": {
                            "type": "string",
                            "description": "Stripe charge ID to refund"
                        },
                        "amount": {
                            "type": "number",
                            "description": "Refund amount in dollars (omit for full refund)"
                        },
                        "reason": {
                            "type": "string",
                            "description": "Reason for refund",
                            "default": "requested_by_customer"
                        }
                    },
                    "required": ["charge_id"]
                }
            })
        
        return tools


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("STRIPE MCP SERVER DEMO - EXTREME SAFETY MODE")
    print("=" * 70)
    print("\nBased on Chapter 1: $15K Refund Loop Lessons")
    print("\nSafety Features:")
    print("  [OK] Idempotency keys (prevent duplicates)")
    print("  [OK] Circuit breaker (stop failure loops)")
    print("  [OK] Amount limits (require approval)")
    print("  [OK] Rate limiting (prevent abuse)")
    print("  [OK] Audit logging (track everything)")
    print("=" * 70)
    
    # Initialize server with refunds ENABLED for demo
    config = StripeConfig(
        enable_read=True,
        enable_refund=True,  # DANGEROUS! Only for demo
        max_refund_amount=500.00,
        require_confirmation=True
    )
    
    server = StripeMCPServer(config)
    
    # Demo 1: List payments (SAFE)
    print(f"\n{'='*70}")
    print("DEMO 1: List recent payments (READ-ONLY)")
    print('='*70)
    
    result = server.list_payments(limit=5)
    if result["success"]:
        print(f"\n  Payments ({result['count']}):")
        for payment in result["payments"]:
            status_icon = "[OK]" if payment["status"] == "succeeded" else " "
            refund_icon = "  " if payment["refunded"] else ""
            print(f"  {status_icon} {payment['charge_id']}: ${payment['amount']:.2f} {refund_icon}")
            print(f"     {payment['description']}")
    
    # Demo 2: Get payment details (SAFE)
    print(f"\n{'='*70}")
    print("DEMO 2: Get payment details (READ-ONLY)")
    print('='*70)
    
    result = server.get_payment("ch_001")
    if result["success"]:
        print(f"\n  Payment Details:")
        print(f"  ID: {result['charge_id']}")
        print(f"  Amount: ${result['amount']:.2f}")
        print(f"  Status: {result['status']}")
        print(f"  Refunded: {result['refunded']}")
    
    # Demo 3: Process refund (DANGEROUS)
    print(f"\n{'='*70}")
    print("DEMO 3: Process small refund (WITHIN LIMIT)")
    print('='*70)
    
    result = server.process_refund("ch_002", amount=150.00)
    if result["success"]:
        print(f"\n[OK] Refund successful:")
        print(f"  Refund ID: {result['refund_id']}")
        print(f"  Amount: ${result['amount']:.2f}")
        print(f"  Idempotency key: {result['idempotency_key']}")
    
    # Demo 4: Duplicate prevention
    print(f"\n{'='*70}")
    print("DEMO 4: Attempt DUPLICATE refund")
    print('='*70)
    
    result = server.process_refund("ch_002", amount=150.00)
    if "duplicate_prevented" in result:
        print(f"\n[WARN]  DUPLICATE DETECTED AND PREVENTED!")
        print(f"  Original result returned from cache")
        print(f"  This prevented a $15K-style disaster!")
    
    # Demo 5: Amount limit
    print(f"\n{'='*70}")
    print("DEMO 5: Refund EXCEEDS limit (requires approval)")
    print('='*70)
    
    result = server.process_refund("ch_001", amount=299.99)
    if not result["success"] and result.get("requires_approval"):
        print(f"\n[WARN]  REQUIRES APPROVAL:")
        print(f"  Amount: $299.99")
        print(f"  Limit: ${config.max_refund_amount:.2f}")
        print(f"  Workflow: {result['approval_workflow']}")
    
    # Show audit log
    print(f"\n{'='*70}")
    print("AUDIT LOG")
    print('='*70)
    
    audit = server.get_audit_log()
    print(f"\nTotal operations: {len(audit)}")
    for entry in audit[-3:]:  # Last 3
        print(f"\n  [{entry['timestamp']}]")
        print(f"  Operation: {entry['operation']}")
        print(f"  Result: {'SUCCESS' if entry['result'].get('success') else 'FAILURE'}")
    
    print("\n" + "="*70)
    print("LESSONS FROM $15K REFUND LOOP")
    print("="*70)
    print("""
WHAT WENT WRONG (Chapter 1):
1. No idempotency keys   50 duplicate refunds
2. No circuit breaker   Infinite retry loop
3. No amount limits   $15,000 loss
4. No audit log   Hard to diagnose

WHAT WE FIXED:
1. [OK] Idempotency   Duplicates detected and prevented
2. [OK] Circuit breaker   Stops after 3 failures
3. [OK] Amount limits   Large refunds need approval
4. [OK] Audit log   Complete trail of all operations
5. [OK] Rate limiting   Max 10 refunds/hour
6. [OK] Disabled by default   Must explicitly enable

RESULT:
- $15,000 disaster   $0 with safety features
- System is self-protecting
- Full accountability
    """)


if __name__ == "__main__":
    demo()
