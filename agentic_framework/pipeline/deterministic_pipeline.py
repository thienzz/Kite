"""
Deterministic Pipeline Pattern
Based on Chapter 5: The Deterministic Pipeline

Level 1 Autonomy: The assembly line pattern with ZERO risk.

From book:
"If you can write an SOP for it, build a Pipeline, not an Agent."

Flow: Input   Extract   Validate   Action
- No loops
- No choices
- AI used for extraction only
- Risk: Zero

Run: python deterministic_pipeline.py
"""

import os
import json
import re
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ============================================================================
# DATA MODELS
# ============================================================================

class InvoiceStatus(Enum):
    """Invoice processing status."""
    PENDING = "pending"
    EXTRACTED = "extracted"
    VALIDATED = "validated"
    APPROVED = "approved"
    REJECTED = "rejected"
    ERROR = "error"


@dataclass
class InvoiceData:
    """Structured invoice data."""
    invoice_number: str
    vendor_name: str
    amount: float
    date: str
    line_items: List[Dict]
    payment_terms: str
    
    # Validation results
    is_valid: bool = False
    validation_errors: List[str] = None
    
    # Approval
    approved: bool = False
    approval_reason: str = ""


@dataclass
class PipelineState:
    """Current state of document in pipeline."""
    document_id: str
    status: InvoiceStatus
    data: Optional[InvoiceData] = None
    created_at: datetime = None
    updated_at: datetime = None
    error: Optional[str] = None


# ============================================================================
# PIPELINE STAGES
# ============================================================================

class Stage1_Extraction:
    """
    Stage 1: Extract structured data from unstructured text.
    
    This is the ONLY place AI is used in the pipeline.
    AI extracts data, but doesn't make decisions.
    """
    
    def __init__(self):
        self.name = "Extraction"
    
    def execute(self, raw_text: str) -> InvoiceData:
        """
        Extract invoice data using AI.
        
        Args:
            raw_text: Raw invoice text
            
        Returns:
            Structured invoice data
        """
        print(f"\n    Stage 1: {self.name}")
        print(f"     Extracting structured data from text...")
        
        # Use LLM for extraction
        prompt = f"""Extract invoice data from this text. Return ONLY valid JSON.

Text:
{raw_text}

Required JSON format:
{{
  "invoice_number": "string",
  "vendor_name": "string", 
  "amount": number,
  "date": "YYYY-MM-DD",
  "line_items": [
    {{"description": "string", "quantity": number, "price": number}}
  ],
  "payment_terms": "string"
}}"""

        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        try:
            data = json.loads(content)
            
            invoice = InvoiceData(
                invoice_number=data["invoice_number"],
                vendor_name=data["vendor_name"],
                amount=float(data["amount"]),
                date=data["date"],
                line_items=data["line_items"],
                payment_terms=data["payment_terms"]
            )
            
            print(f"     [OK] Extracted invoice #{invoice.invoice_number}")
            print(f"       Vendor: {invoice.vendor_name}")
            print(f"       Amount: ${invoice.amount:,.2f}")
            
            return invoice
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"       Extraction failed: {e}")
            raise ValueError(f"Failed to extract invoice data: {e}")


class Stage2_Validation:
    """
    Stage 2: Validate extracted data.
    
    This is DETERMINISTIC - no AI, just rules.
    """
    
    def __init__(self):
        self.name = "Validation"
        
        # Business rules
        self.max_amount = 10000.00
        self.approved_vendors = [
            "Acme Corp",
            "TechSupply Inc", 
            "Office Depot",
            "AWS"
        ]
    
    def execute(self, invoice: InvoiceData) -> InvoiceData:
        """
        Validate invoice data against business rules.
        
        Args:
            invoice: Extracted invoice data
            
        Returns:
            Invoice with validation results
        """
        print(f"\n  [OK] Stage 2: {self.name}")
        print(f"     Validating against business rules...")
        
        errors = []
        
        # Rule 1: Amount limit
        if invoice.amount > self.max_amount:
            errors.append(f"Amount ${invoice.amount:,.2f} exceeds limit ${self.max_amount:,.2f}")
        
        # Rule 2: Approved vendor
        if invoice.vendor_name not in self.approved_vendors:
            errors.append(f"Vendor '{invoice.vendor_name}' not in approved list")
        
        # Rule 3: Valid date
        try:
            date_obj = datetime.strptime(invoice.date, "%Y-%m-%d")
            if date_obj > datetime.now():
                errors.append(f"Invoice date {invoice.date} is in the future")
        except ValueError:
            errors.append(f"Invalid date format: {invoice.date}")
        
        # Rule 4: Line items total matches
        line_items_total = sum(
            item["quantity"] * item["price"]
            for item in invoice.line_items
        )
        
        if abs(line_items_total - invoice.amount) > 0.01:
            errors.append(f"Line items total ${line_items_total:.2f} doesn't match invoice amount ${invoice.amount:.2f}")
        
        # Set validation results
        invoice.is_valid = len(errors) == 0
        invoice.validation_errors = errors if errors else None
        
        if invoice.is_valid:
            print(f"     [OK] All validation checks passed")
        else:
            print(f"       Validation failed:")
            for error in errors:
                print(f"       - {error}")
        
        return invoice


class Stage3_Approval:
    """
    Stage 3: Determine approval routing.
    
    This is DETERMINISTIC - no AI, just business logic.
    """
    
    def __init__(self):
        self.name = "Approval Routing"
        
        # Approval thresholds
        self.auto_approve_limit = 1000.00
    
    def execute(self, invoice: InvoiceData) -> InvoiceData:
        """
        Route invoice for approval based on rules.
        
        Args:
            invoice: Validated invoice data
            
        Returns:
            Invoice with approval decision
        """
        print(f"\n      Stage 3: {self.name}")
        print(f"     Determining approval path...")
        
        if not invoice.is_valid:
            invoice.approved = False
            invoice.approval_reason = "Failed validation checks"
            print(f"       Rejected: {invoice.approval_reason}")
            return invoice
        
        # Auto-approve small amounts
        if invoice.amount <= self.auto_approve_limit:
            invoice.approved = True
            invoice.approval_reason = f"Auto-approved (under ${self.auto_approve_limit:.2f})"
            print(f"     [OK] {invoice.approval_reason}")
        else:
            invoice.approved = False
            invoice.approval_reason = f"Requires manual approval (over ${self.auto_approve_limit:.2f})"
            print(f"       {invoice.approval_reason}")
        
        return invoice


class Stage4_Action:
    """
    Stage 4: Take action based on approval.
    
    This is DETERMINISTIC - execute the decision.
    """
    
    def __init__(self):
        self.name = "Action"
    
    def execute(self, invoice: InvoiceData) -> Dict[str, Any]:
        """
        Execute final action.
        
        Args:
            invoice: Approved/rejected invoice
            
        Returns:
            Action result
        """
        print(f"\n    Stage 4: {self.name}")
        print(f"     Executing final action...")
        
        if invoice.approved:
            # Record in database
            result = self._record_approved_invoice(invoice)
            print(f"     [OK] Invoice recorded for payment")
            
            # Schedule payment
            self._schedule_payment(invoice)
            print(f"     [OK] Payment scheduled")
            
            return {
                "action": "approved_and_recorded",
                "invoice_id": invoice.invoice_number,
                "amount": invoice.amount,
                "payment_date": "2026-01-30"  # Based on terms
            }
        else:
            # Route for manual review
            result = self._route_for_review(invoice)
            print(f"     [OK] Routed for manual review")
            
            return {
                "action": "routed_for_review",
                "invoice_id": invoice.invoice_number,
                "reason": invoice.approval_reason,
                "reviewer": "finance_team"
            }
    
    def _record_approved_invoice(self, invoice: InvoiceData) -> Dict:
        """Record in database (simulated)."""
        return {
            "id": invoice.invoice_number,
            "status": "approved",
            "timestamp": datetime.now().isoformat()
        }
    
    def _schedule_payment(self, invoice: InvoiceData):
        """Schedule payment (simulated)."""
        # In production: integrate with payment system
        pass
    
    def _route_for_review(self, invoice: InvoiceData) -> Dict:
        """Route to review queue (simulated)."""
        return {
            "queue": "manual_review",
            "invoice": invoice.invoice_number,
            "timestamp": datetime.now().isoformat()
        }


# ============================================================================
# DETERMINISTIC PIPELINE
# ============================================================================

class DeterministicPipeline:
    """
    A deterministic invoice processing pipeline.
    
    From Chapter 5:
    "If you can write an SOP for it, build a Pipeline, not an Agent."
    
    This is Level 1 Autonomy:
    - Fixed sequence
    - No loops
    - No choices (by AI)
    - AI used ONLY for extraction
    - All decisions are rule-based
    
    Flow:
    1. Extract (AI)
    2. Validate (Rules)
    3. Approve (Rules)
    4. Action (Rules)
    
    Risk: ZERO - completely predictable
    
    Example:
        pipeline = DeterministicPipeline()
        
        result = pipeline.process(raw_invoice_text)
        
        if result.status == InvoiceStatus.APPROVED:
            print("Invoice auto-approved!")
    """
    
    def __init__(self, name: str = "pipeline"):
        self.name = name
        self.steps: Dict[str, Callable] = {}
        
        # Original fixed stages (kept for backward compatibility if needed)
        self.stage1 = Stage1_Extraction()
        self.stage2 = Stage2_Validation()
        self.stage3 = Stage3_Approval()
        self.stage4 = Stage4_Action()
        
        # Processing history
        self.history: List[Dict] = []
        
        print(f"[OK] Pipeline '{self.name}' initialized")
    
    def add_step(self, name: str, func: Callable):
        """Add a step to the pipeline."""
        self.steps[name] = func
        print(f"  [OK] Added step: {name}")
    
    def execute(self, data: Any) -> Dict:
        """Execute all steps in the pipeline."""
        print(f"\n[START] Executing pipeline: {self.name}")
        current_data = data
        results = {}
        
        for step_name, func in self.steps.items():
            current_data = func(current_data)
            results[step_name] = current_data
            
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "input": data,
            "results": results
        })
        
        return {"status": "success", "results": results}
    
    def process(self, raw_text: str, document_id: str = None) -> PipelineState:
        """
        Process a document through the pipeline.
        
        This is a LINEAR, DETERMINISTIC flow.
        No loops, no retries, no decisions by AI.
        
        Args:
            raw_text: Raw invoice text
            document_id: Document identifier
            
        Returns:
            Final pipeline state
        """
        doc_id = document_id or f"INV-{len(self.history)+1:04d}"
        
        print(f"\n{'='*70}")
        print(f"PROCESSING DOCUMENT: {doc_id}")
        print('='*70)
        
        state = PipelineState(
            document_id=doc_id,
            status=InvoiceStatus.PENDING,
            created_at=datetime.now()
        )
        
        try:
            # Stage 1: Extract (AI)
            state.status = InvoiceStatus.EXTRACTED
            invoice = self.stage1.execute(raw_text)
            state.data = invoice
            
            # Stage 2: Validate (Deterministic)
            state.status = InvoiceStatus.VALIDATED
            invoice = self.stage2.execute(invoice)
            
            # Stage 3: Approve (Deterministic)
            invoice = self.stage3.execute(invoice)
            
            if invoice.approved:
                state.status = InvoiceStatus.APPROVED
            else:
                state.status = InvoiceStatus.REJECTED
            
            # Stage 4: Action (Deterministic)
            action_result = self.stage4.execute(invoice)
            
            state.updated_at = datetime.now()
            
            # Add to history
            self.history.append(state)
            
            print(f"\n{'='*70}")
            print(f"[OK] PIPELINE COMPLETE")
            print(f"  Status: {state.status.value}")
            print(f"  Document: {doc_id}")
            print('='*70)
            
            return state
            
        except Exception as e:
            state.status = InvoiceStatus.ERROR
            state.error = str(e)
            state.updated_at = datetime.now()
            
            self.history.append(state)
            
            print(f"\n{'='*70}")
            print(f"  PIPELINE ERROR")
            print(f"  Error: {e}")
            print('='*70)
            
            return state
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        if not self.history:
            return {"total_processed": 0}
        
        statuses = [state.status for state in self.history]
        
        return {
            "total_processed": len(self.history),
            "approved": statuses.count(InvoiceStatus.APPROVED),
            "rejected": statuses.count(InvoiceStatus.REJECTED),
            "errors": statuses.count(InvoiceStatus.ERROR),
            "success_rate": (
                statuses.count(InvoiceStatus.APPROVED) / len(statuses) * 100
                if statuses else 0
            )
        }


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("DETERMINISTIC PIPELINE DEMO")
    print("=" * 70)
    print("\nBased on Chapter 5: The Deterministic Pipeline")
    print("\nLevel 1 Autonomy - The Assembly Line")
    print("    Fixed sequence")
    print("    No loops")
    print("    No AI decisions")
    print("    Risk: ZERO")
    print("=" * 70)
    
    # Initialize pipeline
    pipeline = DeterministicPipeline()
    
    # Test invoices
    test_invoices = [
        # Case 1: Should auto-approve
        """
        INVOICE #12345
        From: Acme Corp
        Date: 2026-01-10
        
        Line Items:
        - Office supplies (10 x $25.00) = $250.00
        - Printer paper (5 x $15.00) = $75.00
        
        Total: $325.00
        Payment Terms: Net 30
        """,
        
        # Case 2: Needs manual approval (high amount)
        """
        INVOICE #12346
        From: TechSupply Inc
        Date: 2026-01-12
        
        Line Items:
        - Server hardware (2 x $3000.00) = $6000.00
        
        Total: $6000.00
        Payment Terms: Net 30
        """,
        
        # Case 3: Should reject (unapproved vendor)
        """
        INVOICE #12347
        From: Random Supplier LLC
        Date: 2026-01-13
        
        Line Items:
        - Services (1 x $500.00) = $500.00
        
        Total: $500.00
        Payment Terms: Net 30
        """
    ]
    
    for i, invoice_text in enumerate(test_invoices, 1):
        result = pipeline.process(
            invoice_text,
            document_id=f"TEST-{i:03d}"
        )
    
    # Show statistics
    print(f"\n{'='*70}")
    print("PIPELINE STATISTICS")
    print('='*70)
    
    stats = pipeline.get_stats()
    print(f"Total processed: {stats['total_processed']}")
    print(f"Approved: {stats['approved']}")
    print(f"Rejected: {stats['rejected']}")
    print(f"Errors: {stats['errors']}")
    print(f"Success rate: {stats['success_rate']:.1f}%")
    
    print("\n" + "="*70)
    print("WHY PIPELINES WIN FOR SOPs (From Book)")
    print("="*70)
    print("""
1. PREDICTABILITY
   [OK] Same input = same output
   [OK] No surprises
   [OK] Easy to test

2. AUDITABILITY
   [OK] Clear audit trail
   [OK] Know exactly what happened
   [OK] Compliance-friendly

3. SAFETY
   [OK] No infinite loops
   [OK] No hallucinations making decisions
   [OK] AI only extracts, doesn't decide

4. PERFORMANCE
   [OK] Fast execution
   [OK] Low cost (one AI call)
   [OK] Scalable

WHEN TO USE:
- Process is well-defined
- Have clear business rules
- High volume, low variation
- Compliance/audit requirements

WHEN NOT TO USE:
- Process needs exploration
- Rules change frequently
- Need creative problem-solving
- Ambiguous decision-making

KEY PRINCIPLE:
"If you can write an SOP for it, 
 build a Pipeline, not an Agent."
    """)


if __name__ == "__main__":
    demo()
