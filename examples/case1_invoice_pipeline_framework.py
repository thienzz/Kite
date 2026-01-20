"""
CASE STUDY 1: INVOICE PROCESSING PIPELINE
==========================================
Demonstrates: DeterministicPipeline, VectorMemory, Safety Patterns

Complete invoice processing system using framework components:
- Deterministic 4-step pipeline
- Circuit breaker protection
- Idempotency (prevent duplicates)
- Vector memory storage
- Self-healing validation

Run: python case1_framework_pipeline.py
"""

import os
import sys
import time

# Add framework to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic_framework import AgenticAI
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import List, Optional
import json


# ============================================================================
# PYDANTIC MODELS (Deterministic Shell)
# ============================================================================

class LineItem(BaseModel):
    """Individual invoice line item"""
    description: str = Field(min_length=1)
    quantity: int = Field(gt=0)
    unit_price: float = Field(gt=0)
    total: float = Field(gt=0)


class InvoiceData(BaseModel):
    """Strict invoice schema with validation"""
    invoice_number: str = Field(pattern=r"^[A-Z0-9]{3,10}-\d{8}-\d{3}$")
    vendor_name: str = Field(min_length=2, max_length=100)
    amount: float = Field(gt=0)
    currency: str = Field(default="USD", pattern=r"^[A-Z]{3}$")
    invoice_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    line_items: List[LineItem] = Field(default_factory=list)
    
    @field_validator('amount')
    @classmethod
    def check_amount(cls, v):
        if v > 1_000_000:
            raise ValueError(f"Amount ${v:,.2f} seems unrealistic")
        return v


# ============================================================================
# MAIN EXAMPLE
# ============================================================================

def main():
    print("="*80)
    print("CASE STUDY 1: INVOICE PROCESSING PIPELINE")
    print("="*80)
    
    # ========================================================================
    # SETUP: Initialize Framework
    # ========================================================================
    print("\n[START] Initializing framework...")
    ai = AgenticAI()
    print("   [OK] Framework initialized")
    
    # ========================================================================
    # STEP 1: Create Pipeline Workflow
    # ========================================================================
    print("\n  Creating invoice processing pipeline...")
    
    pipeline = ai.create_workflow("invoice_processing")
    
    # Step 1: Load & Extract
    def load_invoice(state):
        """Step 1: Load invoice (mock data for demo)"""
        print("     Loading invoice...")
        
        # Mock invoice data
        state['raw_text'] = """
INVOICE #INV-20250117-001

Vendor: Acme Corporation
Date: 2025-01-17
Amount: $1,250.00

Items:
- Widget Pro (10x) @ $100.00 = $1,000.00
- Shipping = $250.00

Total: $1,250.00
        """
        return state
    
    # Step 2: Extract with LLM (Probabilistic Core)
    def extract_with_llm(state):
        """Step 2: Extract structured data with LLM"""
        print("     Extracting data with LLM...")
        
        schema = InvoiceData.model_json_schema()
        
        prompt = f"""Extract invoice data from this text and return ONLY valid JSON:

{state['raw_text']}

JSON Schema required:
{json.dumps(schema, indent=2)}

Rules:
1. invoice_number: Format XXX-YYYYMMDD-NNN
2. All dates YYYY-MM-DD format
3. amount as number"""

        # Use framework's LLM (protected by circuit breaker)
        response = ai.complete(prompt)
        
        state['extracted_json'] = response
        return state
    
    # Step 3: Validate (Deterministic Shell)
    def validate_data(state):
        """Step 3: Validate with Pydantic + Self-healing"""
        print("     Validating data...")
        
        max_retries = 3
        raw_json = state['extracted_json']
        
        for attempt in range(max_retries):
            try:
                # Clean JSON from markdown
                if '```json' in raw_json:
                    raw_json = raw_json.split('```json')[1].split('```')[0].strip()
                elif '```' in raw_json:
                    raw_json = raw_json.split('```')[1].split('```')[0].strip()
                
                data_dict = json.loads(raw_json)
                validated = InvoiceData(**data_dict)
                
                state['validated_data'] = validated.model_dump()
                print(f"   [OK] Validation passed")
                return state
                
            except Exception as e:
                print(f"   [WARN]  Attempt {attempt+1}/{max_retries} failed: {str(e)[:50]}...")
                
                if attempt == max_retries - 1:
                    raise
                
                # Self-healing: Ask LLM to fix
                print("     Asking LLM to fix...")
                raw_json = ai.complete(f"""
Fix this JSON validation error:

JSON: {raw_json}

Error: {e}

Return ONLY corrected JSON.""")
        
        raise RuntimeError("Validation failed")
    
    # Step 4: Store with Idempotency
    def store_invoice(state):
        """Step 4: Store in vector memory (prevent duplicates)"""
        print("     Storing in vector memory...")
        
        invoice_data = state['validated_data']
        invoice_number = invoice_data['invoice_number']
        
        # Idempotency check
        idempotency_key = f"invoice:{invoice_number}"
        
        if ai.idempotency.is_duplicate(idempotency_key):
            print(f"   [WARN]  Already processed: {invoice_number}")
            state['duplicate'] = True
            return state
        
        # Store in vector memory
        document_text = f"""
Invoice: {invoice_data['invoice_number']}
Vendor: {invoice_data['vendor_name']}
Amount: {invoice_data['currency']} {invoice_data['amount']}
Date: {invoice_data['invoice_date']}
Items: {len(invoice_data.get('line_items', []))}
        """.strip()
        
        ai.vector_memory.add_document(invoice_number, document_text)
        ai.idempotency.store_result(idempotency_key, True)
        
        state['duplicate'] = False
        print(f"   [OK] Stored: {invoice_number}")
        return state
    
    # Register all steps
    pipeline.add_step("load", load_invoice)
    pipeline.add_step("extract", extract_with_llm)
    pipeline.add_step("validate", validate_data)
    pipeline.add_step("store", store_invoice)
    
    print("   [OK] Created 4-step pipeline")
    print("     Steps: Load   Extract   Validate   Store")
    
    # ========================================================================
    # STEP 2: Process Invoice
    # ========================================================================
    print("\n[START] Processing invoice...")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        result = pipeline.execute({'invoice_id': 'INV-001'})
        
        elapsed = time.time() - start_time
        
        # Get final data
        final = result['results']['store']
        
        if final.get('duplicate'):
            print("\n[WARN]  DUPLICATE DETECTED")
            print(f"   Invoice already processed: {final['validated_data']['invoice_number']}")
        else:
            print("\n[OK] INVOICE PROCESSED SUCCESSFULLY")
            print("-" * 80)
            print(f"Invoice Number: {final['validated_data']['invoice_number']}")
            print(f"Vendor: {final['validated_data']['vendor_name']}")
            print(f"Amount: {final['validated_data']['currency']} {final['validated_data']['amount']:,.2f}")
            print(f"Date: {final['validated_data']['invoice_date']}")
            print(f"Items: {len(final['validated_data'].get('line_items', []))}")
            print("-" * 80)
        
        print(f"Processing Time: {elapsed:.2f}s")
        
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
    
    # ========================================================================
    # STEP 3: Test Duplicate Prevention
    # ========================================================================
    print("\n" + "="*80)
    print("TESTING IDEMPOTENCY (Duplicate Prevention)")
    print("="*80)
    
    print("\n  Processing same invoice again...")
    
    result2 = pipeline.execute({'invoice_id': 'INV-001'})
    final2 = result2['results']['store']
    
    if final2.get('duplicate'):
        print("   [OK] Idempotency working! Duplicate prevented.")
    else:
        print("   [ERROR] Warning: Duplicate was not detected")
    
    # ========================================================================
    # STEP 4: Semantic Search
    # ========================================================================
    print("\n" + "="*80)
    print("TESTING VECTOR MEMORY (Semantic Search)")
    print("="*80)
    
    # Add more demo invoices
    print("\n  Adding more invoices to memory...")
    
    demo_invoices = [
        ("INV-20250117-002", "TechStart Inc", 890.50, "2025-01-17"),
        ("INV-20250117-003", "Global Solutions", 2100.00, "2025-01-17"),
    ]
    
    for inv_num, vendor, amount, date in demo_invoices:
        doc = f"Invoice: {inv_num}\nVendor: {vendor}\nAmount: ${amount}\nDate: {date}"
        ai.vector_memory.add_document(inv_num, doc)
    
    print(f"   [OK] Added {len(demo_invoices)} invoices")
    
    # Search
    print("\n  Searching: 'Acme Corporation invoices'")
    
    results = ai.vector_memory.search("Acme Corporation", top_k=3)
    
    print(f"\n   Found {len(results)} results:")
    for doc_id, content, score in results:
        print(f"\n     {doc_id} (similarity: {score:.3f})")
        print(f"     {content[:80]}...")
    
    # ========================================================================
    # STEP 5: Framework Metrics
    # ========================================================================
    print("\n" + "="*80)
    print("FRAMEWORK METRICS")
    print("="*80)
    
    metrics = ai.get_metrics()
    
    print("\nCircuit Breaker:")
    print(f"   {metrics.get('circuit_breaker', {})}")
    
    print("\nIdempotency Manager:")
    print(f"   {metrics.get('idempotency', {})}")
    
    print("\nVector Memory:")
    print(f"   {metrics.get('vector_memory', {})}")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("[OK] CASE STUDY 1 COMPLETE")
    print("="*80)
    
    print("""
Summary - Framework Features Used:

1. Core Components:
   [OK] AgenticAI core initialization
   [OK] LLM provider integration
   
2. Pipeline System:
   [OK] DeterministicPipeline (4 steps)
   [OK] Linear workflow execution
   [OK] State passing between steps
   
3. Safety Patterns:
   [OK] Circuit Breaker (LLM failure protection)
   [OK] Idempotency Manager (duplicate prevention)
   [OK] Self-healing validation (retry with error correction)
   
4. Memory Systems:
   [OK] Vector Memory (document storage)
   [OK] Semantic search (find similar invoices)
   
5. Validation:
   [OK] Pydantic schemas (type safety)
   [OK] Business logic validation
   [OK] Automatic retry on errors

This demonstrates a PRODUCTION-READY invoice processing pipeline
with all safety guarantees and error handling!
    """)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Fatal Error: {e}")
        import traceback
        traceback.print_exc()
