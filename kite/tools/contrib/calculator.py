"""
Calculator Tool - Simple arithmetic operations.
"""

def calculator(expression: str):
    """
    Evaluate a mathematical expression.
    
    Args:
        expression: Mathematical expression (e.g. "2 + 2", "15 * 4")
        
    Returns:
        The result of the expression.
    """
    try:
        # NOTE: Using eval for simplicity in this mock, 
        # but in production, use a safe math parser!
        # Restricting to basic math characters for safety.
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return {"error": "Invalid characters in expression"}
            
        result = eval(expression)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"error": str(e)}
