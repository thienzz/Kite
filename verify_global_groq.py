from kite import Kite
import os

def test_global_config():
    print("="*50)
    print("VERIFYING GLOBAL GROQ CONFIGURATION")
    print("="*50)
    
    ai = Kite()
    
    print(f"\nGlobal LLM Provider: {ai.config.get('llm_provider')}")
    print(f"Global LLM Model:    {ai.config.get('llm_model')}")
    print(f"Global SLM Provider: {ai.config.get('slm_provider')}")
    print(f"Global SLM SQL:      {ai.config.get('slm_sql_model')}")
    
    # Check instance name
    print(f"\nLLM Instance: {ai.llm.name}")
    
    if "Groq" in ai.llm.name:
        print("\n[SUCCESS] Framework is using Groq globally!")
    else:
        print("\n[FAILURE] Framework is NOT using Groq.")
    
    print("="*50)

if __name__ == "__main__":
    test_global_config()
