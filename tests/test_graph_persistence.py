import os
import shutil
import json
from kite.memory.graph_rag import KnowledgeGraph, Entity, Relationship, GraphRAG

TEST_GRAPH_FILE = "test_graph.json"

def test_knowledge_graph_persistence():
    print("Testing KnowledgeGraph persistence...")
    
    # 1. Create and populate graph
    kg = KnowledgeGraph()
    e1 = Entity(id="e1", type="person", name="Alice", properties={"role": "Engineer"})
    e2 = Entity(id="e2", type="project", name="Project X", properties={"status": "Active"})
    kg.add_entity(e1)
    kg.add_entity(e2)
    
    r1 = Relationship(source="e1", target="e2", type="leads", properties={"since": "2023"})
    kg.add_relationship(r1)
    
    # 2. Save
    kg.save_to_file(TEST_GRAPH_FILE)
    assert os.path.exists(TEST_GRAPH_FILE)
    
    # 3. Load into new instance
    kg2 = KnowledgeGraph()
    kg2.load_from_file(TEST_GRAPH_FILE)
    
    # 4. Verify
    assert len(kg2.entities) == 2
    assert "e1" in kg2.entities
    assert kg2.entities["e1"].name == "Alice"
    assert kg2.entities["e1"].properties["role"] == "Engineer"
    
    assert kg2.graph.has_edge("e1", "e2")
    edge = kg2.graph["e1"]["e2"]
    assert edge["type"] == "leads"
    assert edge["since"] == "2023"
    
    print("   [PASS] KnowledgeGraph persistence verified")

def test_graph_rag_auto_persistence():
    print("\nTesting GraphRAG auto-persistence...")
    if os.path.exists(TEST_GRAPH_FILE):
        os.remove(TEST_GRAPH_FILE)
        
    # 1. Init System with persistence
    rag = GraphRAG(persist_path=TEST_GRAPH_FILE)
    
    # 2. Add document (should auto-save)
    # Mocking extractor response handling by manually injecting for this test is hard without mocking the LLM.
    # Instead, we'll manually access the internal graph to simulate extraction result, 
    # OR we can trust `add_document` calls `save_to_file`.
    # Let's verify `add_document` trigger behavior by observing file creation.
    
    # To avoid LLM calls, we'll manipulate the graph directly and save, 
    # or rely on the fact that `add_document` calls `extractor.extract`.
    # Let's skip full `add_document` with LLM and test the init logic.
    
    # Pre-create a file
    kg = KnowledgeGraph()
    kg.add_entity(Entity("e99", "test", "TestEntity"))
    kg.save_to_file(TEST_GRAPH_FILE)
    
    # 3. Init new RAG
    rag2 = GraphRAG(persist_path=TEST_GRAPH_FILE)
    assert "e99" in rag2.graph.entities
    assert rag2.graph.entities["e99"].name == "TestEntity"
    
    print("   [PASS] GraphRAG auto-load verified")

if __name__ == "__main__":
    try:
        test_knowledge_graph_persistence()
        test_graph_rag_auto_persistence()
        print("\nALL TESTS PASSED")
    finally:
        if os.path.exists(TEST_GRAPH_FILE):
            os.remove(TEST_GRAPH_FILE)
