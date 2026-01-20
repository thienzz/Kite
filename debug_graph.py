from kite.memory.graph_rag import GraphRAG, Entity, Relationship

def test_graph_rag():
    graph = GraphRAG()
    
    # Add entities first
    graph.graph.add_entity(Entity("user", "person", "User"))
    graph.graph.add_entity(Entity("product", "product", "Product"))
    graph.graph.add_entity(Entity("company", "company", "Company"))
    
    graph.graph.add_relationship(Relationship("user", "bought", "product"))
    graph.graph.add_relationship(Relationship("product", "manufactured_by", "company"))
    
    print(f"Entities: {graph.graph.entities}")
    print(f"Name to ID: {graph.graph.entity_name_to_id}")
    print(f"Edges: {graph.graph.graph.edges(data=True)}")
    
    neighbors = graph.graph.get_neighbors("User")
    print(f"Neighbors for 'User': {[e.id for e in neighbors]}")
    
    results = [e.id == "product" for e in neighbors]
    print(f"Any product in neighbors? {any(results)}")
    assert any(results)

if __name__ == "__main__":
    test_graph_rag()
