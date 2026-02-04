"""
GraphRAG Implementation
Based on Chapter 3.3: When Vector Search Isn't Enough

Relationship-aware knowledge retrieval using knowledge graphs.

From book - Vector DB Failure Mode:
Query: "Who approved the AlphaCorp contract?"

Vector search finds 3 separate documents but can't connect:
- "Project Zeus budget approved by Sarah"
- "David leads Project Zeus"
- "AlphaCorp partnership with David"

GraphRAG connects: Sarah   Project Zeus   David   AlphaCorp [OK]

Run: python graph_rag.py
"""

import os
import json
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Entity:
    """A node in the knowledge graph."""
    id: str
    type: str  # person, project, company, etc.
    name: str
    properties: Dict = field(default_factory=dict)


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source: str  # entity id
    target: str  # entity id
    type: str    # manages, approves, partners_with, etc.
    properties: Dict = field(default_factory=dict)


@dataclass
class Document:
    """A document with extracted entities and relationships."""
    id: str
    text: str
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)


# ============================================================================
# ENTITY & RELATIONSHIP EXTRACTION
# ============================================================================

class EntityExtractor:
    """
    Extract entities and relationships from text using LLM.
    
    This is the core of GraphRAG - converting unstructured text
    into structured graph data.
    """
    
    def __init__(self, llm = None):
        self.llm = llm
        self.entity_types = [
            "person", "company", "project", "product",
            "location", "department", "role"
        ]
        
        self.relationship_types = [
            "manages", "works_on", "approves", "reports_to",
            "partners_with", "owns", "leads", "member_of"
        ]
    
    def extract(self, text: str) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract entities and relationships from text.
        
        Args:
            text: Input text
            
        Returns:
            (entities, relationships)
        """
        print(f"    Extracting entities and relationships...")
        
        # Create extraction prompt
        prompt = f"""Extract entities and relationships from this text.

Text: {text}

Entity types: {', '.join(self.entity_types)}
Relationship types: {', '.join(self.relationship_types)}

Output ONLY valid JSON with this structure:
{{
  "entities": [
    {{"id": "e1", "type": "person", "name": "Sarah Johnson"}},
    {{"id": "e2", "type": "project", "name": "Project Zeus"}}
  ],
  "relationships": [
    {{"source": "e1", "target": "e2", "type": "approves"}}
  ]
}}

Important:
- Use consistent IDs (e1, e2, e3...)
- Include all mentioned entities
- Capture all relationships
- Keep names as they appear in text"""

        if self.llm:
            response = self.llm.complete(prompt, temperature=0.1)
            content = response.strip()
        else:
            return [], []
        
        # Remove markdown if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        
        try:
            data = json.loads(content)
            
            entities = [
                Entity(
                    id=e["id"],
                    type=e["type"],
                    name=e["name"],
                    properties=e.get("properties", {})
                )
                for e in data.get("entities", [])
            ]
            
            relationships = [
                Relationship(
                    source=r["source"],
                    target=r["target"],
                    type=r["type"],
                    properties=r.get("properties", {})
                )
                for r in data.get("relationships", [])
            ]
            
            print(f"    [OK] Found {len(entities)} entities, {len(relationships)} relationships")
            
            return entities, relationships
            
        except json.JSONDecodeError as e:
            print(f"      Failed to parse: {e}")
            return [], []


# ============================================================================
# KNOWLEDGE GRAPH
# ============================================================================

class KnowledgeGraph:
    """
    Knowledge graph for storing and querying entity relationships.
    
    Uses NetworkX for graph operations and path finding.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.entities: Dict[str, Entity] = {}
        self.entity_name_to_id: Dict[str, str] = {}
        
        print("[OK] Knowledge Graph initialized")
    
    def save_to_file(self, path: str):
        """Save graph to JSON file."""
        data = nx.node_link_data(self.graph)
        try:
            with open(path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"[OK] Graph saved to {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save graph: {e}")

    def load_from_file(self, path: str):
        """Load graph from JSON file."""
        if not os.path.exists(path):
            return
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.graph = nx.node_link_graph(data)
            
            # Rebuild entities dict from graph nodes
            self.entities = {}
            self.entity_name_to_id = {}
            for node_id, attrs in self.graph.nodes(data=True):
                entity = Entity(
                    id=node_id,
                    type=attrs.get('type', 'unknown'),
                    name=attrs.get('name', 'Unknown'),
                    properties={k:v for k,v in attrs.items() if k not in ['type', 'name']}
                )
                self.entities[node_id] = entity
                self.entity_name_to_id[entity.name.lower()] = entity.id
                
            print(f"[OK] Graph loaded from {path} ({len(self.entities)} entities)")
        except Exception as e:
            print(f"[ERROR] Failed to load graph: {e}")

    
    def add_entity(self, entity: Entity):
        """Add entity to graph."""
        self.entities[entity.id] = entity
        self.entity_name_to_id[entity.name.lower()] = entity.id
        
        # Add node with attributes
        self.graph.add_node(
            entity.id,
            type=entity.type,
            name=entity.name,
            **entity.properties
        )
    
    def add_relationship(self, relationship: Relationship):
        """Add relationship to graph."""
        # Ensure entities exist
        if relationship.source not in self.entities:
            print(f"  [WARN]  Warning: Source entity {relationship.source} not found")
            return
        
        if relationship.target not in self.entities:
            print(f"  [WARN]  Warning: Target entity {relationship.target} not found")
            return
        
        # Add edge with attributes
        self.graph.add_edge(
            relationship.source,
            relationship.target,
            type=relationship.type,
            **relationship.properties
        )
    
    def add_document(self, document: Document):
        """Add all entities and relationships from document."""
        print(f"\n  Adding document: {document.id}")
        
        # Add entities
        for entity in document.entities:
            self.add_entity(entity)
        
        # Add relationships
        for rel in document.relationships:
            self.add_relationship(rel)
        
        print(f"  [OK] Added {len(document.entities)} entities, {len(document.relationships)} relationships")
    
    def find_entity(self, name: str) -> Optional[str]:
        """Find entity ID by name (case-insensitive)."""
        return self.entity_name_to_id.get(name.lower())
    
    def find_path(self, source_name: str, target_name: str) -> Optional[List[str]]:
        """
        Find shortest path between two entities.
        
        This is the key feature that vector search can't do!
        
        Args:
            source_name: Source entity name
            target_name: Target entity name
            
        Returns:
            List of entity IDs forming path, or None
        """
        source_id = self.find_entity(source_name)
        target_id = self.find_entity(target_name)
        
        if not source_id:
            print(f"    Entity not found: {source_name}")
            return None
        
        if not target_id:
            print(f"    Entity not found: {target_name}")
            return None
        
        try:
            # Convert to undirected for path finding
            undirected = self.graph.to_undirected()
            path = nx.shortest_path(undirected, source_id, target_id)
            return path
        except nx.NetworkXNoPath:
            print(f"    No path between {source_name} and {target_name}")
            return None
    
    def get_neighbors(self, entity_name: str, max_hops: int = 1) -> List[Entity]:
        """
        Get neighboring entities within N hops.
        
        Args:
            entity_name: Entity to start from
            max_hops: Maximum distance
            
        Returns:
            List of neighboring entities
        """
        entity_id = self.find_entity(entity_name)
        
        if not entity_id:
            return []
        
        # BFS to find neighbors within max_hops
        neighbors = set()
        current_level = {entity_id}
        
        for hop in range(max_hops):
            next_level = set()
            
            for node in current_level:
                # Get successors and predecessors
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            
            neighbors.update(next_level)
            current_level = next_level - neighbors - {entity_id}
        
        return [self.entities[eid] for eid in neighbors if eid in self.entities]
    
    def query_relationship(
        self,
        entity_name: str,
        relationship_type: Optional[str] = None
    ) -> List[Tuple[Entity, str, Entity]]:
        """
        Query relationships from an entity.
        
        Args:
            entity_name: Starting entity
            relationship_type: Filter by relationship type (optional)
            
        Returns:
            List of (source, relationship, target) tuples
        """
        entity_id = self.find_entity(entity_name)
        
        if not entity_id:
            return []
        
        results = []
        
        # Outgoing relationships
        for target_id in self.graph.successors(entity_id):
            edge_data = self.graph[entity_id][target_id]
            rel_type = edge_data.get("type", "unknown")
            
            if relationship_type is None or rel_type == relationship_type:
                results.append((
                    self.entities[entity_id],
                    rel_type,
                    self.entities[target_id]
                ))
        
        # Incoming relationships
        for source_id in self.graph.predecessors(entity_id):
            edge_data = self.graph[source_id][entity_id]
            rel_type = edge_data.get("type", "unknown")
            
            if relationship_type is None or rel_type == relationship_type:
                results.append((
                    self.entities[source_id],
                    rel_type,
                    self.entities[entity_id]
                ))
        
        return results
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "total_entities": len(self.entities),
            "total_relationships": self.graph.number_of_edges(),
            "entity_types": len(set(e.type for e in self.entities.values())),
            "connected_components": nx.number_weakly_connected_components(self.graph)
        }


# ============================================================================
# GRAPH RAG SYSTEM
# ============================================================================

class GraphRAG:
    """
    Complete GraphRAG system combining graph and vector search.
    
    From Chapter 3.3:
    - Vector search for fuzzy matching
    - Graph search for relationships
    - Hybrid approach for best results
    """
    
    def __init__(self, llm = None, persist_path: str = None):
        self.graph = KnowledgeGraph()
        self.extractor = EntityExtractor(llm=llm)
        self.documents: Dict[str, str] = {}
        self.persist_path = persist_path
        
        if self.persist_path and os.path.exists(self.persist_path):
            self.graph.load_from_file(self.persist_path)
            
        print("[OK] GraphRAG system initialized")
    
    def add_document(self, doc_id: str, text: str):
        """
        Add document and extract graph structure.
        
        Args:
            doc_id: Document ID
            text: Document text
        """
        print(f"\n  Processing document: {doc_id}")
        
        # Store document
        self.documents[doc_id] = text
        
        # Extract entities and relationships
        entities, relationships = self.extractor.extract(text)
        
        # Create document object
        document = Document(
            id=doc_id,
            text=text,
            entities=entities,
            relationships=relationships
        )
        
        # Add to graph
        self.graph.add_document(document)
        
        # Auto-save
        if self.persist_path:
            self.graph.save_to_file(self.persist_path)
    
    def hybrid_search(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Hybrid search combining graph and relationship reasoning.
        """
        answer = self.answer_relationship_query(query)
        
        is_relevant = "Connection found" in answer or "approved" in answer or "involved with" in answer
        
        return {
            "answer": answer,
            "success": True,
            "source": "graph_rag",
            "is_relevant": is_relevant,
            "documents": [{"content": answer, "id": "graph_result"}] if is_relevant else []
        }

    def query(self, query_str: str) -> Dict[str, Any]:
        """Query for entities and relationships."""
        answer = self.answer_relationship_query(query_str)
        
        # Extract entities mentioned in query
        entities = []
        for entity in self.graph.entities.values():
            if entity.name.lower() in query_str.lower():
                entities.append(entity.name)
                
        return {
            "answer": answer,
            "entities": entities,
            "success": True
        }

    def answer_relationship_query(self, query: str) -> str:
        """
        Answer queries about relationships.
        
        This is what vector search fails at!
        
        Args:
            query: Natural language query
            
        Returns:
            Answer based on graph analysis
        """
        print(f"\n  Query: {query}")
        
        # Simple pattern matching for demo
        # In production, use LLM to understand query intent
        
        query_lower = query.lower()
        
        # Pattern: "Who approved X?"
        if "who approved" in query_lower or "who signed off" in query_lower:
            return self._handle_approval_query(query)
        
        # Pattern: "How is X related to Y?"
        elif "related to" in query_lower or "connection between" in query_lower:
            return self._handle_connection_query(query)
        
        # Pattern: "What does X work on?"
        elif "work on" in query_lower or "working on" in query_lower:
            return self._handle_works_on_query(query)
        
        else:
            return "I can answer questions about relationships, approvals, and connections. Try: 'Who approved the AlphaCorp contract?'"
    
    def _handle_approval_query(self, query: str) -> str:
        """Handle 'who approved X' queries."""
        # Extract what was approved (simple approach)
        words = query.lower().split()
        
        # Look for entity names in query
        for entity_id, entity in self.graph.entities.items():
            if entity.name.lower() in query.lower():
                # Find who approved it
                approvers = []
                for source_id in self.graph.graph.predecessors(entity_id):
                    edge = self.graph.graph[source_id][entity_id]
                    if edge.get("type") == "approves":
                        approver = self.graph.entities[source_id]
                        approvers.append(approver.name)
                
                if approvers:
                    return f"{', '.join(approvers)} approved {entity.name}."
        
        return "I couldn't find approval information for that."
    
    def _handle_connection_query(self, query: str) -> str:
        """Handle 'how is X related to Y' queries."""
        # Extract entity names (simplified)
        words = query.split()
        
        entity_names = []
        for entity in self.graph.entities.values():
            if entity.name.lower() in query.lower():
                entity_names.append(entity.name)
        
        if len(entity_names) >= 2:
            path = self.graph.find_path(entity_names[0], entity_names[1])
            
            if path:
                # Build readable path
                path_desc = []
                for i in range(len(path) - 1):
                    source = self.graph.entities[path[i]]
                    target = self.graph.entities[path[i + 1]]
                    edge = self.graph.graph.get_edge_data(path[i], path[i + 1])
                    
                    if not edge:
                        edge = self.graph.graph.get_edge_data(path[i + 1], path[i])
                    
                    rel_type = edge.get("type", "connected to") if edge else "connected to"
                    path_desc.append(f"{source.name} {rel_type} {target.name}")
                
                return f"Connection found: {'   '.join(path_desc)}"
        
        return "I couldn't find a connection between those entities."
    
    def _handle_works_on_query(self, query: str) -> str:
        """Handle 'what does X work on' queries."""
        for entity_id, entity in self.graph.entities.items():
            if entity.name.lower() in query.lower() and entity.type == "person":
                # Find projects they work on
                projects = []
                for target_id in self.graph.graph.successors(entity_id):
                    edge = self.graph.graph[entity_id][target_id]
                    if edge.get("type") in ["works_on", "leads"]:
                        target = self.graph.entities[target_id]
                        projects.append(f"{target.name} ({edge.get('type')})")
                
                if projects:
                    return f"{entity.name} is involved with: {', '.join(projects)}"
        
        return "I couldn't find work information for that person."


# ============================================================================
# DEMO
# ============================================================================

def demo():
    print("=" * 70)
    print("GRAPHRAG DEMO")
    print("=" * 70)
    print("\nBased on Chapter 3.3: When Vector Search Isn't Enough")
    print("\nDemonstrating relationship-aware queries that")
    print("vector databases cannot answer.\n")
    print("=" * 70)
    
    # Initialize GraphRAG
    graph_rag = GraphRAG()
    
    # Add documents (from book example)
    documents = {
        "doc1": """
        The Project Zeus budget for Q4 2025 was approved by Sarah Johnson,
        the VP of Engineering. The project aims to modernize our infrastructure
        and is expected to cost $2.5M over 6 months.
        """,
        
        "doc2": """
        David Chen has been leading Project Zeus since September 2025.
        He reports directly to Sarah Johnson and manages a team of 12 engineers.
        The project is currently on schedule and within budget.
        """,
        
        "doc3": """
        AlphaCorp has entered into a strategic partnership with our company.
        David Chen negotiated the terms of the partnership, which includes
        joint development on Project Zeus infrastructure components.
        """,
        
        "doc4": """
        The AlphaCorp contract was finalized in October 2025.
        The partnership focuses on cloud infrastructure and will leverage
        the technologies developed in Project Zeus.
        """
    }
    
    print("\n  Adding documents to GraphRAG...")
    for doc_id, text in documents.items():
        graph_rag.add_document(doc_id, text.strip())
    
    # Show graph statistics
    print("\n" + "="*70)
    print("KNOWLEDGE GRAPH STATISTICS")
    print("="*70)
    stats = graph_rag.graph.get_stats()
    print(f"Total entities: {stats['total_entities']}")
    print(f"Total relationships: {stats['total_relationships']}")
    print(f"Entity types: {stats['entity_types']}")
    print(f"Connected components: {stats['connected_components']}")
    
    # Test queries (from book example)
    print("\n" + "="*70)
    print("RELATIONSHIP QUERIES")
    print("="*70)
    print("\nThese queries require graph traversal.")
    print("Vector search would fail!\n")
    
    queries = [
        "Who approved the AlphaCorp contract?",
        "How is Sarah Johnson related to AlphaCorp?",
        "What does David Chen work on?"
    ]
    
    for i, query in enumerate(queries, 1):
        print(f"\n{' '*70}")
        print(f"Query {i}: {query}")
        print(' '*70)
        
        answer = graph_rag.answer_relationship_query(query)
        print(f"\n  Answer:\n{answer}")
    
    # Show path finding
    print("\n" + "="*70)
    print("PATH FINDING EXAMPLE")
    print("="*70)
    print("\nFinding connection: Sarah Johnson   AlphaCorp")
    
    path = graph_rag.graph.find_path("Sarah Johnson", "AlphaCorp")
    
    if path:
        print("\n[LINK] Connection found:")
        for i in range(len(path)):
            entity = graph_rag.graph.entities[path[i]]
            print(f"  {i+1}. {entity.name} ({entity.type})")
            
            if i < len(path) - 1:
                edge = graph_rag.graph.graph.get_edge_data(path[i], path[i+1])
                if not edge:
                    edge = graph_rag.graph.graph.get_edge_data(path[i+1], path[i])
                rel_type = edge.get("type", " ") if edge else " "
                print(f"        {rel_type}")
    
    print("\n" + "="*70)
    print("WHY VECTOR SEARCH FAILS (From Book)")
    print("="*70)
    print("""
Vector Search Approach:
1. Query: "Who approved AlphaCorp contract?"
2. Finds 3 separate documents:
   - Document 1: "Sarah approved Project Zeus"
   - Document 2: "David leads Project Zeus"
   - Document 3: "AlphaCorp partnership with David"
3. CANNOT connect the dots!  

GraphRAG Approach:
1. Query: "Who approved AlphaCorp contract?"
2. Traverses graph:
   Sarah   approves   Project Zeus
   David   leads   Project Zeus
   David   negotiates   AlphaCorp partnership
3. Finds path: Sarah   Project Zeus   David   AlphaCorp [OK]
4. Answer: "Sarah approved the project that led to AlphaCorp partnership"

Key Insight:
- Vectors: Good for similarity matching
- Graphs: Good for relationship reasoning
- Hybrid: Best of both worlds!
    """)


if __name__ == "__main__":
    demo()
