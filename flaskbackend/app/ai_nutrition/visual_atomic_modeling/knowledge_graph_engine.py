"""
Knowledge Graph Engine with GPT-4 Integration
==============================================

Converts disease nutritional data into queryable knowledge graphs
and integrates with GPT-4 for intelligent reasoning.
"""

import json
import os
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, asdict
from datetime import datetime
import networkx as nx
from comprehensive_disease_db import ComprehensiveDiseaseDatabase
from disease_optimization_engine import MultiDiseaseOptimizer

if TYPE_CHECKING:
    try:
        from openai import OpenAI, AzureOpenAI  # type: ignore
        import anthropic  # type: ignore
    except ImportError:
        pass


@dataclass
class KnowledgeNode:
    """Node in the knowledge graph"""
    id: str
    type: str  # 'disease', 'nutrient', 'food', 'restriction', 'person'
    properties: Dict[str, Any]
    relationships: List[Dict[str, str]]


@dataclass
class KnowledgeEdge:
    """Edge connecting nodes"""
    source: str
    target: str
    relationship: str
    properties: Dict[str, Any]


class NutritionalKnowledgeGraph:
    """Build and query knowledge graphs from nutritional data"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: List[KnowledgeEdge] = []
        
    def build_from_optimization_result(
        self, 
        result: Dict[str, Any],
        family_members: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Build knowledge graph from optimization result
        
        Args:
            result: Output from MultiDiseaseOptimizer
            family_members: Family member data with diseases
            
        Returns:
            Knowledge graph structure
        """
        
        # Add family member nodes
        for member in family_members:
            self._add_person_node(member)
        
        # Add disease nodes
        db = ComprehensiveDiseaseDatabase()
        for member in family_members:
            for disease_id in member.get('diseases', []):
                disease = db.get_disease(disease_id)
                if disease:
                    self._add_disease_node(disease, member['name'])
        
        # Add nutrient nodes from guidelines
        for nutrient, guideline in result['unified_nutritional_targets'].items():
            self._add_nutrient_node(nutrient, guideline)
        
        # Add food restriction nodes
        for restriction in result['food_restrictions']:
            self._add_food_restriction_node(restriction)
        
        # Add recommended food nodes
        for food in result['recommended_foods']:
            self._add_recommended_food_node(food)
        
        # Build graph structure
        self._build_graph_structure()
        
        return self.export_graph()
    
    def _add_person_node(self, member: Dict[str, Any]):
        """Add person node to graph"""
        node_id = f"person:{member['name']}"
        
        node = KnowledgeNode(
            id=node_id,
            type='person',
            properties={
                'name': member['name'],
                'age': member.get('age'),
                'diseases': member.get('diseases', []),
                'dietary_preferences': member.get('dietary_preferences')
            },
            relationships=[]
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
    
    def _add_disease_node(self, disease, person_name: str):
        """Add disease node to graph"""
        node_id = f"disease:{disease.disease_id}"
        
        if node_id not in self.nodes:
            node = KnowledgeNode(
                id=node_id,
                type='disease',
                properties={
                    'disease_id': disease.disease_id,
                    'name': disease.name,
                    'category': disease.category,
                    'icd10_codes': disease.icd10_codes,
                    'meal_timing_important': disease.meal_timing_important,
                    'portion_control_critical': disease.portion_control_critical
                },
                relationships=[]
            )
            
            self.nodes[node_id] = node
            self.graph.add_node(node_id, **asdict(node))
        
        # Create relationship: person HAS_DISEASE disease
        edge = KnowledgeEdge(
            source=f"person:{person_name}",
            target=node_id,
            relationship='HAS_DISEASE',
            properties={'diagnosed': True}
        )
        self.edges.append(edge)
    
    def _add_nutrient_node(self, nutrient: str, guideline: Dict[str, Any]):
        """Add nutrient node to graph"""
        node_id = f"nutrient:{nutrient}"
        
        node = KnowledgeNode(
            id=node_id,
            type='nutrient',
            properties={
                'name': nutrient,
                'target_min': guideline.get('target_min'),
                'target_max': guideline.get('target_max'),
                'unit': guideline.get('unit'),
                'priority': guideline.get('priority'),
                'reasons': guideline.get('reasons', [])
            },
            relationships=[]
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
        
        # Create relationships to diseases that require this nutrient
        for reason in guideline.get('reasons', []):
            edge = KnowledgeEdge(
                source=f"disease:{reason.get('disease', '')}",
                target=node_id,
                relationship='REQUIRES_NUTRIENT',
                properties={
                    'reason': reason.get('reason'),
                    'priority': guideline.get('priority')
                }
            )
            self.edges.append(edge)
    
    def _add_food_restriction_node(self, restriction: Dict[str, Any]):
        """Add food restriction node to graph"""
        food_item = restriction['food_item']
        node_id = f"restriction:{food_item}"
        
        node = KnowledgeNode(
            id=node_id,
            type='restriction',
            properties={
                'food_item': food_item,
                'restriction_type': restriction['restriction_type'],
                'severity': restriction['severity'],
                'reason': restriction['reason'],
                'alternative': restriction['alternative'],
                'affects_members': restriction.get('affects_members', []),
                'related_diseases': restriction.get('related_diseases', [])
            },
            relationships=[]
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
        
        # Create relationships
        for member in restriction.get('affects_members', []):
            edge = KnowledgeEdge(
                source=f"person:{member}",
                target=node_id,
                relationship='MUST_AVOID',
                properties={'severity': restriction['severity']}
            )
            self.edges.append(edge)
    
    def _add_recommended_food_node(self, food: str):
        """Add recommended food node to graph"""
        node_id = f"food:{food}"
        
        node = KnowledgeNode(
            id=node_id,
            type='food',
            properties={
                'name': food,
                'recommended': True
            },
            relationships=[]
        )
        
        self.nodes[node_id] = node
        self.graph.add_node(node_id, **asdict(node))
    
    def _build_graph_structure(self):
        """Build graph edges from collected relationships"""
        for edge in self.edges:
            if edge.source in self.graph.nodes and edge.target in self.graph.nodes:
                self.graph.add_edge(
                    edge.source,
                    edge.target,
                    relationship=edge.relationship,
                    **edge.properties
                )
    
    def export_graph(self) -> Dict[str, Any]:
        """Export graph in standard format"""
        
        nodes_list = []
        edges_list = []
        
        # Export nodes
        for node_id, node in self.nodes.items():
            nodes_list.append({
                'id': node.id,
                'type': node.type,
                'properties': node.properties
            })
        
        # Export edges
        for u, v, data in self.graph.edges(data=True):
            edges_list.append({
                'source': u,
                'target': v,
                'relationship': data.get('relationship'),
                'properties': {k: v for k, v in data.items() if k != 'relationship'}
            })
        
        # Graph statistics
        stats = {
            'total_nodes': len(nodes_list),
            'total_edges': len(edges_list),
            'node_types': {},
            'relationship_types': {}
        }
        
        for node in nodes_list:
            node_type = node['type']
            stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        for edge in edges_list:
            rel_type = edge['relationship']
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
        
        return {
            'nodes': nodes_list,
            'edges': edges_list,
            'statistics': stats,
            'created_at': datetime.now().isoformat()
        }
    
    def query_graph(self, query_type: str, **params) -> List[Dict[str, Any]]:
        """Query the knowledge graph"""
        
        if query_type == 'person_diseases':
            person_name = params.get('person')
            person_id = f"person:{person_name}"
            if person_id in self.graph:
                diseases = [
                    self.graph.nodes[v]['properties']
                    for u, v, data in self.graph.edges(person_id, data=True)
                    if data.get('relationship') == 'HAS_DISEASE'
                ]
                return diseases
        
        elif query_type == 'disease_nutrients':
            disease_id = params.get('disease_id')
            disease_node_id = f"disease:{disease_id}"
            if disease_node_id in self.graph:
                nutrients = [
                    self.graph.nodes[v]['properties']
                    for u, v, data in self.graph.edges(disease_node_id, data=True)
                    if data.get('relationship') == 'REQUIRES_NUTRIENT'
                ]
                return nutrients
        
        elif query_type == 'person_restrictions':
            person_name = params.get('person')
            person_id = f"person:{person_name}"
            if person_id in self.graph:
                restrictions = [
                    self.graph.nodes[v]['properties']
                    for u, v, data in self.graph.edges(person_id, data=True)
                    if data.get('relationship') == 'MUST_AVOID'
                ]
                return restrictions
        
        return []
    
    def export_for_llm(self) -> str:
        """Export graph as LLM-friendly text"""
        
        export_data = self.export_graph()
        
        text_output = []
        text_output.append("=== NUTRITIONAL KNOWLEDGE GRAPH ===\n")
        
        # People and their diseases
        text_output.append("FAMILY MEMBERS:\n")
        for node in export_data['nodes']:
            if node['type'] == 'person':
                props = node['properties']
                text_output.append(f"- {props['name']} (Age: {props.get('age', 'N/A')})")
                text_output.append(f"  Diseases: {', '.join(props.get('diseases', []))}")
                text_output.append(f"  Dietary Preference: {props.get('dietary_preferences', 'None')}\n")
        
        # Nutritional guidelines
        text_output.append("\nNUTRITIONAL GUIDELINES:\n")
        for node in export_data['nodes']:
            if node['type'] == 'nutrient':
                props = node['properties']
                range_str = f"{props.get('target_min', 'N/A')}-{props.get('target_max', 'N/A')}"
                text_output.append(f"- {props['name']}: {range_str} {props.get('unit', '')}")
                text_output.append(f"  Priority: {props.get('priority')}")
                if props.get('reasons'):
                    text_output.append(f"  Required for: {len(props['reasons'])} conditions\n")
        
        # Food restrictions
        text_output.append("\nFOOD RESTRICTIONS:\n")
        for node in export_data['nodes']:
            if node['type'] == 'restriction':
                props = node['properties']
                text_output.append(f"- {props['restriction_type'].upper()} {props['food_item']}")
                text_output.append(f"  Severity: {props['severity']} | Reason: {props['reason']}")
                text_output.append(f"  Affects: {', '.join(props.get('affects_members', []))}")
                text_output.append(f"  Alternative: {props.get('alternative', 'None')}\n")
        
        # Recommended foods
        text_output.append("\nRECOMMENDED FOODS:\n")
        recommended = [n['properties']['name'] for n in export_data['nodes'] if n['type'] == 'food']
        text_output.append(', '.join(recommended) + "\n")
        
        # Statistics
        text_output.append(f"\nGRAPH STATISTICS:")
        text_output.append(f"- Total Nodes: {export_data['statistics']['total_nodes']}")
        text_output.append(f"- Total Relationships: {export_data['statistics']['total_edges']}")
        text_output.append(f"- Node Types: {export_data['statistics']['node_types']}")
        text_output.append(f"- Relationship Types: {export_data['statistics']['relationship_types']}")
        
        return '\n'.join(text_output)


class GPT4IntegrationEngine:
    """
    GPT-4 Integration for Knowledge Graph Reasoning
    
    Supports multiple LLM providers:
    - OpenAI GPT-4
    - Azure OpenAI
    - Anthropic Claude
    """
    
    def __init__(self, api_key: Optional[str] = None, provider: str = 'openai'):
        """
        Initialize GPT-4 integration
        
        Args:
            api_key: API key (or set OPENAI_API_KEY env var)
            provider: 'openai', 'azure', or 'anthropic'
        """
        self.provider = provider
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.client = None
        self.model = 'gpt-4-turbo-preview'
        
        if not self.api_key and provider == 'openai':
            print("⚠️  No API key found. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client"""
        try:
            if self.provider == 'openai':
                from openai import OpenAI  # type: ignore
                self.client = OpenAI(api_key=self.api_key)
                print(f"✅ OpenAI client initialized (Model: {self.model})")
            
            elif self.provider == 'azure':
                from openai import AzureOpenAI  # type: ignore
                self.client = AzureOpenAI(
                    api_key=self.api_key,
                    api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-02-15-preview'),
                    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT')
                )
                print("✅ Azure OpenAI client initialized")
            
            elif self.provider == 'anthropic':
                import anthropic  # type: ignore
                self.client = anthropic.Anthropic(api_key=self.api_key)
                self.model = 'claude-3-opus-20240229'
                print(f"✅ Anthropic client initialized (Model: {self.model})")
        
        except ImportError as e:
            print(f"⚠️  {self.provider} library not installed. Run: pip install openai (or anthropic)")
            self.client = None
        except Exception as e:
            print(f"⚠️  Failed to initialize {self.provider} client: {e}")
            self.client = None
    
    def test_connection(self) -> Dict[str, Any]:
        """Test LLM API connection"""
        
        if not self.client:
            return {
                'status': 'error',
                'message': 'Client not initialized',
                'api_key_set': bool(self.api_key),
                'provider': self.provider
            }
        
        try:
            if self.provider in ['openai', 'azure']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a nutritional AI assistant."},
                        {"role": "user", "content": "Say 'OK' if you can read this."}
                    ],
                    max_tokens=10,
                    temperature=0
                )
                
                return {
                    'status': 'success',
                    'message': 'Connection successful',
                    'provider': self.provider,
                    'model': self.model,
                    'response': response.choices[0].message.content,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                }
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=10,
                    messages=[
                        {"role": "user", "content": "Say 'OK' if you can read this."}
                    ]
                )
                
                return {
                    'status': 'success',
                    'message': 'Connection successful',
                    'provider': self.provider,
                    'model': self.model,
                    'response': response.content[0].text,
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens
                    }
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'provider': self.provider,
                'model': self.model
            }
    
    def query_knowledge_graph(
        self,
        knowledge_graph_text: str,
        query: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query knowledge graph using GPT-4
        
        Args:
            knowledge_graph_text: Text representation of knowledge graph
            query: User question about nutrition/health
            context: Additional context
            
        Returns:
            LLM response with reasoning
        """
        
        if not self.client:
            return {
                'status': 'error',
                'message': 'LLM client not initialized',
                'answer': None
            }
        
        system_prompt = """You are an expert nutritional AI that analyzes knowledge graphs containing disease information, nutritional guidelines, food restrictions, and recommendations.

Your task is to answer questions based on the provided knowledge graph data. Always:
1. Ground your answers in the specific data from the knowledge graph
2. Cite which diseases or conditions drive specific recommendations
3. Explain conflicts or trade-offs when multiple diseases have different requirements
4. Provide actionable, personalized advice
5. Note any critical restrictions or safety concerns

Be precise, evidence-based, and helpful."""

        user_prompt = f"""KNOWLEDGE GRAPH DATA:
{knowledge_graph_text}

{f'ADDITIONAL CONTEXT: {context}' if context else ''}

USER QUESTION: {query}

Please analyze the knowledge graph and provide a comprehensive answer."""

        try:
            if self.provider in ['openai', 'azure']:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1500,
                    temperature=0.3
                )
                
                return {
                    'status': 'success',
                    'answer': response.choices[0].message.content,
                    'model': self.model,
                    'usage': {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }
                }
            
            elif self.provider == 'anthropic':
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1500,
                    temperature=0.3,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_prompt}
                    ]
                )
                
                return {
                    'status': 'success',
                    'answer': response.content[0].text,
                    'model': self.model,
                    'usage': {
                        'input_tokens': response.usage.input_tokens,
                        'output_tokens': response.usage.output_tokens
                    }
                }
        
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'answer': None
            }
    
    def generate_meal_plan(
        self,
        knowledge_graph_text: str,
        preferences: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate personalized meal plan using GPT-4"""
        
        query = f"""Based on the knowledge graph, generate a detailed 7-day meal plan with:
- Breakfast, lunch, dinner, and 2 snacks per day
- All meals must respect food restrictions (severity: critical and high)
- Optimize for the nutritional guidelines provided
- Include recommended foods where possible
- Provide portion sizes
- Note which family member(s) each meal is optimized for

Preferences:
- Cuisine: {preferences.get('cuisine', 'Any')}
- Meal prep time: {preferences.get('prep_time', 'Medium')}
- Budget: {preferences.get('budget', 'Medium')}
- Cooking skill: {preferences.get('skill_level', 'Intermediate')}"""

        return self.query_knowledge_graph(knowledge_graph_text, query)


# Example usage and testing
if __name__ == "__main__":
    print("="*80)
    print("KNOWLEDGE GRAPH & GPT-4 INTEGRATION TEST")
    print("="*80)
    
    # Create sample family data
    family_members = [
        {
            'name': 'John',
            'age': 55,
            'diseases': ['diabetes_type2', 'hypertension'],
            'dietary_preferences': 'non-vegetarian'
        },
        {
            'name': 'Mary',
            'age': 52,
            'diseases': ['osteoporosis', 'hypothyroidism'],
            'dietary_preferences': 'vegetarian'
        }
    ]
    
    # Run optimization
    print("\n1. Running multi-disease optimization...")
    optimizer = MultiDiseaseOptimizer()
    result = optimizer.optimize_for_family(family_members)
    print(f"   ✅ Optimization complete")
    
    # Build knowledge graph
    print("\n2. Building knowledge graph...")
    kg = NutritionalKnowledgeGraph()
    graph_data = kg.build_from_optimization_result(result, family_members)
    
    print(f"   ✅ Knowledge graph built")
    print(f"   - Nodes: {graph_data['statistics']['total_nodes']}")
    print(f"   - Edges: {graph_data['statistics']['total_edges']}")
    print(f"   - Node types: {graph_data['statistics']['node_types']}")
    
    # Export for LLM
    print("\n3. Exporting knowledge graph for LLM...")
    kg_text = kg.export_for_llm()
    print(f"   ✅ Exported {len(kg_text)} characters")
    
    # Save to file
    with open('knowledge_graph_export.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    print("   ✅ Saved to knowledge_graph_export.json")
    
    with open('knowledge_graph_text.txt', 'w') as f:
        f.write(kg_text)
    print("   ✅ Saved to knowledge_graph_text.txt")
    
    # Test GPT-4 connection
    print("\n4. Testing GPT-4 connection...")
    gpt4 = GPT4IntegrationEngine()
    
    test_result = gpt4.test_connection()
    print(f"\n   Status: {test_result['status']}")
    print(f"   Provider: {test_result.get('provider')}")
    print(f"   Model: {test_result.get('model')}")
    
    if test_result['status'] == 'success':
        print(f"   Response: {test_result.get('response')}")
        print(f"   Tokens used: {test_result.get('usage')}")
        
        # Test knowledge graph query
        print("\n5. Querying knowledge graph with GPT-4...")
        query = "What are the most critical food restrictions for John and why?"
        
        response = gpt4.query_knowledge_graph(kg_text, query)
        
        if response['status'] == 'success':
            print(f"\n   Question: {query}")
            print(f"\n   Answer:\n{response['answer']}")
            print(f"\n   Tokens: {response['usage']}")
        else:
            print(f"   Error: {response.get('message')}")
    else:
        print(f"   Error: {test_result.get('message')}")
        print("\n   ℹ️  To enable GPT-4:")
        print("      1. Install: pip install openai")
        print("      2. Set environment variable: OPENAI_API_KEY=your-key-here")
        print("      3. Or pass api_key to GPT4IntegrationEngine()")
    
    print("\n" + "="*80)
    print("Knowledge graph successfully created!")
    print("Files generated:")
    print("  - knowledge_graph_export.json (structured data)")
    print("  - knowledge_graph_text.txt (LLM-readable format)")
    print("="*80)
