import json
import os
import sys
import time

import anthropic
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains.conversation.base import ConversationChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, UnstructuredWordDocumentLoader, TextLoader
from streamlit_agraph import agraph, Node as ANode, Edge as AEdge, Config
from typing import List
import base64
import pymupdf
import json

load_dotenv()

ENVIRONMENT = os.environ.get("ENVIRONMENT")
BOWHEAD_ANTHROPIC_AI_KEY = os.environ.get("BOWHEAD_ANTHROPIC_AI_KEY")

if ENVIRONMENT == 'prod':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_chroma import Chroma

BOWHEAD_OPEN_AI_KEY = os.environ.get("BOWHEAD_OPEN_AI_KEY")


def get_conversation_memory(messages, model):
    return ConversationChain(
        llm=model,
        memory=ConversationSummaryBufferMemory(
            llm=model,
            chat_memory=messages,
            return_messages=True,
        )
    )


def stream_data(data):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)


def load_and_split_webpage(website_url):
    loader = WebBaseLoader(
        web_paths=(website_url,)
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    return splits


def load_and_split_documents(
        files,
        separators=["\n\n\n", "\n\n"],
        chunk_size=1000
):
    docs = []

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=300,
        length_function=len,
        is_separator_regex=False,
    )

    for uploaded_file in files:
        temp_file = f"./{uploaded_file.name}"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())

            file_extension = uploaded_file.name.split(".")[-1]
            if file_extension == "pdf":
                loader = PyPDFLoader(temp_file)
            elif file_extension == "docx":
                loader = UnstructuredWordDocumentLoader(temp_file)
            else:
                loader = TextLoader(temp_file, encoding="utf-8")

            # loader = UnstructuredFileLoader(temp_file, strategy="fast")
            docs.extend(
                loader.load_and_split(text_splitter=text_splitter),
            )

            os.remove(temp_file)

    return docs


def generate_embeddings(docs):
    collection_name = 'multimate'
    client = Chroma(collection_name=collection_name)
    client.delete_collection()

    db = client.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(openai_api_key=BOWHEAD_OPEN_AI_KEY),
    )
    return db


def create_knowledge_graph_embeddings(docs):
    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key='sk-proj-eRdnDmqyDy9VCllbTaIyT3BlbkFJyLPYzaL1Or8YB2zoxjRQ')

    # Store the embeddings in Neo4j
    return Neo4jVector.from_documents(
        docs,
        embeddings,
        username="neo4j",
        url="neo4j+s://8c69bbed.databases.neo4j.io",
        password="la1_QihBk8goh64ZiVTd7u005uftVUEK5pfyeHTT7SM",
        index_name="document_index"
    )


class Node:
    def __init__(self, id: str, name: str, type: str):
        self.id = id
        self.name = name
        self.type = type

    def __repr__(self):
        return f"Node(id='{self.id}', name='{self.name}', type='{self.type}')"


class Relationship:
    def __init__(self, source: Node, target: Node, type: str):
        self.source = source
        self.target = target
        self.type = type

    def __repr__(self):
        return f"Relationship(source={self.source}, target={self.target}, type='{self.type}')"


def generate_knowledge_graph(files, graph):
    nodes = []
    relationships = []
    for uploaded_file in files:
        temp_file = f"./{uploaded_file.name}"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            doc = pymupdf.open(temp_file)
            for page in doc:
                pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
                image_bytes = pix.tobytes("png")
                image_data = base64.b64encode(image_bytes).decode("utf-8")
                prompt = create_comprehensive_prompt()
                client = anthropic.Anthropic(api_key=BOWHEAD_ANTHROPIC_AI_KEY)
                message = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=4096,
                    system=prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": image_data,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": 'Tip: Make sure to answer in the correct format and do not include any explanations'
                                }
                            ],
                        }
                    ],
                )
                print(message.content[0].text)
                results = parse_results(message.content[0].text)
                nodes.extend(results["nodes"])
                relationships.extend(results["relationships"])
                cypher_statement = generate_cypher_statements(results["nodes"], results["relationships"])
                graph.query(cypher_statement)

    return nodes, relationships


def create_comprehensive_prompt():
    prompt_template = """
        # Knowledge Graph Instructions
        ## 1. Overview
        You are an advanced algorithm designed for extracting information in structured formats to build a knowledge graph. 
        Your goal is to analyze the provided image and extract as much information as possible without sacrificing accuracy. 
        The image may contain text, diagrams, and other visual elements. 

        - **Nodes** represent entities and concepts.
        - **Relationships** represent connections between entities or concepts.

        ## 2. Labeling Nodes
        - **Consistency**: Use consistent and elementary types for node labels.
            - For example, label an entity representing a person as **'person'**. 
            Avoid using specific terms like 'mathematician' or 'scientist'.
        - **Node IDs**: Use names or human-readable identifiers found in the text. Avoid using integers as node IDs.
        - **Extract All Relevant Entities**: Identify and extract all relevant entities from the text and diagrams.

        ## 3. Identifying Relationships
        - **Consistency**: Use general and timeless relationship types. 
            - For example, use **'PROFESSOR'** instead of **'BECAME_PROFESSOR'**.
        - **Relationship Clarity**: Ensure that relationships clearly define the connections between entities.

        ## 4. Coreference Resolution
        - **Maintain Entity Consistency**: Ensure consistency in entity references. 
            - If an entity like "John Doe" is mentioned multiple times with different names or pronouns (e.g., "Joe", "he"), use the most complete identifier ("John Doe") throughout the knowledge graph.

        ## 5. Strict Compliance
        Adhere strictly to these rules. Non-compliance will result in termination.

        ## 6. Image for Processing
        Analyze the image below and extract nodes and relationships. 

        ## 7. Output Format
        Output the extracted information in the following format and do not prepend or append the extracted information with any sentence:
        {"nodes": ["id": "","name": "","type": ""}],"relationships": [{"source" {"id": "","name": "","type": ""}, "target": {"id": "","name": "","type": ""},"type": ""}]}
        Where id is unique numeric value which can never be regenerated. Ensure that in a situation where you run out of tokens terminate the 
        output properly such that it can be parsed correctly.
        
    """

    return prompt_template


def parse_results(result_json: str) -> dict:
    parsed_result = json.loads(result_json)

    nodes = [Node(node['id'], node['name'], node['type']) for node in parsed_result['nodes']]
    relationships = [
        Relationship(
            Node(rel['source']['id'], rel['source']['name'], rel['source']['type']),
            Node(rel['target']['id'], rel['source']['name'], rel['target']['type']),
            rel['type']
        ) for rel in parsed_result['relationships']
    ]

    return {
        "nodes": nodes,
        "relationships": relationships
    }


def generate_cypher_statements(nodes, relationships) -> str:
    # Create nodes
    node_statements = []
    for node in nodes:
        node_statements.append(f"CREATE (n{node.id}: {node.type} {{name: '{node.name}'}})")

    # Create relationships
    relationship_statements = []
    for relationship in relationships:
        source = relationship.source
        target = relationship.target
        relationship_statements.append(f"CREATE (n{source.id})-[:{relationship.type}]->(n{target.id})")

    # Combine all statements into one script
    cypher_script = "\n".join(node_statements + relationship_statements)

    return cypher_script


def generate_graph(nodes, relationships):
    agraph_nodes = [ANode(id=node.id, label=node.name) for node in nodes]

    agraph_edges = [
        AEdge(source=rel.source.id, target=rel.target.id, label=rel.type)
        for rel in relationships
    ]

    config = Config(
        width=800,
        height=600,
        directed=True,
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=True,
        node={'labelProperty': 'label'},
        link={'labelProperty': 'label', 'renderLabel': True},
    )

    return agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)


class KnowledgeGraphBuilder:
    def __init__(self, db_url: str, db_username: str, db_password: str, llm_api_key: str):
        self.knowledge_graph = Neo4jGraph(url=db_url, username=db_username, password=db_password)
        self.client = anthropic.Anthropic(api_key=llm_api_key)
        self.all_nodes = []
        self.all_relationships = []
        self.last_id = 0

    def pdf_to_image(self, files) -> List[str]:
        base64strings = []

        for uploaded_file in files:
            temp_file = f"./{uploaded_file.name}"
            with open(temp_file, "wb") as file:
                file.write(uploaded_file.getvalue())
                doc = pymupdf.open(temp_file)
                for page in doc:
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
                    base64strings.append(base64.b64encode(pix.tobytes("png")).decode())
        return base64strings

    def geenrate_prompt(self):
        prompt_template = """
            # Knowledge Graph Instructions
            ## 1. Overview
            You are an advanced algorithm designed for extracting information in structured formats to build a knowledge graph. 
            Your goal is to analyze the provided image and extract as much information as possible without sacrificing accuracy. 
            The image may contain text, diagrams, and other visual elements. 

            - **Nodes** represent entities and concepts.
            - **Relationships** represent connections between entities or concepts.

            ## 2. Labeling Nodes
            - **Consistency**: Use consistent and elementary types for node labels.
                - For example, label an entity representing a person as **'person'**. 
                Avoid using specific terms like 'mathematician' or 'scientist'.
            - **Node IDs**: Use names or human-readable identifiers found in the text. Avoid using integers as node IDs.
            - **Extract All Relevant Entities**: Identify and extract all relevant entities from the text and diagrams.

            ## 3. Identifying Relationships
            - **Consistency**: Use general and timeless relationship types. 
                - For example, use **'PROFESSOR'** instead of **'BECAME_PROFESSOR'**.
            - **Relationship Clarity**: Ensure that relationships clearly define the connections between entities.

            ## 4. Coreference Resolution
            - **Maintain Entity Consistency**: Ensure consistency in entity references. 
                - If an entity like "John Doe" is mentioned multiple times with different names or pronouns (e.g., "Joe", "he"), use the most complete identifier ("John Doe") throughout the knowledge graph.

            ## 5. Strict Compliance
            Adhere strictly to these rules. Non-compliance will result in termination.

            ## 6. Image for Processing
            Analyze the image below and extract nodes and relationships. 

            ## 7. Output Format
            Output the extracted information in the following format and do not prepend or append the extracted information with any sentence:
            {"nodes": ["id": "","name": "","type": ""}],"relationships": [{"source" {"id": "","name": "","type": ""}, "target": {"id": "","name": "","type": ""},"type": ""}]}
            Where id is unique numeric value which can never be regenerated. Ensure that in a situation where you run out of tokens terminate the 
            output properly such that it can be parsed correctly.

        """
        return prompt_template

    def parse_graph(self, result_json: str):
        parsed_result = json.loads(result_json)

        nodes = []
        for node in parsed_result['nodes']:
            self.last_id += 1
            nodes.append(Node(self.last_id, node['name'], node['type']))

        relationships = []
        for rel in parsed_result['relationships']:
            source_node = next(
                node for node in nodes if node.name == rel['source']['name'] and node.type == rel['source']['type'])
            target_node = next(
                node for node in nodes if node.name == rel['target']['name'] and node.type == rel['target']['type'])
            relationships.append(Relationship(source_node, target_node, rel['type']))

        self.all_nodes.extend(nodes)
        self.all_relationships.extend(relationships)

    def merge_nodes(self):
        unique_nodes = {}

        for node in self.all_nodes:
            if node.name in unique_nodes:
                if node.id < unique_nodes[node.name].id:
                    unique_nodes[node.name] = node
            else:
                unique_nodes[node.name] = node

        id_mapping = {node.id: unique_nodes[node.name] for node in self.all_nodes}

        updated_relationships = []
        for rel in self.all_relationships:
            new_source = id_mapping[rel.source.id]
            new_target = id_mapping[rel.target.id]
            updated_relationships.append(Relationship(new_source, new_target, rel.type))

        self.all_nodes, self.all_relationships = list(unique_nodes.values()), updated_relationships

    def generate_cypher_statements(self) -> str:
        node_statements = []
        for node in self.all_nodes:
            node_statements.append(f"CREATE (n{node.id}: {node.type} {{name: '{node.name}'}})")

        relationship_statements = []
        for relationship in self.all_relationships:
            source = relationship.source
            target = relationship.target
            relationship_statements.append(f"CREATE (n{source.id})-[:{relationship.type}]->(n{target.id})")

        return "\n".join(node_statements + relationship_statements)

    def process_images(self, images: List[str]):
        for image in images:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4096,
                system=self.geenrate_prompt(),
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": image,
                                },
                            },
                            {
                                "type": "text",
                                "text": 'Tip: Make sure to answer in the correct format and do not include any explanations'
                            }
                        ],
                    }
                ],
            )
            self.parse_graph(message.content[0].text)

    def build_knowledge_graph(self, pdf_file: str):
        self.process_images(self.pdf_to_image(pdf_file))
        self.merge_nodes()
        self.knowledge_graph.query(self.generate_cypher_statements())
