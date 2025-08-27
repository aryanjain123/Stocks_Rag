from pinecone import Pinecone
import os
from dotenv import load_dotenv
import requests

# Try to import OpenAI, but handle if it's not available
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    print("Warning: OpenAI module not found. Answer generation will be disabled.")
    print("To enable answer generation, run: pip install openai")
    OPENAI_AVAILABLE = False

load_dotenv()

# ================================
# Setup Pinecone
# ================================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "ollama-embeddings2"
index = pc.Index(index_name)

# ================================
# Setup OpenRouter (if available)
# ================================
if OPENAI_AVAILABLE:
    OPENROUTER_API_KEY = "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"
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

# ================================
# Ollama embeddings
# ================================
def embed_with_ollama(text, model="bge-m3"):
    url = "http://localhost:11434/api/embed"
    payload = {"model": model, "input": text}

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        # Extract embeddings from the new API response format
        embeddings = data.get("embeddings", [])
        if embeddings:
            return embeddings[0]  # Return the first (and only) embedding
        else:
            print("Warning: No embeddings found in response")
            return []
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ollama. Please ensure Ollama is running at http://localhost:11434")
        return []
    except requests.exceptions.Timeout:
        print("Error: Request to Ollama timed out")
        return []
    except Exception as e:
        print(f"Error: Unexpected error generating embedding: {e}")
        return []

# ================================
# Generate Answer with OpenRouter
# ================================
def generate_answer(query, contexts):
    if not OPENAI_AVAILABLE:
        return None
        
    # Prepare context text
    context_text = "\n\n".join([f"Document {i+1} (Relevance: {ctx['score']:.2f}):\n{ctx['text']}" 
                                for i, ctx in enumerate(contexts)])
    
    # Create prompt
    prompt = f"""
    You are a helpful assistant answering questions based on the provided documents.
    
    Question: {query}
    
    Relevant Documents:
    {context_text}
    
    Instructions:
    - Answer the question based ONLY on the information provided in the documents above
    - If the documents don't contain relevant information, say so
    - Reference specific documents when making statements (e.g., "According to Document 1...")
    - Provide a comprehensive answer that directly addresses the question
    - Keep your answer focused and concise
    """
    
    try:
        completion = client.chat.completions.create(
            model="deepseek/deepseek-r1-0528-qwen3-8b:free",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        if "429" in str(e):
            print("Rate limit exceeded. Please wait before making another request or consider upgrading your OpenRouter account.")
        else:
            print(f"Error: Error generating answer: {e}")
        return None

# ================================
# Query Function
# ================================
def query_index(query_text, top_k=20):
    print(f"Generating embedding for query: '{query_text[:50]}...'")
    
    # Generate embedding for the query
    query_embedding = embed_with_ollama(query_text)
    
    if not query_embedding:
        print("Error: Failed to generate embedding for query")
        return []
    
    print(f"Query embedding generated with {len(query_embedding)} dimensions")
    
    # Query Pinecone
    print("Querying Pinecone...")
    
    # Get all available namespaces
    try:
        stats = index.describe_index_stats()
        namespaces = list(stats.get('namespaces', {}).keys())
        print(f"Found namespaces: {namespaces}")
    except Exception as e:
        print(f"Warning: Could not get namespaces: {e}")
        namespaces = ['pfc1', 'pfc2']  # Default to known namespaces
    
    all_matches = []
    
    # Query each namespace separately and combine results
    for namespace in namespaces:
        print(f"Querying namespace: {namespace}")
        results = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            namespace=namespace
        )
        
        namespace_matches = results.get('matches', [])
        print(f"Namespace {namespace} returned {len(namespace_matches)} matches")
        all_matches.extend(namespace_matches)
    
    # Sort all matches by score (descending) and take top_k
    all_matches.sort(key=lambda x: x.get('score', 0), reverse=True)
    top_matches = all_matches[:top_k]
    
    print(f"Combined results: {len(top_matches)} matches")
    
    # Extract and return the text from results
    texts = []
    for match in top_matches:
        # Handle encoding issues in text
        raw_text = match.get('metadata', {}).get('text', '')
        try:
            # Try to clean up the text encoding
            clean_text = raw_text.encode('utf-8', errors='replace').decode('utf-8')
        except:
            clean_text = "[Text unavailable due to encoding issues]"
            
        texts.append({
            'text': clean_text,
            'score': match.get('score', 0),
            'id': match.get('id', 'unknown'),
            'namespace': match.get('namespace', 'unknown')
        })
    
    return texts

# ================================
# Utility Functions
# ================================
def view_index_stats():
    """Display statistics about the Pinecone index"""
    try:
        stats = index.describe_index_stats()
        print("Pinecone Index Statistics:")
        print(f"   Index Dimension: {stats.get('dimension', 'Unknown')}")
        print(f"   Index Count: {stats.get('total_vector_count', 0)}")
        print(f"   Namespaces: {list(stats.get('namespaces', {}).keys())}")
        for namespace, info in stats.get('namespaces', {}).items():
            print(f"     - {namespace}: {info.get('vector_count', 0)} vectors")
    except Exception as e:
        print(f"Error: Error getting index stats: {e}")

def view_sample_data():
    """Fetch and display a sample of data from the index"""
    try:
        # Try to fetch a sample vector
        sample = index.fetch(ids=['pfc1.pdf_0'])
        if sample.get('vectors'):
            print("Sample data found:")
            for id, vector_data in sample['vectors'].items():
                print(f"   ID: {id}")
                print(f"   Metadata: {vector_data.get('metadata', {})}")
        else:
            print("No sample data found with ID 'pfc1.pdf_0'")
            print("Trying to fetch any data...")
            # Try to query with a broad search
            results = index.query(
                vector=[0.0] * 1024,  # Zero vector
                top_k=1,
                include_metadata=True
            )
            if results.get('matches'):
                print("Found data with zero vector query")
                match = results['matches'][0]
                print(f"   ID: {match.get('id')}")
                print(f"   Score: {match.get('score')}")
                print(f"   Metadata: {match.get('metadata', {})}")
            else:
                print("No data found in index")
    except Exception as e:
        print(f"Error: Error fetching sample data: {e}")

# ================================
# Main Interaction Loop
# ================================
if __name__ == "__main__":
    print("RAG Query Interface")
    print("Type 'exit' to quit")
    print("Type 'stats' to view index statistics")
    print("Type 'sample' to view sample data")
    if not OPENAI_AVAILABLE:
        print("Answer generation disabled - install 'openai' package to enable")
    print()
    
    try:
        while True:
            try:
                query = input("Enter your query: ").strip()
            except EOFError:
                print("\nGoodbye!")
                break
                
            if query.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
                
            if query.lower() == 'stats':
                view_index_stats()
                continue
                
            if query.lower() == 'sample':
                view_sample_data()
                continue
                
            if not query:
                continue
                
            print(f"\nSearching for: {query}")
            results = query_index(query, top_k=5)  # Using 5 for better context window
            
            if not results:
                print("No relevant documents found\n")
                continue
                
            print(f"\nFound {len(results)} relevant documents:")
            for i, result in enumerate(results, 1):
                print(f"   {i}. Score: {result['score']:.4f} | From: {result['namespace']} | ID: {result['id']}")
            
            # Only try to generate answer if OpenAI is available
            if OPENAI_AVAILABLE:
                print("\nGenerating comprehensive answer...")
                answer = generate_answer(query, results)
                
                if answer:
                    # Handle encoding issues in answer
                    try:
                        clean_answer = answer.encode('utf-8', errors='replace').decode('utf-8')
                        print(f"\nAnswer:\n{clean_answer}\n")
                    except:
                        print("\nAnswer: [Unable to display due to encoding issues]\n")
                else:
                    print("Failed to generate answer\n")
            else:
                print("\nRelevant document excerpts:")
                for i, result in enumerate(results, 1):
                    print(f"\n--- Document {i} (Score: {result['score']:.4f}) ---")
                    # Handle encoding issues in text
                    try:
                        clean_text = result['text'].encode('utf-8', errors='replace').decode('utf-8')
                        print(clean_text)
                    except:
                        print("[Text unavailable due to encoding issues]")
                print()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")

