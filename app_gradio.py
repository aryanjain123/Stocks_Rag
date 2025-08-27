import gradio as gr
from query import query_index, generate_answer, OPENAI_AVAILABLE

def rag_query(query_text):
    """Process a query and return the response"""
    if not query_text.strip():
        return "Please enter a query."
    
    # Get results from the index
    results = query_index(query_text, top_k=5)
    
    if not results:
        return "No relevant documents found."
    
    # Generate answer if OpenAI is available
    if OPENAI_AVAILABLE:
        answer = generate_answer(query_text, results)
        if answer:
            return answer
        else:
            return "Failed to generate answer."
    else:
        # If OpenAI is not available, return the top result
        return results[0]['text']

# Create the Gradio interface
with gr.Blocks(title="RAG Query Interface") as app:
    gr.Markdown("# RAG Query Interface")
    gr.Markdown("Enter your query below to search the documents and get a response.")
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Enter your query:",
                placeholder="Type your question here...",
                lines=3
            )
            submit_btn = gr.Button("Submit Query", variant="primary")
        
        with gr.Column():
            response_output = gr.Textbox(
                label="Response:",
                interactive=False,
                lines=10
            )
    
    # Example queries
    gr.Examples(
        examples=[
            "What is PFC's growth strategy?",
            "How has Stallion performed recently?",
            "What did management say about future outlook about stallion?"
        ],
        inputs=query_input
    )
    
    # Connect the components
    submit_btn.click(
        fn=rag_query,
        inputs=query_input,
        outputs=response_output
    )
    
    # Allow Enter key to submit
    query_input.submit(
        fn=rag_query,
        inputs=query_input,
        outputs=response_output
    )

if __name__ == "__main__":
    app.launch()