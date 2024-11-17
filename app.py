import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
#pip install transformers torch gradio

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

def chat_with_llama(message, history):
    # Format the input for the model
    prompt = f"Human: {message}\nAssistant:"
    
    # Generate a response
    response = generator(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    
    # Extract the assistant's response
    assistant_response = response.split("Assistant:")[-1].strip()
    
    return assistant_response

# Create the Gradio interface
iface = gr.ChatInterface(
    chat_with_llama,
    title="Llama 2 Chatbot",
    description="Chat with Llama 2 AI model",
)

# Launch the interface
iface.launch()

