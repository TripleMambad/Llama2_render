from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
from huggingface_hub import login

app = Flask(__name__)

# Set Hugging Face token
hf_token = os.getenv('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')  # Use the environment variable name
if hf_token:
    login(token=hf_token)
else:
    raise ValueError("Hugging Face API token is not set. Please set the 'HF_API_TOKEN' environment variable.")

model_name = "meta-llama/Llama-2-7b-chat-hf"

# Load model and tokenizer
try:
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        lesson_title = data.get('lesson_title', '')
        student_class = data.get('student_class', '')
        input_text = f"Lesson Title: {lesson_title}\n Class: {student_class}\n"
        
        inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_length=860, temperature=0.5)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        return jsonify({"lesson_plan": response[0]})  # Return a well-structured JSON object
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))  # Use the PORT environment variable, default to 10000 if not set
    app.run(host='0.0.0.0', port=port, debug=True)  # Enable debug mode for development
