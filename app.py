from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Hugging Face API token
API_TOKEN = "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"

# Authenticate with Hugging Face API
from huggingface_hub import login
login(API_TOKEN)

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=API_TOKEN)
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=API_TOKEN)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    lesson_title = data.get("lesson_title")
    student_class = data.get("student_class")

    prompt = f"Create a lesson titled '{lesson_title}' for a class '{student_class}':"

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=1000)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"response": response})

if __name__ == '__main__'
