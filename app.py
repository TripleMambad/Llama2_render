from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import torch
from flask import Flask, request, jsonify

app = Flask(__name__)

# Authenticate to Hugging Face
login("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")

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to("cuda")

@app.route('/generate_lesson', methods=['POST'])
def generate_lesson():
    data = request.json
    title = data.get('title', '')
    student_class = data.get('class', '')

    prompt = f"Generate a lesson plan for the title: {title} for class: {student_class}"

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs.input_ids, max_length=1000)
    lesson_plan = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return jsonify({"lesson_plan": lesson_plan})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
