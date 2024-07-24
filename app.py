import os
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login, HfApi, HfFolder

app = Flask(__name__)

# Set Hugging Face API token (for demonstration purposes)
HF_API_TOKEN = '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'

# Authenticate with Hugging Face
login(HF_API_TOKEN)

# Load the model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.json
        lesson_title = data.get('lesson_title')
        student_class = data.get('student_class')

        if not lesson_title or not student_class:
            return jsonify({'error': 'Missing lesson_title or student_class in request'}), 400

        # Generate a response based on the lesson_title and student_class
        inputs = tokenizer(f"Lesson: {lesson_title}, Class: {student_class}", return_tensors="pt")
        outputs = model.generate(inputs['input_ids'], max_length=1000, temperature=0.5)
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({'response': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    port = int(os.getenv('PORT', 8000))  # Use PORT environment variable or default to 8000
    app.run(host='0.0.0.0', port=port)
