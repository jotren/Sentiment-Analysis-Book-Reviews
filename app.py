from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Initialize Flask app
app = Flask(__name__)

# Load the model and tokenizer once (during app initialization)
model_path = "./models/emotion-english-roberta-large"  # Path to the locally cloned model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
emotion_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Define the route for batch emotion analysis
@app.route('/analyze_emotions', methods=['POST'])
def analyze_emotions():
    data = request.json
    texts = data.get('texts', [])

    if not texts:
        return jsonify({'error': 'No texts provided'}), 400

    # Analyze the emotions for each text
    results = []
    for text in texts:
        result = emotion_pipeline(text, truncation=True, max_length=512, top_k=None)
        # Append the list of (label, score) for each text
        results.append([(res['label'], res['score']) for res in result])
    
    # Return the results as JSON
    return jsonify({'emotion_results': results})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
