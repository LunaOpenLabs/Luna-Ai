from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration

app = Flask(__name__)

# Load the pre-trained T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)  # Using the new tokenizer behavior

@app.route("/")
def index():
    return render_template("chat.html")  # Using 'chat.html' template

@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        # Get the message from the client
        user_message = request.json.get("message")
        
        # Prepend a task prompt to clarify whether it's text generation or something else
        prompt = "chat: " + user_message  # Use "chat:" for conversational replies
        
        # Tokenize and generate response using T5
        inputs = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
        response = t5_model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
        
        # Decode and return the generated response
        bot_reply = t5_tokenizer.decode(response[0], skip_special_tokens=True)
        return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True)
