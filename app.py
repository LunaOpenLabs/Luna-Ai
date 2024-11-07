from flask import Flask, render_template, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from textblob import TextBlob  # For sentiment analysis, translation, and corrections
import re  # Regular expression for text cleanup
import math

app = Flask(__name__)

# Load the pre-trained T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
t5_tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)  # Using the new tokenizer behavior

# Helper function for sentiment analysis
def sentiment_analysis(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity  # Returns polarity

# Helper function for text translation
def translate_text(text, target_lang="en"):
    analysis = TextBlob(text)
    return str(analysis.translate(to=target_lang))

@app.route("/")
def index():
    return render_template("chat.html")  # Using 'chat.html' template

@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        # Get the message from the client
        user_message = request.json.get("message")
        
        # Prepend a task prompt to clarify whether it's text generation or something else
        response = ""

        # Check the type of request based on the command prefix
        if user_message.lower().startswith("summarize:"):
            text = user_message[10:].strip()
            response = generate_summary(text)

        elif user_message.lower().startswith("answer:"):
            question = user_message[7:].strip()
            response = generate_answer(question)

        elif user_message.lower().startswith("sentiment:"):
            text = user_message[10:].strip()
            sentiment_score = sentiment_analysis(text)
            response = f"Sentiment score: {sentiment_score}"

        elif user_message.lower().startswith("translate:"):
            text_to_translate = user_message[10:].strip()
            translated_text = translate_text(text_to_translate)
            response = translated_text

        elif user_message.lower().startswith("classify:"):
            text = user_message[9:].strip()
            response = classify_text(text)

        elif user_message.lower().startswith("generate:"):
            prompt = user_message[9:].strip()
            response = generate_text(prompt)

        # Add more cases for other logics here

        else:
            # Use more specific prompt for conversation to avoid "chat: chat" duplication
            prompt = f"conversation: {user_message}"  # Improved prompt for conversational replies

            # Tokenize and generate response using T5
            inputs = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            generated = t5_model.generate(inputs, max_length=150, num_beams=5, early_stopping=True, 
                                         temperature=0.7, top_p=0.9, do_sample=True)  # Adjusting temperature and top-p for better diversity
            response = t5_tokenizer.decode(generated[0], skip_special_tokens=True)

            # Clean up the response (strip extra tokens like "chat:" if present)
            response = response.replace("chat:", "").strip()

        return jsonify({"response": response})

# Function for text summarization using T5
def generate_summary(text):
    prompt = "summarize: " + text
    inputs = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    summary = t5_model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    return t5_tokenizer.decode(summary[0], skip_special_tokens=True)

# Function for generating answers using T5
def generate_answer(question):
    prompt = "answer: " + question
    inputs = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    answer = t5_model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    return t5_tokenizer.decode(answer[0], skip_special_tokens=True)

# Function for generating text using T5
def generate_text(text):
    prompt = "generate: " + text
    inputs = t5_tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
    generated = t5_model.generate(inputs, max_length=150, num_beams=5, early_stopping=True)
    return t5_tokenizer.decode(generated[0], skip_special_tokens=True)

# Function for text classification (can be expanded with a trained classifier)
def classify_text(text):
    return f"Classified as: {text[:20]}"  # Dummy classification logic for now

# Function for cleaning up text (removing special characters, etc.)
def clean_text(text):
    cleaned_text = re.sub(r"[^A-Za-z0-9 ]+", "", text)
    return cleaned_text

if __name__ == "__main__":
    app.run(debug=True)
