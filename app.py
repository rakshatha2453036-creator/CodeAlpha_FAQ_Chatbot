from flask import Flask, render_template, request, jsonify
import json
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

app = Flask(__name__)

with open("faq_data.json") as f:
    faqs = json.load(f)

questions = [faq["question"] for faq in faqs]
answers = [faq["answer"] for faq in faqs]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"]
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    best_match = similarity.argmax()

    if similarity[0][best_match] < 0.2:
        reply = "Sorry, I couldn't understand your question."
    else:
        reply = answers[best_match]

    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
