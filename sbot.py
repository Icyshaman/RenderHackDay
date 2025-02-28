from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai.foundation_models import ModelInference
import spacy
from flask import Flask, request, jsonify
import os

class SBot:
    def __init__(self):
        self.app = Flask(__name__)
        self.setup_routes()
        self.nlp = spacy.load("en_core_web_sm")
        API_KEY = os.getenv("API_KEY")
        PROJECT_ID = os.getenv("PROJECT_ID")
        REGION_URL = os.getenv("REGION_URL")
        credentials = {
            "apikey": API_KEY,
            "url": REGION_URL
        }
        self.api_client = APIClient(credentials)
        self.api_client.set.default_project(PROJECT_ID)
        self.model = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            credentials=credentials,
            project_id=PROJECT_ID
        )
        self.labels = ["create_api", "list_api", "list_api_project", "list_project", "deploy_api", "irrelevant"]
        self.question_words = {"what", "when", "where", "who", "why", "how", "is", "are", "do", "does", "did", "can", "could", "should", "would", "will", "shall"}

    def is_valid_sentence_length(self, doc):
        if len(list(doc.sents)) > 1:
            return False
        return True
        
    def is_question_or_declarative(self, doc, user_input):
        if doc[0].text in self.question_words or user_input.strip().endswith("?"):
            return True
        return False
    
    def classify_intent(self, user_input):
        prompt = f"Classify this intent: '{user_input}'. Options: {', '.join(self.labels)}. Return only the intent."
        parameters = {
            "decoding_method": "sample",
            "temperature": 0.5,
            "top_k": 60,
            "top_p": 0.8,
            "max_new_tokens": 20
        }
        response = self.model.generate(
            prompt=prompt, params=parameters
        )
        return response["results"][0]["generated_text"].strip()

    def setup_routes(self):
        @self.app.route("/get-intnet", methods=["POST"])
        def get_intent():
            data = request.json
            if "sentence" not in data:
                return jsonify({"error": "No sentence provided"}), 400
            user_input = data["sentence"].lower()
            if not user_input or not user_input.strip():
                return jsonify({"error": "No sentence provided"}), 400
            doc = self.nlp(user_input)
            if not self.is_valid_sentence_length(doc):
                return jsonify({"error": "More than 1 sentence is provided"}), 400
            if self.is_question_or_declarative(doc, user_input):
                return jsonify({"intent": self.labels[-1]}), 200
            intent = self.classify_intent(user_input)
            return jsonify({"type": intent}), 200
    
    def run(self, host="0.0.0.0", port=5000, debug=False):
        self.app.run(host=host, port=port, debug=debug)

# Run the chatbot
if __name__ == "__main__":
    sbot = SBot()
    sbot.run()