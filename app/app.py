from flask import Flask, render_template, request, jsonify
# Import  chatbot logic here
from chatbot_module import Chatbot

app = Flask(__name__)
chatbot = Chatbot()  #  chatbot logic is encapsulated in a Chatbot class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    # Get the chatbot response. 
    response = chatbot.ask(question)
    return jsonify(answer=response)

if __name__ == '__main__':
    app.run(debug=True)
