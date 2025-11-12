from flask import Flask, request, jsonify
from ai_chatbot import LocalAIChatbot

app = Flask(__name__)
bot = LocalAIChatbot()

@app.route('/webhook', methods=['POST'])
def webhook():
    """Nhận tin nhắn từ Google Chat"""
    data = request.json
    
    # Lấy text từ Google Chat
    if 'message' in data:
        user_message = data['message'].get('text', '')
        
        # Bot trả lời
        response = bot.chat(user_message)
        
        # Trả về Google Chat
        return jsonify({
            'text': response['answer']
        })
    
    return jsonify({'text': 'OK'})

@app.route('/train', methods=['POST'])
def train_endpoint():
    """API để train bot"""
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    
    if question and answer:
        bot.add_training_pair(question, answer)
        return jsonify({'status': 'success', 'message': 'Đã thêm training data'})
    
    return jsonify({'status': 'error', 'message': 'Thiếu question hoặc answer'}), 400

if __name__ == '__main__':
    print("🚀 Server đang chạy tại http://localhost:5123")
    app.run(host='0.0.0.0', port=5123, debug=True)