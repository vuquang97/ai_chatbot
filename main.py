from flask import Flask, request, jsonify, send_from_directory
from ai_chatbot import LocalAIChatbot
import os

app = Flask(__name__, static_folder='static')
bot = LocalAIChatbot()

# ✅ Route trang chủ - Admin Panel
@app.route('/', methods=['GET'])
def home():
    return send_from_directory('static', 'index.html')

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'status': 'running',
        'message': '🤖 AI Chatbot API is running!',
        'endpoints': {
            'webhook': '/webhook (POST)',
            'train': '/train (POST)',
            'admin': '/admin/data (GET)',
            'update': '/admin/update/:id (PUT)',
            'delete': '/admin/delete/:id (DELETE)'
        }
    })

# ✅ ADMIN API - Lấy tất cả training data
@app.route('/admin/data', methods=['GET'])
def get_all_data():
    """Lấy toàn bộ training data"""
    # Thêm ID cho mỗi item
    data_with_ids = []
    for idx, item in enumerate(bot.training_data):
        item_copy = item.copy()
        item_copy['id'] = idx + 1
        data_with_ids.append(item_copy)
    
    return jsonify({
        'training_data': data_with_ids,
        'total': len(data_with_ids)
    })

# ✅ ADMIN API - Cập nhật training data
@app.route('/admin/update/<int:item_id>', methods=['PUT'])
def update_data(item_id):
    """Cập nhật 1 cặp Q&A"""
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    
    # Convert ID sang index (ID bắt đầu từ 1)
    index = item_id - 1
    
    if 0 <= index < len(bot.training_data):
        bot.training_data[index]['question'] = question
        bot.training_data[index]['answer'] = answer
        bot.save_data()
        bot.train()
        
        return jsonify({
            'status': 'success',
            'message': 'Đã cập nhật'
        })
    
    return jsonify({
        'status': 'error',
        'message': 'ID không tồn tại'
    }), 404

# ✅ ADMIN API - Xóa training data
@app.route('/admin/delete/<int:item_id>', methods=['DELETE'])
def delete_data(item_id):
    """Xóa 1 cặp Q&A"""
    # Convert ID sang index
    index = item_id - 1
    
    if 0 <= index < len(bot.training_data):
        deleted_item = bot.training_data.pop(index)
        bot.save_data()
        bot.train()
        
        return jsonify({
            'status': 'success',
            'message': f'Đã xóa: {deleted_item["question"][:50]}...'
        })
    
    return jsonify({
        'status': 'error',
        'message': 'ID không tồn tại'
    }), 404

# Webhook cho Google Chat
@app.route('/webhook', methods=['POST'])
def webhook():
    """Nhận tin nhắn từ Google Chat"""
    data = request.json
    
    if 'message' in data:
        user_message = data['message'].get('text', '')
        response = bot.chat(user_message)
        
        return jsonify({
            'text': response['answer']
        })
    
    return jsonify({'text': 'OK'})

# Train API
@app.route('/train', methods=['POST'])
def train_endpoint():
    """API để train bot"""
    data = request.json
    question = data.get('question')
    answer = data.get('answer')
    
    if question and answer:
        bot.add_training_pair(question, answer)
        return jsonify({
            'status': 'success',
            'message': 'Đã thêm training data'
        })
    
    return jsonify({
        'status': 'error',
        'message': 'Thiếu question hoặc answer'
    }), 400

# ✅ Enable CORS để gọi API từ frontend
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE')
    return response

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5123))
    app.run(host='0.0.0.0', port=port)