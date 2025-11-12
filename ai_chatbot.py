# ai_chatbot.py - AI Chatbot tự build chạy local

import json
import os
import re
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

class LocalAIChatbot:
    def __init__(self, data_file='training_data.json', model_file='chatbot_model.pkl'):
        self.data_file = data_file
        self.model_file = model_file
        self.training_data = []
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            norm='l2',  # ✅ THÊM DÒNG NÀY
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=False
        )
        self.vectors = None
        
        # Load dữ liệu nếu có
        self.load_data()
        
    def load_data(self):
        """Load training data từ file"""
        if os.path.exists(self.data_file):
            with open(self.data_file, 'r', encoding='utf-8') as f:
                self.training_data = json.load(f)
            print(f"✓ Đã load {len(self.training_data)} cặp Q&A")
            
            # Load model đã train
            if os.path.exists(self.model_file):
                with open(self.model_file, 'rb') as f:
                    saved_data = pickle.load(f)
                    self.vectorizer = saved_data['vectorizer']
                    self.vectors = saved_data['vectors']
                print("✓ Đã load model")
        else:
            print("! Chưa có dữ liệu training, bắt đầu từ đầu")
            self.training_data = []
    
    def save_data(self):
        """Lưu training data"""
        with open(self.data_file, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        print("✓ Đã lưu training data")
    
    def save_model(self):
        """Lưu model đã train"""
        if self.vectors is not None:
            with open(self.model_file, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.vectorizer,
                    'vectors': self.vectors
                }, f)
            print("✓ Đã lưu model")
    
    def preprocess_text(self, text):
        """Tiền xử lý text"""
        # Chuyển về lowercase
        text = text.lower()
        # Loại bỏ ký tự đặc biệt
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def add_training_pair(self, question, answer):
        """Thêm cặp Q&A vào training data"""
        pair = {
            'question': question,
            'answer': answer,
            'timestamp': datetime.now().isoformat()
        }
        self.training_data.append(pair)
        self.save_data()
        print(f"✓ Đã thêm: Q: {question[:50]}...")
        
        # Retrain model
        self.train()
    
    def train(self):
        """Train model với dữ liệu hiện có"""
        if not self.training_data:
            print("! Không có dữ liệu để train")
            return
        
        questions = [self.preprocess_text(pair['question']) for pair in self.training_data]
        
        # Tạo TF-IDF vectors
        self.vectors = self.vectorizer.fit_transform(questions)
        self.save_model()
        print(f"✓ Đã train với {len(questions)} câu hỏi")
    
    def find_best_answer(self, question, threshold=0.3):
        """Tìm câu trả lời phù hợp nhất"""
        if not self.training_data or self.vectors is None:
            return None, 0
        
        # Preprocess câu hỏi
        processed_question = self.preprocess_text(question)
        
        # Vector hóa câu hỏi
        question_vector = self.vectorizer.transform([processed_question])
        
        # Tính cosine similarity
        similarities = cosine_similarity(question_vector, self.vectors)[0]
        
        # Tìm best match
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        
        if best_score >= threshold:
            return self.training_data[best_idx]['answer'], best_score
        
        return None, best_score
    
    def chat(self, question):
        """Trả lời câu hỏi"""
        answer, confidence = self.find_best_answer(question)
        
        if answer:
            return {
                'answer': answer,
                'confidence': float(confidence),
                'source': 'trained'
            }
        else:
            return {
                'answer': "Xin lỗi, tôi chưa được train để trả lời câu hỏi này. Bạn có thể dạy tôi không?",
                'confidence': 0,
                'source': 'unknown'
            }
    
    def interactive_mode(self):
        """Chế độ chat tương tác"""
        print("\n" + "="*60)
        print("🤖 AI CHATBOT - Chế độ tương tác")
        print("="*60)
        print("Lệnh:")
        print("  'train' - Thêm training data")
        print("  'stats' - Xem thống kê")
        print("  'exit'  - Thoát")
        print("="*60 + "\n")
        
        while True:
            user_input = input("👤 Bạn: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'exit':
                print("👋 Tạm biệt!")
                break
            
            elif user_input.lower() == 'train':
                self.train_interactive()
                continue
            
            elif user_input.lower() == 'stats':
                self.show_stats()
                continue
            
            # Chat bình thường
            response = self.chat(user_input)
            confidence_bar = "█" * int(response['confidence'] * 10)
            
            print(f"🤖 Bot: {response['answer']}")
            print(f"   📊 Độ tin cậy: [{confidence_bar:<10}] {response['confidence']*100:.1f}%")
            print(f"   🔍 Nguồn: {response['source']}\n")
    
    def train_interactive(self):
        """Training mode tương tác"""
        print("\n--- Chế độ Training ---")
        print("(Nhập 'back' để quay lại)\n")
        
        while True:
            question = input("📝 Câu hỏi: ").strip()
            if question.lower() == 'back':
                break
            if not question:
                continue
            
            answer = input("💡 Câu trả lời: ").strip()
            if answer.lower() == 'back':
                break
            if not answer:
                continue
            
            self.add_training_pair(question, answer)
            
            more = input("\n➕ Thêm cặp khác? (y/n): ").strip().lower()
            if more != 'y':
                break
        
        print("\n✓ Hoàn tất training!\n")
    
    def show_stats(self):
        """Hiển thị thống kê"""
        print("\n" + "="*60)
        print("📊 THỐNG KÊ CHATBOT")
        print("="*60)
        print(f"Tổng số cặp Q&A: {len(self.training_data)}")
        
        if self.training_data:
            print(f"\n📚 5 cặp Q&A gần nhất:")
            for i, pair in enumerate(self.training_data[-5:], 1):
                print(f"\n{i}. Q: {pair['question'][:60]}...")
                print(f"   A: {pair['answer'][:60]}...")
        
        print("="*60 + "\n")
    
    def bulk_import(self, qa_pairs):
        """Import hàng loạt training data"""
        for q, a in qa_pairs:
            self.add_training_pair(q, a)
        print(f"✓ Đã import {len(qa_pairs)} cặp Q&A")


# ============================================
# PHẦN TÍCH HỢP GOOGLE CHAT
# ============================================

class GoogleChatIntegration:
    def __init__(self, webhook_url=None):
        self.webhook_url = webhook_url
    
    def send_message(self, text):
        """Gửi tin nhắn đến Google Chat"""
        if not self.webhook_url:
            print("⚠️ Chưa cấu hình webhook URL")
            return False
        
        import requests
        payload = {"text": text}
        
        try:
            response = requests.post(self.webhook_url, json=payload)
            if response.status_code == 200:
                print("✓ Đã gửi đến Google Chat")
                return True
            else:
                print(f"✗ Lỗi: {response.status_code}")
                return False
        except Exception as e:
            print(f"✗ Lỗi kết nối: {e}")
            return False


# ============================================
# DEMO & USAGE
# ============================================

if __name__ == "__main__":
    # Khởi tạo chatbot
    bot = LocalAIChatbot()
    
    # Nếu chưa có dữ liệu, thêm một số ví dụ
    if len(bot.training_data) == 0:
        print("\n🎓 Import dữ liệu mẫu...")
        sample_data = [
            ("Xin chào", "Xin chào! Tôi là AI chatbot. Tôi có thể giúp gì cho bạn?"),
            ("Bạn tên gì", "Tôi là AI chatbot được tạo bằng Python. Bạn có thể gọi tôi là Bot!"),
            ("Bạn có thể làm gì", "Tôi có thể trả lời câu hỏi dựa trên những gì bạn dạy tôi. Bạn có thể train thêm cho tôi!"),
            ("Thời tiết hôm nay thế nào", "Xin lỗi, tôi không có khả năng kiểm tra thời tiết thời gian thực. Nhưng bạn có thể dạy tôi cách trả lời!"),
            ("Cảm ơn", "Không có chi! Rất vui được giúp đỡ bạn 😊"),
            ("Tạm biệt", "Tạm biệt! Hẹn gặp lại bạn! 👋"),
        ]
        bot.bulk_import(sample_data)
    
    # Chạy chế độ interactive
    bot.interactive_mode()