# Thư viện chuẩn
import json
import os
import random
import pickle
import re
import logging
from datetime import datetime, timedelta
from vietnam_cities import VIETNAM_CITIES

# Import từ điển dịch thuật từ file weather_conditions_vi.py
from weather_conditions_vi import WEATHER_CONDITIONS_VI

# Thư viện bên thứ ba
import numpy as np
import nltk
import pytz
import requests
import speech_recognition as sr
from sklearn.preprocessing import LabelEncoder

# Thư viện cho PhoBERT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải dữ liệu cần thiết cho NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

class ChatbotDataset(Dataset):
    """Dataset class để chuẩn bị dữ liệu cho PhoBERT."""
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        label = self.labels[idx]

        # Tokenize câu
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DeepLearningChatbot:
    def __init__(self, intents_file="chatbot_intents_vi_augmented.json", model_dir="phobert_chatbot_model"):
        """
        Khởi tạo chatbot học sâu với PhoBERT.

        Args:
            intents_file (str): Đường dẫn đến file JSON chứa dữ liệu intents.
            model_dir (str): Thư mục để lưu mô hình PhoBERT đã fine-tune.
        """
        self.intents_file = intents_file
        self.model_dir = model_dir
        self.weather_api_key = os.getenv("WEATHERAPI_KEY")  # Lấy API key từ biến môi trường

        # Kiểm tra nếu API key không tồn tại
        if not self.weather_api_key:
            raise ValueError("Không tìm thấy API key cho WeatherAPI. Vui lòng đặt biến môi trường WEATHERAPI_KEY.")

        # Tải PhoBERT tokenizer và mô hình
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Tải hoặc huấn luyện mô hình
        if os.path.exists(self.model_dir):
            self.load_model()
        else:
            self.train_model()

    def load_model(self):
        """Tải mô hình PhoBERT đã fine-tune."""
        logging.info("Đang tải mô hình PhoBERT...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.to(self.device)
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        self.classes = sorted(list(set(intent["tag"] for intent in self.intents["intents"])))
        logging.info("Tải mô hình PhoBERT thành công.")

    def train_model(self):
        """Huấn luyện mô hình PhoBERT trên dữ liệu intents."""
        logging.info("Bắt đầu huấn luyện mô hình PhoBERT...")

        # Tải file intents
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)

        # Chuẩn bị dữ liệu huấn luyện
        sentences = []
        labels = []
        self.classes = sorted(list(set(intent["tag"] for intent in self.intents["intents"])))
        label_encoder = LabelEncoder()
        label_encoder.fit(self.classes)

        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                sentences.append(pattern)
                labels.append(label_encoder.transform([intent["tag"]])[0])

        # Tạo dataset
        dataset = ChatbotDataset(sentences, labels, self.tokenizer)

        # Tải mô hình PhoBERT
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "vinai/phobert-base",
            num_labels=len(self.classes)
        )
        self.model.to(self.device)

        # Cấu hình huấn luyện
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
        )

        # Tạo Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        # Huấn luyện mô hình
        trainer.train()

        # Lưu mô hình
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        logging.info("Huấn luyện mô hình PhoBERT hoàn tất và đã lưu.")

    def fine_tune_on_new_data(self, new_sentences, new_labels):
        """Fine-tune mô hình PhoBERT trên dữ liệu mới."""
        logging.info("Bắt đầu fine-tune mô hình PhoBERT trên dữ liệu mới...")

        # Tạo dataset từ dữ liệu mới
        dataset = ChatbotDataset(new_sentences, new_labels, self.tokenizer)

        # Cấu hình fine-tune
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=1,  # Chỉ fine-tune 1 epoch để nhanh
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_steps=500,
            save_total_limit=2,
        )

        # Tạo Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
        )

        # Fine-tune mô hình
        trainer.train()

        # Lưu mô hình
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)
        logging.info("Fine-tune mô hình PhoBERT hoàn tất và đã lưu.")

    def predict_class(self, sentence):
        """
        Dự đoán ý định của câu bằng PhoBERT.

        Args:
            sentence (str): Câu hỏi từ người dùng.

        Returns:
            list: Danh sách các ý định dự đoán với xác suất.
        """
        self.model.eval()
        encoding = self.tokenizer(
            sentence,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        results = [[i, prob] for i, prob in enumerate(probs)]
        results.sort(key=lambda x: x[1], reverse=True)
        return [{"intent": self.classes[r[0]], "probability": str(r[1])} for r in results]

    def get_response(self, intents_list, intents_json, message):
        """
        Lấy phản hồi dựa trên ý định dự đoán.

        Args:
            intents_list (list): Danh sách các ý định dự đoán từ mô hình.
            intents_json (dict): Dữ liệu intents từ file JSON.
            message (str): Tin nhắn gốc từ người dùng.

        Returns:
            str: Phản hồi của bot.
        """
        if not intents_list:
            return "Tôi không hiểu ý bạn. Bạn có thể diễn đạt lại được không?"

        tag = intents_list[0]["intent"]
        list_of_intents = intents_json["intents"]

        for intent in list_of_intents:
            if intent["tag"] == tag:
                response = random.choice(intent["responses"])

                # Xử lý các intent đặc biệt
                if tag == "hoi_gio":
                    response = self._handle_hoi_gio(response)
                elif tag == "hoi_ngay_mai":
                    response = self._handle_hoi_ngay_mai(response)
                elif tag == "thoi_tiet":
                    response = self._handle_thoi_tiet(message, is_tomorrow=False)
                elif tag == "thoi_tiet_ngay_mai":
                    response = self._handle_thoi_tiet(message, is_tomorrow=True)

                return response

        return "Tôi chưa biết cách trả lời câu hỏi này."

    def _handle_hoi_gio(self, response):
        """Xử lý intent hoi_gio: Trả về thời gian hiện tại."""
        tz = pytz.timezone('Asia/Ho_Chi_Minh')
        now = datetime.now(tz)
        time_str = now.strftime("%H:%M:%S")
        date_str = now.strftime("%d/%m/%Y")
        day = now.strftime("%d")
        month = now.strftime("%m")
        year = now.strftime("%Y")
        response = response.replace("{time}", time_str).replace("{date}", date_str)
        response = response.replace("{day}", day).replace("{month}", month).replace("{year}", year)
        return response

    def _handle_hoi_ngay_mai(self, response):
        """Xử lý intent hoi_ngay_mai: Trả về ngày mai."""
        tz = pytz.timezone('Asia/Ho_Chi_Minh')
        tomorrow = datetime.now(tz) + timedelta(days=1)
        tomorrow_str = tomorrow.strftime("%d/%m/%Y")
        tomorrow_day = tomorrow.strftime("%A")
        days_vi = {
            "Monday": "Thứ Hai",
            "Tuesday": "Thứ Ba",
            "Wednesday": "Thứ Tư",
            "Thursday": "Thứ Năm",
            "Friday": "Thứ Sáu",
            "Saturday": "Thứ Bảy",
            "Sunday": "Chủ Nhật"
        }
        tomorrow_day_vi = days_vi.get(tomorrow_day, tomorrow_day)
        return response.replace("{tomorrow}", f"{tomorrow_day_vi}, ngày {tomorrow_str}")

    def _handle_thoi_tiet(self, message, is_tomorrow=False):
        """Xử lý intent thoi_tiet: Lấy thông tin thời tiết từ WeatherAPI.com."""
        message = message.lower()

        # Xác định thành phố
        city = "Ho Chi Minh"  # Mặc định là TP. Hồ Chí Minh
        city_vi = "Thành phố Hồ Chí Minh"  # Tên tiếng Việt mặc định
        for city_vi_name, city_en in VIETNAM_CITIES.items():
            if city_vi_name.lower() in message:
                city = city_en
                city_vi = city_vi_name
                break

        # Xác định loại yêu cầu
        request_type = "full"
        if "nhiệt độ" in message:
            request_type = "temperature"
        elif "mưa" in message:
            request_type = "rain"
        elif "nắng" in message or "lạnh" in message or "nóng" in message or "đẹp" in message or "xấu" in message or "dễ chịu" in message:
            request_type = "description"

        # Gọi API WeatherAPI.com
        base_url = "http://api.weatherapi.com/v1"
        if is_tomorrow:
            # Dự báo thời tiết ngày mai
            url = f"{base_url}/forecast.json?key={self.weather_api_key}&q={city}&days=2&aqi=no&alerts=no"
            try:
                api_response = requests.get(url)
                api_response.raise_for_status()
                data = api_response.json()
                city_name = city_vi  # Sử dụng tên thành phố tiếng Việt

                # Lấy dữ liệu dự báo ngày mai
                forecast = data["forecast"]["forecastday"][1]  # Ngày mai là ngày thứ 2 trong danh sách
                date = forecast["date"]
                temp = forecast["day"]["avgtemp_c"]
                feels_like = forecast["day"]["avgtemp_c"]  # WeatherAPI không có feels_like cho dự báo ngày, dùng avgtemp_c
                description = forecast["day"]["condition"]["text"]
                # Dịch mô tả thời tiết sang tiếng Việt
                description_vi = WEATHER_CONDITIONS_VI.get(description, description)
                humidity = forecast["day"]["avghumidity"]
                rain = forecast["day"]["totalprecip_mm"]
                wind_speed = forecast["day"]["maxwind_kph"]  # Tốc độ gió tối đa trong ngày
                # WeatherAPI không cung cấp áp suất cho dự báo ngày, bỏ qua áp suất trong trường hợp này

                if request_type == "temperature":
                    return f"Dự báo nhiệt độ ngày mai tại {city_name} là {temp}°C."
                elif request_type == "rain":
                    if rain > 0:
                        return f"Dự báo ngày mai tại {city_name} có mưa, lượng mưa {rain}mm. {description_vi.capitalize()}."
                    else:
                        return f"Dự báo ngày mai tại {city_name} không có mưa. {description_vi.capitalize()}."
                elif request_type == "description":
                    return f"Dự báo ngày mai: {description_vi.capitalize()} tại {city_name}, nhiệt độ {temp}°C."
                else:
                    return f"Dự báo thời tiết ngày mai tại {city_name} ({date}): {description_vi}, nhiệt độ {temp}°C, độ ẩm {humidity}%, tốc độ gió {wind_speed} km/h."
            except requests.exceptions.RequestException as e:
                return f"Không thể lấy thông tin thời tiết: {str(e)}"
            except (KeyError, IndexError):
                return "Dữ liệu thời tiết không hợp lệ. Vui lòng thử lại sau."
        else:
            # Thời tiết hiện tại
            url = f"{base_url}/current.json?key={self.weather_api_key}&q={city}&aqi=no"
            try:
                api_response = requests.get(url)
                api_response.raise_for_status()
                data = api_response.json()
                temp = data["current"]["temp_c"]
                feels_like = data["current"]["feelslike_c"]
                description = data["current"]["condition"]["text"]
                # Dịch mô tả thời tiết sang tiếng Việt
                description_vi = WEATHER_CONDITIONS_VI.get(description, description)
                humidity = data["current"]["humidity"]
                city_name = city_vi  # Sử dụng tên thành phố tiếng Việt
                rain = data["current"]["precip_mm"]
                wind_speed = data["current"]["wind_kph"]  # Tốc độ gió
                pressure = data["current"]["pressure_mb"]  # Áp suất

                if request_type == "temperature":
                    return f"Nhiệt độ tại {city_name} là {temp}°C, cảm giác như {feels_like}°C."
                elif request_type == "rain":
                    if rain > 0:
                        return f"Có mưa tại {city_name}, lượng mưa {rain}mm. {description_vi.capitalize()}."
                    else:
                        return f"Không có mưa tại {city_name}. {description_vi.capitalize()}."
                elif request_type == "description":
                    return f"{description_vi.capitalize()} tại {city_name}, nhiệt độ {temp}°C."
                else:
                    return f"Thời tiết tại {city_name}: {description_vi}, nhiệt độ {temp}°C, độ ẩm {humidity}%, tốc độ gió {wind_speed} km/h, áp suất {pressure} mb."
            except requests.exceptions.RequestException as e:
                return f"Không thể lấy thông tin thời tiết: {str(e)}"
            except (KeyError, IndexError):
                return "Dữ liệu thời tiết không hợp lệ. Vui lòng thử lại sau."

    def chat(self, message):
        """
        Xử lý tin nhắn và trả về phản hồi.

        Args:
            message (str): Tin nhắn từ người dùng.

        Returns:
            str: Phản hồi của bot.
        """
        intents = self.predict_class(message)
        return self.get_response(intents, self.intents, message)

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Đang lắng nghe...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio, language='vi-VN')
                print(f"Bạn đã nói: {text}")
                return text
            except sr.UnknownValueError:
                return "Xin lỗi, tôi không thể nhận diện được giọng nói."
            except sr.RequestError:
                return "Xin lỗi, có lỗi xảy ra khi kết nối đến dịch vụ nhận diện giọng nói."

    def learn(self, message, response, tag=None):
        """
        Học từ đầu vào của người dùng bằng cách fine-tune PhoBERT.

        Args:
            message (str): Câu hỏi hoặc mẫu câu từ người dùng.
            response (str): Phản hồi mong muốn.
            tag (str, optional): Tag của intent. Nếu không cung cấp, sẽ tự động tạo.

        Returns:
            str: Thông báo kết quả học.
        """
        logging.info(f"Đang học: '{message}' -> '{response}' với tag '{tag}'")

        # Xác định tag
        if not tag:
            intents = self.predict_class(message)
            if intents and float(intents[0]["probability"]) > 0.7:
                tag = intents[0]["intent"]
            else:
                words = re.sub(r'[^\w\s]', '', message.lower()).split()
                tag = "_".join(words[:2]) if len(words) > 1 else words[0] if words else "tu_hoc"

        # Thêm dữ liệu mới vào file intents
        tag_exists = False
        for intent in self.intents["intents"]:
            if intent["tag"] == tag:
                if message.lower() not in [p.lower() for p in intent["patterns"]]:
                    intent["patterns"].append(message)
                if response not in intent["responses"]:
                    intent["responses"].append(response)
                tag_exists = True
                break

        if not tag_exists:
            new_intent = {"tag": tag, "patterns": [message], "responses": [response]}
            self.intents["intents"].append(new_intent)
            # Cập nhật self.classes nếu có intent mới
            if tag not in self.classes:
                self.classes.append(tag)
                # Cập nhật số lượng nhãn trong mô hình
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_dir,
                    num_labels=len(self.classes)
                )
                self.model.to(self.device)

        # Lưu file intents
        with open(self.intents_file, 'w', encoding='utf-8') as f:
            json.dump(self.intents, f, indent=4, ensure_ascii=False)

        # Chuẩn bị dữ liệu mới để fine-tune
        label_encoder = LabelEncoder()
        label_encoder.fit(self.classes)
        new_sentences = [message]
        new_labels = [label_encoder.transform([tag])[0]]

        # Fine-tune mô hình trên dữ liệu mới
        self.fine_tune_on_new_data(new_sentences, new_labels)

        return f"Đã học thành công: '{message}' -> '{response}' với tag '{tag}'."

if __name__ == "__main__":
    bot = DeepLearningChatbot()
    print(bot.chat("Xin chào"))
#set WEATHERAPI_KEY=a45b275f281e4177a4c75446252603