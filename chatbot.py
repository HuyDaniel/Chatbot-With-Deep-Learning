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
from nltk.stem import WordNetLemmatizer
import pytz
import requests
import speech_recognition as sr

# TensorFlow và Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD

# Tối ưu hóa cho CPU AMD
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Tắt GPU
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Bật oneDNN
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # Hiển thị cảnh báo để debug

# Tối ưu hóa multi-threading
num_threads = os.cpu_count()  # Lấy số luồng của CPU
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Tải dữ liệu cần thiết cho NLTK
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

class DeepLearningChatbot:
    def __init__(self, intents_file="chatbot_intents_vi.json", model_file="chatbot_model_vi.h5", 
                 words_file="words_vi.pkl", classes_file="classes_vi.pkl"):
        """
        Khởi tạo chatbot học sâu.

        Args:
            intents_file (str): Đường dẫn đến file JSON chứa dữ liệu intents.
            model_file (str): Đường dẫn đến file mô hình đã huấn luyện.
            words_file (str): Đường dẫn đến file chứa danh sách từ vựng.
            classes_file (str): Đường dẫn đến file chứa danh sách các intent.
        """
        self.lemmatizer = WordNetLemmatizer()
        self.intents_file = intents_file
        self.model_file = model_file
        self.words_file = words_file
        self.classes_file = classes_file
        self.error_threshold = 0.1  # Ngưỡng xác suất để nhận diện intent
        self.weather_api_key = os.getenv("WEATHERAPI_KEY")  # Lấy API key từ biến môi trường

        # Kiểm tra nếu API key không tồn tại
        if not self.weather_api_key:
            raise ValueError("Không tìm thấy API key cho WeatherAPI. Vui lòng đặt biến môi trường WEATHERAPI_KEY.")

        # Kiểm tra và tải hoặc huấn luyện mô hình
        if self._check_model_files_exist():
            self.load_model()
        else:
            self.train_model()

    def _check_model_files_exist(self):
        """Kiểm tra xem các file mô hình và dữ liệu đã tồn tại chưa."""
        return (os.path.exists(self.model_file) and 
                os.path.exists(self.words_file) and 
                os.path.exists(self.classes_file))

    def _clear_temp_files(self):
        """
        Xóa các file tạm thời (mô hình và dữ liệu huấn luyện).

        Returns:
            str: Thông báo kết quả xóa.
        """
        files_to_delete = [self.model_file, self.words_file, self.classes_file]
        deleted_files = []
        not_found_files = []

        for file in files_to_delete:
            try:
                if os.path.exists(file):
                    os.remove(file)
                    deleted_files.append(file)
                else:
                    not_found_files.append(file)
            except Exception as e:
                return f"Lỗi khi xóa file {file}: {str(e)}"

        result = ""
        if deleted_files:
            result += f"Đã xóa các file: {', '.join(deleted_files)}.\n"
        if not_found_files:
            result += f"Các file không tồn tại: {', '.join(not_found_files)}."
        return result if result else "Không có file nào để xóa."

    def clear_temp_files(self):
        """
        Xóa các file tạm thời (mô hình và dữ liệu huấn luyện) theo yêu cầu.

        Returns:
            str: Thông báo kết quả xóa.
        """
        return self._clear_temp_files()

    def load_model(self):
        """Tải mô hình đã huấn luyện và dữ liệu từ các file."""
        logging.info("Đang tải mô hình và dữ liệu...")
        self.words = pickle.load(open(self.words_file, 'rb'))
        self.classes = pickle.load(open(self.classes_file, 'rb'))
        self.model = load_model(self.model_file)
        with open(self.intents_file, 'r', encoding='utf-8') as f:
            self.intents = json.load(f)
        logging.info("Tải mô hình và dữ liệu thành công.")

    def train_model(self):
        """Huấn luyện mô hình mới dựa trên file intents."""
        logging.info("Bắt đầu huấn luyện mô hình mới...")
        # Tải hoặc tạo file intents
        self._load_or_create_intents()

        # Chuẩn bị dữ liệu huấn luyện
        words, classes, documents = self._prepare_training_data()

        # Lưu danh sách từ và classes
        pickle.dump(words, open(self.words_file, 'wb'))
        pickle.dump(classes, open(self.classes_file, 'wb'))

        # Tạo dữ liệu huấn luyện
        train_x, train_y = self._create_training_data(words, classes, documents)

        # Xây dựng và huấn luyện mô hình
        self.model = self._build_model(len(train_x[0]), len(train_y[0]))
        self.model.fit(np.array(train_x), np.array(train_y), epochs=300, batch_size=5, verbose=1)
        self.model.save(self.model_file)

        self.words = words
        self.classes = classes
        logging.info("Huấn luyện mô hình hoàn tất và đã lưu.")

    def _load_or_create_intents(self):
        """Tải file intents hoặc tạo file mặc định nếu không tồn tại."""
        try:
            with open(self.intents_file, 'r', encoding='utf-8') as f:
                self.intents = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logging.warning("Không tìm thấy file intents hoặc file bị lỗi. Tạo file intents mặc định...")
            # Tạo file intents mặc định với nội dung tiếng Việt
            self.intents = {
                "intents": [
                    {
                        "tag": "chao_hoi",
                        "patterns": ["Xin chào", "Chào bạn", "Hello", "Chào buổi sáng", "Bạn khỏe không"],
                        "responses": ["Xin chào!", "Chào bạn!", "Rất vui được gặp bạn!", "Chào bạn, tôi có thể giúp gì cho bạn?"]
                    },
                    {
                        "tag": "tam_biet",
                        "patterns": ["Tạm biệt", "Hẹn gặp lại", "Chào tạm biệt", "Gặp lại sau"],
                        "responses": ["Tạm biệt bạn!", "Hẹn gặp lại!", "Chúc bạn một ngày tốt lành!"]
                    },
                    {
                        "tag": "cam_on",
                        "patterns": ["Cảm ơn", "Cảm ơn bạn", "Thật hữu ích"],
                        "responses": ["Không có gì!", "Rất vui được giúp bạn!", "Không có chi!"]
                    },
                    {
                        "tag": "gioi_thieu",
                        "patterns": ["Bạn là ai?", "Bạn là gì?", "Cho tôi biết về bạn"],
                        "responses": ["Tôi là chatbot học sâu!", "Tôi là trợ lý AI của bạn, được hỗ trợ bởi deep learning."]
                    },
                    {
                        "tag": "tro_giup",
                        "patterns": ["Giúp đỡ", "Tôi cần giúp đỡ", "Bạn có thể giúp tôi không?", "Bạn có thể làm gì?"],
                        "responses": ["Tôi có thể trả lời câu hỏi, trò chuyện và học những điều mới!"]
                    },
                    {
                        "tag": "hoi_gio",
                        "patterns": ["Bây giờ là mấy giờ?", "Mấy giờ rồi?", "Cho tôi biết giờ hiện tại", "Hôm nay là ngày bao nhiêu?"],
                        "responses": ["Để tôi kiểm tra... Bây giờ là {time} ngày {date}."]
                    },
                    {
                        "tag": "hoi_ngay_mai",
                        "patterns": ["Ngày mai là ngày bao nhiêu?", "Ngày mai là ngày gì?", "Cho tôi biết ngày mai", "Ngày mai là thứ mấy?"],
                        "responses": ["Để tôi kiểm tra ngày mai cho bạn."]
                    },
                    {
                        "tag": "thoi_tiet",
                        "patterns": ["Thời tiết hôm nay thế nào?", "Trời có mưa không?", "Thời tiết ở TP.HCM ra sao?", "Báo nhiệt độ", "Báo gió và áp suất"],
                        "responses": ["Để tôi kiểm tra thời tiết cho bạn..."]
                    },
                    {
                        "tag": "thoi_tiet_ngay_mai",
                        "patterns": ["Thời tiết ngày mai thế nào?", "Ngày mai trời có mưa không?", "Thời tiết ngày mai ở TP.HCM ra sao?"],
                        "responses": ["Để tôi kiểm tra dự báo thời tiết ngày mai cho bạn..."]
                    }
                ]
            }
            with open(self.intents_file, 'w', encoding='utf-8') as f:
                json.dump(self.intents, f, indent=4, ensure_ascii=False)

    def _prepare_training_data(self):
        """Chuẩn bị dữ liệu huấn luyện từ file intents."""
        words = []
        classes = []
        documents = []
        ignore_chars = ['?', '!', '.', ',']

        for intent in self.intents["intents"]:
            for pattern in intent["patterns"]:
                word_list = nltk.word_tokenize(pattern)
                words.extend(word_list)
                documents.append((word_list, intent["tag"]))
                if intent["tag"] not in classes:
                    classes.append(intent["tag"])

        words = [self.lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore_chars]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))

        return words, classes, documents

    def _create_training_data(self, words, classes, documents):
        """Tạo dữ liệu huấn luyện từ danh sách từ và documents."""
        training = []
        output_empty = [0] * len(classes)

        for document in documents:
            bag = []
            word_patterns = document[0]
            word_patterns = [self.lemmatizer.lemmatize(word.lower()) for word in word_patterns]
            for word in words:
                bag.append(1) if word in word_patterns else bag.append(0)
            output_row = list(output_empty)
            output_row[classes.index(document[1])] = 1
            training.append([bag, output_row])

        random.shuffle(training)
        training = np.array(training, dtype=object)
        train_x = list(training[:, 0])
        train_y = list(training[:, 1])

        return train_x, train_y

    def _build_model(self, input_size, output_size):
        """Xây dựng mô hình mạng nơ-ron."""
        model = Sequential([
            Dense(128, input_shape=(input_size,), activation='relu'),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(output_size, activation='softmax')
        ])

        sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        return model

    def clean_sentence(self, sentence):
        """Chuẩn hóa câu để dự đoán."""
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
        return sentence_words

    def create_bow(self, sentence):
        """Tạo bag of words từ câu."""
        sentence_words = self.clean_sentence(sentence)
        bag = [0] * len(self.words)
        for word in sentence_words:
            for i, w in enumerate(self.words):
                if w == word:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        """
        Dự đoán ý định của câu.

        Args:
            sentence (str): Câu hỏi từ người dùng.

        Returns:
            list: Danh sách các ý định dự đoán với xác suất.
        """
        bow = self.create_bow(sentence)
        res = self.model.predict(np.array([bow]), verbose=0)[0]
        results = [[i, r] for i, r in enumerate(res) if r > self.error_threshold]
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
        """Xử lý intent thoi_tiet: Lấy thông tin thời tiết từ WeatherAPI.com và dịch sang tiếng Việt."""
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
                # Chuyển định dạng ngày từ YYYY-MM-DD sang DD/MM/YYYY
                date_obj = datetime.strptime(date, "%Y-%m-%d")
                date_vi = date_obj.strftime("%d/%m/%Y")
                temp = forecast["day"]["avgtemp_c"]
                feels_like = forecast["day"]["avgtemp_c"]  # WeatherAPI không có feels_like cho dự báo ngày, dùng avgtemp_c
                description = forecast["day"]["condition"]["text"]
                # Dịch mô tả thời tiết sang tiếng Việt
                description_vi = WEATHER_CONDITIONS_VI.get(description, description)
                if description_vi == description:  # Nếu không dịch được, ghi log để kiểm tra
                    logging.warning(f"Mô tả thời tiết '{description}' chưa được dịch trong WEATHER_CONDITIONS_VI.")
                humidity = forecast["day"]["avghumidity"]
                rain = forecast["day"]["totalprecip_mm"]
                wind_speed = forecast["day"]["maxwind_kph"]  # Tốc độ gió tối đa trong ngày
                # WeatherAPI không cung cấp áp suất cho dự báo ngày, bỏ qua áp suất trong trường hợp này

                if request_type == "temperature":
                    return f"Dự báo nhiệt độ ngày mai tại {city_name} là {temp}°C."
                elif request_type == "rain":
                    if rain > 0:
                        return f"Dự báo ngày mai tại {city_name} sẽ có mưa, lượng mưa khoảng {rain} mm. Trời {description_vi.lower()}."
                    else:
                        return f"Dự báo ngày mai tại {city_name} không có mưa. Trời {description_vi.lower()}."
                elif request_type == "description":
                    return f"Dự báo ngày mai tại {city_name}, trời {description_vi.lower()}, nhiệt độ trung bình {temp}°C."
                else:
                    return f"Dự báo thời tiết ngày mai ({date_vi}) tại {city_name}: Trời {description_vi.lower()}, nhiệt độ trung bình {temp}°C, độ ẩm {humidity}%, tốc độ gió tối đa {wind_speed} km/h."
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
                if description_vi == description:  # Nếu không dịch được, ghi log để kiểm tra
                    logging.warning(f"Mô tả thời tiết '{description}' chưa được dịch trong WEATHER_CONDITIONS_VI.")
                humidity = data["current"]["humidity"]
                city_name = city_vi  # Sử dụng tên thành phố tiếng Việt
                rain = data["current"]["precip_mm"]
                wind_speed = data["current"]["wind_kph"]  # Tốc độ gió
                pressure = data["current"]["pressure_mb"]  # Áp suất

                if request_type == "temperature":
                    return f"Nhiệt độ hiện tại tại {city_name} là {temp}°C, cảm giác như {feels_like}°C."
                elif request_type == "rain":
                    if rain > 0:
                        return f"Hiện tại tại {city_name} đang có mưa, lượng mưa {rain} mm. Trời {description_vi.lower()}."
                    else:
                        return f"Hiện tại tại {city_name} không có mưa. Trời {description_vi.lower()}."
                elif request_type == "description":
                    return f"Hiện tại tại {city_name}, trời {description_vi.lower()}, nhiệt độ {temp}°C."
                else:
                    return f"Thời tiết hiện tại tại {city_name}: Trời {description_vi.lower()}, nhiệt độ {temp}°C, cảm giác như {feels_like}°C, độ ẩm {humidity}%, tốc độ gió {wind_speed} km/h, áp suất {pressure} mb."
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
        Học từ đầu vào của người dùng.

        Args:
            message (str): Câu hỏi hoặc mẫu câu từ người dùng.
            response (str): Phản hồi mong muốn.
            tag (str, optional): Tag của intent. Nếu không cung cấp, sẽ tự động tạo.

        Returns:
            str: Thông báo kết quả học.
        """
        logging.info(f"Đang học: '{message}' -> '{response}' với tag '{tag}'")
        if not tag:
            intents = self.predict_class(message)
            if intents and float(intents[0]["probability"]) > 0.7:
                tag = intents[0]["intent"]
            else:
                words = re.sub(r'[^\w\s]', '', message.lower()).split()
                tag = "_".join(words[:2]) if len(words) > 1 else words[0] if words else "tu_hoc"

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

        with open(self.intents_file, 'w', encoding='utf-8') as f:
            json.dump(self.intents, f, indent=4, ensure_ascii=False)

        # Xóa các file tạm thời trước khi huấn luyện lại
        clear_result = self._clear_temp_files()
        logging.info(clear_result)

        self.train_model()
        return f"Đã học thành công: '{message}' -> '{response}' với tag '{tag}'."

if __name__ == "__main__":
    bot = DeepLearningChatbot()
    print(bot.chat("Xin chào"))