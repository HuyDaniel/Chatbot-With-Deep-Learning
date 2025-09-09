# Chatbot-With-Deep-Learning
# Hiện tại file chatbotfilebackupPhoBERT, Appbackup, augment_data, chatbot_intents_vi_augmented - file được tạo ra từ file augment_data, nên bỏ trong 1 môi trường ảo riêng để chạy riêng biệt. Do Mô hình có tới 135 triệu tham số nên xử lý rất lâu. Do code chỉ đang chạy tối ưu bằng torch CPU, chưa thiết lập torch dành cho GPU nên chạy siêu chậm. Dẫn đến lag máy nếu CPU yếu. Nên tích hợp CUDA toolkit thì sẽ chạy nhanh hơn và tốc độ xử lý tối ưu hơn. Vì là chạy bằng GPU nên cần cài đặt thêm 1 thư viện chuẩn của torch-gpu (check bằng msi-nvidia)
Phương án 1: (chạy bản low dùng được ngay)
Chuẩn bị môi trường: # Windows / macOS / Linux đều được
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Cài thư viện tối thiểu cho bản .h5
pip install --upgrade pip
pip install numpy pandas scikit-learn nltk unidecode
pip install "tensorflow==2.12.*" "keras==2.12.*"
pip install tk  # nếu Windows/macOS thường ok; Linux có thể phải apt: sudo apt-get install python3-tk
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet')"
# thử file này trước
python app.py
# nếu không có giao diện/console hiện lên, thử:
python chatbot.py
Chạy với windows PowerShell
select-string -Path *.py -Pattern "__main__"
Chạy Với Mac:
grep -nR "__main__" .
LOAD KERAS MODEL
tf.keras.models.load_model("chatbot_model_vi.h5", compile=False)

