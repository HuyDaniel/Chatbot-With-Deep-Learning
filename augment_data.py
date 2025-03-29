import json
import random

# Từ điển từ đồng nghĩa (có thể mở rộng thêm)
SYNONYMS = {
    "xin chào": ["chào", "hello", "hi", "chào bạn", "chào nhé"],
    "tạm biệt": ["bye", "hẹn gặp lại", "gặp lại sau", "chào tạm biệt", "tạm biệt nhé"],
    "cảm ơn": ["cám ơn", "thanks", "cảm ơn nhiều", "cảm ơn nhé", "cảm ơn bạn"],
    "giúp đỡ": ["hỗ trợ", "giúp", "giúp mình", "hỗ trợ mình", "giúp tôi"],
    "thời tiết": ["trời", "nhiệt độ", "dự báo", "thời tiết hôm nay", "trời hôm nay"],
    "ngày mai": ["mai", "ngày tới", "ngày sau", "ngày kế tiếp", "ngày tiếp theo"],
    "hôm nay": ["bây giờ", "hiện tại", "ngày này", "hôm nay đây", "ngày hôm nay"],
    "mấy giờ": ["bao nhiêu giờ", "giờ hiện tại", "thời gian bây giờ", "giờ là bao nhiêu", "giờ này là mấy"],
    "ngày bao nhiêu": ["ngày mấy", "ngày nào", "hôm nay ngày gì", "ngày hiện tại", "ngày hôm nay là bao nhiêu"],
    "thứ mấy": ["thứ gì", "thứ bao nhiêu", "ngày mai thứ gì", "thứ nào", "ngày mai là thứ gì"]
}

def augment_sentence(sentence):
    """Tạo các biến thể của một câu bằng cách thay thế từ đồng nghĩa và thêm nhiễu."""
    words = sentence.split()
    augmented_sentences = [sentence]

    # Thay thế từ đồng nghĩa
    for i, word in enumerate(words):
        for key, synonyms in SYNONYMS.items():
            if word.lower() == key.lower():
                for synonym in synonyms:
                    new_words = words.copy()
                    new_words[i] = synonym
                    new_sentence = " ".join(new_words)
                    if new_sentence not in augmented_sentences:
                        augmented_sentences.append(new_sentence)

    # Thêm nhiễu (thêm từ "ơi", "nhé", "nha", v.v.)
    noise_words = ["ơi", "nhé", "nha", "bạn ơi", "bạn nhé", "nào", "với", "mình ơi", "mình nhé"]
    for noise in noise_words:
        new_sentence = sentence + " " + noise
        if new_sentence not in augmented_sentences:
            augmented_sentences.append(new_sentence)

    return augmented_sentences

def augment_data(input_file, output_file, target_samples=1000):
    """Tăng số lượng mẫu trong file JSON."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    augmented_intents = []
    current_samples = sum(len(intent["patterns"]) for intent in data["intents"])
    print(f"Số mẫu hiện tại: {current_samples}")

    for intent in data["intents"]:
        tag = intent["tag"]
        patterns = intent["patterns"]
        responses = intent["responses"]
        augmented_patterns = set(patterns)

        # Tăng số lượng mẫu cho mỗi intent
        while len(augmented_patterns) < target_samples // len(data["intents"]):
            for pattern in patterns:
                new_patterns = augment_sentence(pattern)
                augmented_patterns.update(new_patterns)
                if len(augmented_patterns) >= target_samples // len(data["intents"]):
                    break

        augmented_intents.append({
            "tag": tag,
            "patterns": list(augmented_patterns),
            "responses": responses
        })

    # Lưu file mới
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({"intents": augmented_intents}, f, indent=4, ensure_ascii=False)

    new_samples = sum(len(intent["patterns"]) for intent in augmented_intents)
    print(f"Số mẫu sau khi tăng: {new_samples}")

if __name__ == "__main__":
    augment_data("chatbot_intents_vi.json", "chatbot_intents_vi_augmented.json", target_samples=1000)