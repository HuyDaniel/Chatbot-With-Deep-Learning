import customtkinter as ctk
from tkinter import messagebox
import tkinter.ttk as ttk  # Sử dụng ttk từ tkinter thay vì ttkbootstrap
from chatbot import DeepLearningChatbot  # Giả định bạn đã có lớp này

# Tắt DPI scaling để tránh lỗi
ctk.deactivate_automatic_dpi_awareness()


class ChatbotApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Chatbot Học Sâu - Tiếng Việt")
        self.root.geometry("700x800")
        self.root.minsize(600, 700)
        self.chatbot = DeepLearningChatbot()  # Khởi tạo chatbot
        self.recording_popup = None  # Biến để lưu cửa sổ pop-up
        self.setup_ui()

    def create_recording_popup(self):
        """Tạo cửa sổ pop-up hiển thị khi đang ghi âm"""
        if self.recording_popup is not None:
            self.recording_popup.destroy()  # Đóng pop-up cũ nếu có

        self.recording_popup = ctk.CTkToplevel(self.root)
        self.recording_popup.title("Đang ghi âm")
        self.recording_popup.geometry("300x150")
        self.recording_popup.resizable(False, False)
        self.recording_popup.transient(self.root)  # Đặt pop-up là cửa sổ con của cửa sổ chính
        self.recording_popup.grab_set()  # Đảm bảo pop-up nhận focus

        # Tùy chỉnh giao diện pop-up
        self.recording_popup.configure(fg_color="#2A2D3E")

        # Nhãn hiển thị "Đang ghi âm..."
        label = ctk.CTkLabel(
            self.recording_popup,
            text="Đang ghi âm...",
            font=("Helvetica", 16, "bold"),
            text_color="#DCDDDE"
        )
        label.pack(pady=20)

        # Thêm biểu tượng micro
        micro_icon = ctk.CTkLabel(
            self.recording_popup,
            text="🎤",
            font=("Helvetica", 40),
            text_color="#FF9500"
        )
        micro_icon.pack()

        # Đảm bảo pop-up ở trên cùng
        self.recording_popup.lift()
        self.recording_popup.attributes('-topmost', True)  # Đặt pop-up luôn ở trên cùng
        self.recording_popup.update()  # Cập nhật giao diện để tránh lỗi hiển thị

    def close_recording_popup(self):
        """Đóng cửa sổ pop-up sau khi ghi âm xong"""
        if self.recording_popup is not None:
            self.recording_popup.destroy()
            self.recording_popup = None

    def record_and_send(self):
        """Ghi âm và gửi tin nhắn từ giọng nói"""
        try:
            # Hiển thị pop-up trước khi ghi âm
            self.create_recording_popup()
            self.status_var.set("Đang ghi âm...")
            self.root.update_idletasks()

            # Ghi âm và chuyển đổi thành văn bản
            message = self.chatbot.speech_to_text()

            # Đóng pop-up sau khi ghi âm xong
            self.close_recording_popup()

            # Xử lý kết quả ghi âm
            if message.startswith("Xin lỗi"):
                self.append_to_chat("Chatbot", message)
            else:
                self.append_to_chat("Bạn", message)
                response = self.chatbot.chat(message)
                self.append_to_chat("Chatbot", response)
            self.status_var.set("Sẵn sàng")
        except Exception as e:
            self.close_recording_popup()  # Đóng pop-up nếu có lỗi
            self.status_var.set(f"Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")

    def setup_ui(self):
        """Thiết lập giao diện người dùng với CustomTkinter"""
        # Cấu hình theme
        ctk.set_appearance_mode("dark")  # Theme: "light", "dark", "system"
        ctk.set_default_color_theme("dark-blue")  # Màu theme: "blue", "green", "dark-blue"

        # Frame chính
        main_frame = ctk.CTkFrame(self.root, fg_color="#1A1B26", corner_radius=0)
        main_frame.pack(fill=ctk.BOTH, expand=True)

        # Tabview cho các tab
        self.tabview = ctk.CTkTabview(
            main_frame,
            fg_color="#2A2D3E",
            segmented_button_fg_color="#3B3F52",
            segmented_button_selected_color="#7289DA",
            segmented_button_selected_hover_color="#5A6EBB",
            text_color="#DCDDDE"
        )
        self.tabview.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # Tab "Trò chuyện"
        chat_tab = self.tabview.add("Trò chuyện")

        # Tab "Dạy học"
        learning_tab = self.tabview.add("Dạy học")

        # Thanh trạng thái
        self.status_var = ctk.StringVar(value="Sẵn sàng")
        status_bar = ctk.CTkLabel(
            main_frame,
            textvariable=self.status_var,
            anchor=ctk.W,
            fg_color="#1A1B26",
            text_color="#7289DA",
            font=("Helvetica", 12)
        )
        status_bar.pack(side=ctk.BOTTOM, fill=ctk.X, padx=10, pady=5)

        # === Giao diện Tab "Trò chuyện" ===
        # Frame chứa lịch sử trò chuyện
        chat_frame = ctk.CTkFrame(chat_tab, fg_color="#2A2D3E", corner_radius=10)
        chat_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=10)

        # Lịch sử trò chuyện (CTkTextbox)
        self.chat_history = ctk.CTkTextbox(
            chat_frame,
            wrap=ctk.WORD,
            state=ctk.DISABLED,
            font=("Helvetica", 14),
            fg_color="#36393F",
            text_color="#DCDDDE",
            border_spacing=10
        )
        self.chat_history.pack(fill=ctk.BOTH, expand=True, padx=5, pady=5)

        # Frame chứa các nút điều khiển
        control_frame = ctk.CTkFrame(chat_tab, fg_color="#2A2D3E", corner_radius=10)
        control_frame.pack(fill=ctk.X, padx=10, pady=(0, 5))

        # Nút "Xóa lịch sử trò chuyện"
        clear_chat_button = ctk.CTkButton(
            control_frame,
            text="Xóa lịch sử",
            command=self.clear_chat_history,
            width=120,
            fg_color="#FF5555",
            hover_color="#CC4444",
            font=("Helvetica", 12, "bold")
        )
        clear_chat_button.pack(side=ctk.LEFT, padx=(0, 5))

        # Nút "Xóa dữ liệu tạm thời"
        clear_temp_button = ctk.CTkButton(
            control_frame,
            text="Xóa dữ liệu tạm thời",
            command=self.clear_temp_files,
            width=150,
            fg_color="#FF9500",
            hover_color="#CC7700",
            font=("Helvetica", 12, "bold")
        )
        clear_temp_button.pack(side=ctk.LEFT)

        # Frame nhập tin nhắn
        input_frame = ctk.CTkFrame(chat_tab, fg_color="#2A2D3E", corner_radius=10)
        input_frame.pack(fill=ctk.X, padx=10, pady=5)

        # Ô nhập tin nhắn
        self.message_input = ctk.CTkEntry(
            input_frame,
            font=("Helvetica", 14),
            placeholder_text="Nhập tin nhắn...",
            fg_color="#36393F",
            text_color="#DCDDDE",
            border_color="#7289DA",
            border_width=2,
            corner_radius=10
        )
        self.message_input.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 5))
        self.message_input.bind("<Return>", self.send_message)

        # Nút gửi
        send_button = ctk.CTkButton(
            input_frame,
            text="Gửi",
            command=self.send_message,
            width=100,
            fg_color="#7289DA",
            hover_color="#5A6EBB",
            font=("Helvetica", 14, "bold")
        )
        send_button.pack(side=ctk.RIGHT)

        # Nút ghi âm
        record_button = ctk.CTkButton(
            input_frame,
            text="🎤 Nói",
            command=self.record_and_send,
            width=100,
            fg_color="#FF9500",
            hover_color="#CC7700",
            font=("Helvetica", 14, "bold")
        )
        record_button.pack(side=ctk.RIGHT, padx=(0, 5))

        # === Giao diện Tab "Dạy học" ===
        # Frame nhập dữ liệu dạy học
        learn_frame = ctk.CTkFrame(learning_tab, fg_color="#2A2D3E", corner_radius=10)
        learn_frame.pack(fill=ctk.BOTH, expand=False, padx=10, pady=10)

        # Nhãn và ô nhập tin nhắn người dùng
        ctk.CTkLabel(
            learn_frame,
            text="Tin nhắn người dùng:",
            font=("Helvetica", 14),
            text_color="#DCDDDE"
        ).grid(row=0, column=0, sticky=ctk.W, padx=10, pady=5)
        self.learn_message = ctk.CTkEntry(
            learn_frame,
            font=("Helvetica", 14),
            fg_color="#36393F",
            text_color="#DCDDDE",
            border_color="#7289DA",
            border_width=2,
            corner_radius=10
        )
        self.learn_message.grid(row=0, column=1, columnspan=2, sticky=ctk.EW, padx=10, pady=5)

        # Nhãn và ô nhập phản hồi của bot
        ctk.CTkLabel(
            learn_frame,
            text="Phản hồi của bot:",
            font=("Helvetica", 14),
            text_color="#DCDDDE"
        ).grid(row=1, column=0, sticky=ctk.W, padx=10, pady=5)
        self.learn_response = ctk.CTkEntry(
            learn_frame,
            font=("Helvetica", 14),
            fg_color="#36393F",
            text_color="#DCDDDE",
            border_color="#7289DA",
            border_width=2,
            corner_radius=10
        )
        self.learn_response.grid(row=1, column=1, columnspan=2, sticky=ctk.EW, padx=10, pady=5)

        # Nhãn và ô nhập chủ đề (tùy chọn)
        ctk.CTkLabel(
            learn_frame,
            text="Chủ đề (tùy chọn):",
            font=("Helvetica", 14),
            text_color="#DCDDDE"
        ).grid(row=2, column=0, sticky=ctk.W, padx=10, pady=5)
        self.learn_tag = ctk.CTkEntry(
            learn_frame,
            font=("Helvetica", 14),
            fg_color="#36393F",
            text_color="#DCDDDE",
            border_color="#7289DA",
            border_width=2,
            corner_radius=10
        )
        self.learn_tag.grid(row=2, column=1, columnspan=2, sticky=ctk.EW, padx=10, pady=5)

        # Mô tả chủ đề
        ctk.CTkLabel(
            learn_frame,
            text="Chủ đề để nhóm các tin nhắn tương tự. Nếu để trống, hệ thống tự động gán.",
            font=("Helvetica", 12),
            text_color="#7289DA",
            wraplength=600
        ).grid(row=3, column=0, columnspan=3, padx=10, pady=5)

        # Nút dạy bot
        teach_button = ctk.CTkButton(
            learn_frame,
            text="Dạy Bot",
            command=self.teach_chatbot,
            fg_color="#7289DA",
            hover_color="#5A6EBB",
            font=("Helvetica", 14, "bold")
        )
        teach_button.grid(row=4, column=0, columnspan=3, pady=15)

        # Frame hiển thị intents
        intents_frame = ctk.CTkFrame(learning_tab, fg_color="#2A2D3E", corner_radius=10)
        intents_frame.pack(fill=ctk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Treeview hiển thị intents (sử dụng ttk.Treeview từ tkinter.ttk)
        self.intents_tree = ttk.Treeview(
            intents_frame,
            columns=("tag", "patterns", "responses"),
            show="headings"
        )
        self.intents_tree.heading("tag", text="Chủ đề")
        self.intents_tree.heading("patterns", text="Mẫu câu hỏi")
        self.intents_tree.heading("responses", text="Mẫu câu trả lời")
        self.intents_tree.column("tag", width=100)
        self.intents_tree.column("patterns", width=250)
        self.intents_tree.column("responses", width=250)
        self.intents_tree.pack(fill=ctk.BOTH, expand=True, padx=5, pady=5)

        # Cập nhật style cho Treeview
        style = ttk.Style()
        style.configure("Treeview", background="#36393F", foreground="#DCDDDE", fieldbackground="#36393F")
        style.configure("Treeview.Heading", background="#7289DA", foreground="#FFFFFF", font=("Helvetica", 12, "bold"))

        # Cập nhật intents ban đầu
        self.update_intents_tree()

        # Tin nhắn chào mừng
        self.append_to_chat("Chatbot", "Xin chào! Tôi là chatbot học sâu tiếng Việt. Tôi có thể giúp gì cho bạn hôm nay?")
        self.message_input.focus()

    def append_to_chat(self, sender, message):
        """Thêm tin nhắn vào lịch sử trò chuyện"""
        self.chat_history.configure(state=ctk.NORMAL)
        if self.chat_history.get("1.0", "end-1c") != "":
            self.chat_history.insert(ctk.END, "\n\n")
        sender_tag = "user" if sender == "Bạn" else "bot"
        self.chat_history.insert(ctk.END, f"{sender}: ", sender_tag)
        self.chat_history.insert(ctk.END, message)
        # Chỉ tùy chỉnh màu sắc, không dùng font trong tag_config
        self.chat_history.tag_config("user", foreground="#7289DA")
        self.chat_history.tag_config("bot", foreground="#4CAF50")
        self.chat_history.configure(state=ctk.DISABLED)
        self.chat_history.yview(ctk.END)

    def send_message(self, event=None):
        """Gửi tin nhắn và nhận phản hồi từ chatbot"""
        message = self.message_input.get().strip()
        if not message:
            return
        self.append_to_chat("Bạn", message)
        try:
            self.status_var.set("Đang xử lý...")
            self.root.update_idletasks()
            response = self.chatbot.chat(message)
            self.append_to_chat("Chatbot", response)
            self.status_var.set("Sẵn sàng")
        except Exception as e:
            self.status_var.set(f"Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {str(e)}")
        self.message_input.delete(0, ctk.END)

    def teach_chatbot(self):
        """Dạy chatbot phản hồi mới"""
        message = self.learn_message.get().strip()
        response = self.learn_response.get().strip()
        tag = self.learn_tag.get().strip() or None
        if not message or not response:
            messagebox.showwarning("Cần nhập dữ liệu", "Vui lòng nhập cả tin nhắn và phản hồi")
            return
        try:
            self.status_var.set("Đang học...")
            self.root.update_idletasks()
            result = self.chatbot.learn(message, response, tag)
            messagebox.showinfo("Học xong", result)
            self.update_intents_tree()
            self.learn_message.delete(0, ctk.END)
            self.learn_response.delete(0, ctk.END)
            self.learn_tag.delete(0, ctk.END)
            self.status_var.set("Sẵn sàng")
        except Exception as e:
            self.status_var.set(f"Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi học: {str(e)}")

    def clear_temp_files(self):
        """Xóa các file tạm thời và hiển thị thông báo"""
        try:
            self.status_var.set("Đang xóa dữ liệu tạm thời...")
            self.root.update_idletasks()
            result = self.chatbot.clear_temp_files()
            self.append_to_chat("Chatbot", result)
            self.status_var.set("Sẵn sàng")
        except Exception as e:
            self.status_var.set(f"Lỗi: {str(e)}")
            messagebox.showerror("Lỗi", f"Đã xảy ra lỗi khi xóa dữ liệu: {str(e)}")

    def clear_chat_history(self):
        """Xóa toàn bộ lịch sử trò chuyện"""
        self.chat_history.configure(state=ctk.NORMAL)
        self.chat_history.delete("1.0", ctk.END)
        self.chat_history.configure(state=ctk.DISABLED)
        self.append_to_chat("Chatbot", "Lịch sử trò chuyện đã được xóa.")

    def update_intents_tree(self):
        """Cập nhật danh sách intents trong treeview"""
        for item in self.intents_tree.get_children():
            self.intents_tree.delete(item)
        for intent in self.chatbot.intents["intents"]:
            patterns = ", ".join(intent["patterns"][:3]) + ("..." if len(intent["patterns"]) > 3 else "")
            responses = ", ".join(intent["responses"][:3]) + ("..." if len(intent["responses"]) > 3 else "")
            self.intents_tree.insert("", ctk.END, values=(intent["tag"], patterns, responses))

if __name__ == "__main__":
    root = ctk.CTk()
    app = ChatbotApp(root)
    root.mainloop()