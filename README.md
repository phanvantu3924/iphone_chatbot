# iPhone RAG Chatbot: Hệ thống Tư vấn Điện thoại Offline

Dự án này triển khai một Chatbot tư vấn sản phẩm thông minh sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)** để cung cấp thông tin chính xác, không bịa đặt (No Hallucination) về các dòng iPhone. Hệ thống được tối ưu hóa để chạy hiệu quả trong môi trường **Offline** và trên các máy tính có **tài nguyên hạn chế (CPU/RAM 8GB)**.

## 🌟 Tính năng Chính

* Tư vấn Ngữ nghĩa (Semantic Search): Tìm kiếm thông số kỹ thuật (chip, pin, giá, camera) dựa trên ý nghĩa của câu hỏi, vượt qua giới hạn tìm kiếm từ khóa.
* Safe RAG (Không Hallucination): Sử dụng **Prompt Engineering** nghiêm ngặt và **Ngưỡng cắt Độ tương đồng (Similarity Threshold)** để buộc LLM chỉ trả lời dựa trên Context được truy xuất từ file dữ liệu.
* Xử lý Ngoại lệ: Tự động phát hiện và từ chối các câu hỏi nằm ngoài phạm vi (ví dụ: hỏi về Samsung, Xiaomi) và các câu hỏi không rõ ràng.
* Tối ưu Hiệu năng: Sử dụng mô hình **Qwen2.5-0.5B-Instruct** siêu nhẹ để đảm bảo tốc độ phản hồi chấp nhận được trên môi trường CPU.
* Client-Server: Backend được xây dựng bằng **FastAPI** cung cấp API RESTful cho Frontend.

## 💡 Kiến trúc và Công nghệ

* **Backend (RAG Core):** Python, FastAPI (Điều phối luồng Retrieval và Generation).
* **LLM (Generation):** Qwen2.5-0.5B-Instruct (Mô hình sinh văn bản, tối ưu cho CPU).
* **Embeddings:** Sentence Transformer (`paraphrase-MiniLM-L6-v2`) (Chuyển đổi văn bản thành vector).
* **Vector Database:** FAISS (IndexFlatL2) (Lưu trữ và tìm kiếm vector Embeddings tốc độ cao).
* **Data Source:** `phones.txt` (File dữ liệu tĩnh chứa thông số iPhone).

---

## ⚙️ Hướng dẫn Cài đặt và Khởi chạy

Để chạy dự án này, bạn cần có **Python 3.8+** và **pip** đã được cài đặt.

### Bước 1: Chuẩn bị Môi trường

1.  **Clone repository** và chuyển đến thư mục dự án.
2.  **Tạo và kích hoạt môi trường ảo (venv):** `python -m venv venv`
3.  **Kích hoạt môi trường ảo:** (Ví dụ cho Windows): `.\venv\Scripts\activate`

### Bước 2: Cài đặt Thư viện

Sử dụng file `requirements.txt` để cài đặt tất cả các dependencies.

`pip install -r requirements.txt`

### Bước 3: Tải Mô hình AI và Xây dựng FAISS Index

Chạy script này để tải mô hình LLM và Embedding Model, sau đó xây dựng Index từ dữ liệu `phones.txt`.

1.  **Tải AI Models và lưu vào thư mục ./models:** `python download_models.py`
2.  **Xây dựng Index từ phones.txt:** `python build_index.py`

### Bước 4: Khởi động Server Backend

Chạy server FastAPI:

`python backend.py`

Server sẽ khởi động tại địa chỉ: `http://0.0.0.0:8000`.

## 💬 Hướng dẫn Sử dụng (Test Cases)

Mở trình duyệt tại địa chỉ `http://localhost:8000` hoặc mở trực tiếp file `index.html` để tương tác.

**Lưu ý:** Để có kết quả tốt nhất từ mô hình 0.5B, hãy hỏi những câu hỏi có ngữ nghĩa rõ ràng:

* **Tra cứu Chi tiết:** Ví dụ: *Giá bán của iPhone 17 Pro Max là bao nhiêu?*
* **So sánh:** Ví dụ: *So sánh RAM và pin giữa iPhone 15 và iPhone 16*
* **Tư vấn Ngữ nghĩa:** Ví dụ: *Điện thoại nào có pin trâu nhất và rẻ nhất?*
* **Kiểm tra Ngoại lệ:** Ví dụ: *Giá Samsung S23 Ultra là bao nhiêu?*

---
*Dự án được thực hiện bởi **Phan Văn Tú** cho Đồ án Tốt nghiệp 2025.*
*Giáo viên Hướng dẫn: **Th.S Lê Đức Quang**.*
