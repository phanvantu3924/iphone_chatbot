from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# =============================
# SETUP FOLDER
# =============================
os.makedirs('./models', exist_ok=True)

print("=" * 60)
print("ĐANG TẢI AI MODELS OFFLINE...")
print("=" * 60)

# =============================
# 1) TẢI EMBEDDING MODEL
# =============================
print("\n[1/2] Tải Embedding Model (MiniLM)...")

embedding_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
embedding_model.save("./models/embedding_model")

print("✅ Embedding Model xong!")


# =============================
# 2) TẢI Qwen2.5-0.5B-INSTRUCT
# =============================
print("\n[2/2] Tải LLM Model (Qwen2.5-0.5B-Instruct)...")
print("⏰ Đợi 1–3 phút... (model nhỏ nên tải nhanh)")

LLM_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LLM_PATH = "./models/llm_model"

os.makedirs(LLM_PATH, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(
    LLM_ID,
    trust_remote_code=True,
    local_files_only=False

)

model = AutoModelForCausalLM.from_pretrained(
    LLM_ID,
    trust_remote_code=True,
    local_files_only=True
)

tokenizer.save_pretrained(LLM_PATH)
model.save_pretrained(LLM_PATH)

print("✅ Qwen 0.5B đã tải xong!")
print("=" * 60)
print("🎉 Mọi thứ đã sẵn sàng!")
print("=" * 60)
