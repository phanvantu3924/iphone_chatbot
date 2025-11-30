# backend.py — OFFLINE RAG + Qwen0.5B (NO ACCELERATE VERSION)

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import faiss
import numpy as np
import torch
from typing import List, Dict
import os
import re

# ===========================================
# FASTAPI
# ===========================================
app = FastAPI(title="iPhone Chatbot – Offline RAG No-Hallucination")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("🚀 STARTING OFFLINE RAG SYSTEM...\n")

# ===========================================
# LOAD EMBEDDING MODEL
# ===========================================
print("[1/5] Loading embedding model...")
embedding_model = SentenceTransformer("./models/embedding_model")
print("✅ Embedding model loaded!\n")

# ===========================================
# LOAD QWEN 0.5B — KHÔNG DÙNG device_map!
# ===========================================
print("[2/5] Loading Qwen2.5-0.5B-Instruct...")

LLM_MODEL_PATH = "./models/llm_model"

tokenizer = AutoTokenizer.from_pretrained(
    LLM_MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

llm_model = AutoModelForCausalLM.from_pretrained(
    LLM_MODEL_PATH,
    trust_remote_code=True,
    dtype=torch.float32,
    low_cpu_mem_usage=True,
    local_files_only=True
)

# CHUYỂN MODEL SANG CPU (KHÔNG cần accelerate)
llm_model.to("cpu")
llm_model.eval()
print("✅ Qwen 0.5B loaded!\n")

# ===========================================
# LOAD phones.txt
# ===========================================
print("[3/5] Loading phones.txt...")

def load_phone_data(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    products = []

    for block in blocks:
        lines = block.strip().split("\n")
        if not lines:
            continue

        p = {"name": lines[0], "full": block}
        for line in lines[1:]:
            if ":" in line:
                k, v = line.split(":", 1)
                p[k.strip().lower()] = v.strip()

        products.append(p)

    return products

phones = load_phone_data("phones.txt")
print(f"✅ Loaded {len(phones)} products!\n")

# ===========================================
# LOAD FAISS INDEX
# ===========================================
print("[4/5] Loading FAISS index...")
if not os.path.exists("phones.index"):
    raise FileNotFoundError("phones.index not found. Run build_index.py first.")
faiss_index = faiss.read_index("phones.index")
print("✅ FAISS index loaded!\n")

# ===========================================
# PRECOMPUTE EMBEDDINGS
# ===========================================
print("[5/5] Precomputing embeddings...")

def structured(p: Dict) -> str:
    keys = ["giá", "màn hình", "chip", "camera", "pin"]
    return p["name"] + " | " + " | ".join([p[k] for k in keys if k in p])

product_texts = [structured(p) for p in phones]
product_embs = embedding_model.encode(product_texts, convert_to_numpy=True).astype("float32")
product_norms = np.linalg.norm(product_embs, axis=1)

print("✅ Embedding ready!\n")

THRESHOLD = 0.20

# ===========================================
# RETRIEVE
# ===========================================
def retrieve(q: str) -> List[Dict]:
    q_emb = embedding_model.encode([q], convert_to_numpy=True)[0]
    q_norm = np.linalg.norm(q_emb)

    dist, idx = faiss_index.search(np.expand_dims(q_emb, 0), 5)

    results = []
    for i, d in zip(idx[0], dist[0]):
        p = phones[i].copy()
        # Tính Cosine Similarity
        cos = float(np.dot(q_emb, product_embs[i]) / (q_norm * product_norms[i] + 1e-9))
        p["distance"] = float(d)
        p["similarity"] = cos
        results.append(p)

    return sorted(results, key=lambda x: x["distance"])

# ===========================================
# LLM ANSWER
# ===========================================
def llm_answer(query: str, products: List[Dict]) -> str:
    top = products[0]
    
    # Kiểm tra ngữ cảnh có liên quan đủ không
    if top["similarity"] < THRESHOLD:
        return "Em xin lỗi, em chưa tìm thấy thông tin chi tiết này trong dữ liệu sản phẩm."

    ctx = structured(top)

    # PROMPT NGHIÊM NGẶT (FIX LỖI TRẢ LỜI LẠC ĐỀ/HALLUCINATION)
    prompt = f"""
Bạn là một TRỢ LÝ TƯ VẤN SẢN PHẨM IPHONE CHUYÊN NGHIỆP, lịch sự.
Nhiệm vụ của bạn là CHỈ TRẢ LỜI CÂU HỎI của khách hàng dựa trên DỮ LIỆU SẢN PHẨM được cung cấp.

[DỮ LIỆU SẢN PHẨM]
{ctx}

[CÂU HỎI]
{query}

[HƯỚNG DẪN BẮT BUỘC]
1. KHÔNG được sử dụng bất kỳ lời chào, kết thúc thư, hoặc mẫu form nào.
2. TUYỆT ĐỐI KHÔNG BỊA ĐẶT hoặc TƯ VẤN THÔNG TIN KHÔNG CÓ TRONG DỮ LIỆU.
3. Nếu không đủ thông tin, CHỈ trả lời: "Em xin lỗi, em chưa tìm thấy thông tin chi tiết này trong dữ liệu sản phẩm."
4. Trả lời ngắn gọn, tối đa 3 câu.

TRẢ LỜI:
"""

    tks = tokenizer(prompt, return_tensors="pt").to("cpu")

    with torch.no_grad():
        out = llm_model.generate(
            **tks,
            max_new_tokens=150,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    
    # POST-PROCESSING (FIX LỖI MẪU FORM VÀ DÒNG RÁC)
    if "TRẢ LỜI:" in text:
        text = text.split("TRẢ LỜI:")[-1].strip()
    
    # Loại bỏ các chuỗi rác/mẫu form (Hallucination)
    text = re.sub(r"\[.*?\]", "", text, flags=re.IGNORECASE) # Loại bỏ bất kỳ chuỗi nào trong ngoặc []
    

    # 3. Loại bỏ các dòng trống hoặc rác còn sót lại
    lines = []
    for line in text.split('\n'):
        line_clean = line.strip()
        # Loại bỏ các chuỗi lỗi Hallucination/Fallback phổ biến
        if any(keyword in line_clean for keyword in [
            "Em xin lỗi vì sự nhầm lẫn", 
            "Em cần thêm thông tin để giúp đỡ", 
            "Chúc em thành công", 
            "Trân trọng", 
            "Em xin lỗi, tôi không có thông tin chi tiết về",
            "liên hệ với tôi",
            "email hotline",
            "info@iphone.com",
            "Hãy nhớ rằng",
            "0987654321"
        ]):
            continue # Bỏ qua dòng chứa các lỗi này
        
        # Loại bỏ các ký tự đầu dòng không cần thiết
        if line_clean:
            lines.append(line_clean)
            
    return '\n'.join(lines)

# ===========================================
# API
# ===========================================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    relevant_products: list

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    q = req.message
    
    # KIỂM TRA NGOẠI LỆ (FIX LỖI HỎI SẢN PHẨM KHÔNG CÓ)
    if any(brand in q.lower() for brand in ["samsung", "xiaomi", "oppo", "android"]):
        ans = "Em xin lỗi, em chỉ có dữ liệu về các dòng iPhone. Xin quý khách vui lòng hỏi về sản phẩm iPhone."
        return ChatResponse(response=ans, relevant_products=[])

    results = retrieve(q)
    ans = llm_answer(q, results)
    
    # XỬ LÝ LỖI KHÔNG TÌM THẤY CONTEXT TRƯỚC KHI TRẢ LỜI
    if ans == "Em xin lỗi, em chưa tìm thấy thông tin chi tiết này trong dữ liệu sản phẩm.":
        # Nếu LLM trả lời fallback, ta vẫn kiểm tra ngưỡng
        if results and results[0]["similarity"] < THRESHOLD:
            return ChatResponse(response=ans, relevant_products=[])

    return ChatResponse(response=ans, relevant_products=results[:2])

@app.get("/")
def root():
    return {"status": "running", "products_loaded": len(phones)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)