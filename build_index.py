# build_index.py (FAISS FIX CHUẨN 2025)
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def load_phone_data(file_path: str):
    """
    Chỉ embed:
    - Tên sản phẩm
    - Giá
    - Chip
    - Màn hình
    => Match chính xác tuyệt đối khi search tên máy.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = content.strip().split("\n\n")
    phone_list = []

    for block in blocks:
        lines = block.strip().split("\n")
        
        if len(lines) == 0:
            continue

        name = lines[0].strip()

        price = ""
        chip = ""
        man_hinh = ""

        for line in lines:
            lower = line.lower()

            if lower.startswith("giá"):
                price = line.strip()
            elif lower.startswith("chip"):
                chip = line.strip()
            elif "màn hình" in lower:
                man_hinh = line.strip()

        text = name + " | " + price + " | " + chip + " | " + man_hinh
        phone_list.append(text.strip())

    return phone_list


print("\n==== TẠO FAISS INDEX CHUẨN ====\n")

print("🔍 Loading embedding model...")
embedding_model = SentenceTransformer("./models/embedding_model")
print("✔ Loaded!")

print("\n📄 Reading phones.txt...")
texts = load_phone_data("phones.txt")
print(f"✔ Loaded {len(texts)} products!")

print("\n➡ Data after cleaning:")
for t in texts:
    print(" •", t)

print("\n✨ Generating embeddings...")
embeddings = embedding_model.encode(texts, convert_to_numpy=True)
dimension = embeddings.shape[1]

print("\n⚙ Creating FAISS index...")
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype("float32"))

faiss.write_index(index, "phones.index")

print("\n🎉 DONE! phones.index đã được tạo mới và tối ưu.")
