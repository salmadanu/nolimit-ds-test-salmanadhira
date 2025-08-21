
# RAG Chatbot

### Q&A Chatbot untuk menjawab pertanyaan seputar *computational media analysis*

Dataset merupakan 30 *file* dalam format **PDF** (60,5 MB) berupa jurnal, buku, dan artikel dalam Bahasa Inggris seputar analisis media secara komputasional seperti analisis *framing*, deteksi berita palsu, dan deteksi propaganda menggunakan metode komputasional seperti *deep learning* dan pemodelan topik.



## Flowchart
Nanti masukin flowchart disini

Sedikit tuning dilakukan untuk menemukan embedding parameter yang menghasilkan jawaban terbaik, yang didokumentasikan melalui log iterasi(link ke excel).
## ⚙️ Tech Stack

**PDF Extraction & Preprocess:** 
- `PyMuPDF`
- `Re`

**Embedding Model:**
- `LangChain`, `HuggingFaceEmbeddings`
- `sentence-transformers/all-mpnet-base-v2`

**Embedding Model:**
- `FAISS`

**Large Language Model (LLM):**
- `google/flan-t5-large`




## 📙 Menjalankan dengan Google Colab
Colab dapat diakses melal
## 🔄 Deployment dengan Streamlit Cloud
## 🔄 Deployment dengan HuggingFace Spaces
## 💡 Saran Perbaikan
