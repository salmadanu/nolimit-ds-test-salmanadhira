
# RAG Chatbot

### Q&A Chatbot untuk menjawab pertanyaan seputar *computational media analysis*

Dataset merupakan 30 *file* dalam format **PDF** (60,5 MB) berupa jurnal, buku, dan artikel dalam Bahasa Inggris seputar analisis media secara komputasional seperti analisis *framing*, deteksi berita palsu, dan deteksi propaganda menggunakan metode komputasional seperti *deep learning* dan pemodelan topik.



## Flowchart
![RAG Chatbot Flowchart](flowchart.png)
Tuning pada beberapa parameter dan model dilakukan untuk menemukan embedding parameter yang menghasilkan jawaban terbaik, yang didokumentasikan melalui [log iterasi](https://docs.google.com/spreadsheets/d/1fvNhsdH15O83DG2wBWyW-Ziuv8HC-XclSvzU3IECP48/edit?usp=sharing).

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




## 📙 Menjalankan dengan [Google Colab](https://colab.research.google.com/drive/1EsHIbeDpwCz_GL9gZOdVpnw9S5NC809c?usp=sharing)


## 🔄 Deployment dengan Streamlit Cloud
## 🔄 Deployment dengan HuggingFace Spaces
## 💡 Saran Perbaikan
