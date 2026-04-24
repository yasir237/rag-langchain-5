# LangChain ile Vektör Store ve Benzerlik Araması
> Chunk'ları ChromaDB'ye atma ve doğal dil sorusuyla en alakalı içeriği bulma — RAG serisinin 5. adımı

[![Colab'da Aç](https://img.shields.io/badge/Colab'da%20Aç-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/yasir237/rag-langchain-5/blob/main/rag_langchain_5.ipynb)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)

---

## Problem

Chunk'ları embed ettin, elinde vektörler var — ama ne yapacaksın? Her soru geldiğinde tüm vektörlerle tek tek karşılaştırma yapmak hem yavaş hem ölçeklenmez. Üstelik "hangi chunk bu soruya en yakın?" sorusuna programatik olarak nasıl cevap vereceksin?

## Çözüm

ChromaDB, vektörleri saklar ve bir sorgu geldiğinde cosine similarity ile en yakın chunk'ları anında bulur. Tek bir `similarity_search()` çağrısı yeterli — sen sadece soruyu yazarsın, Chroma ilgili içeriği getirir. Bu adım RAG pipeline'ının **retrieval** motorudur.

---

## Pipeline Mimarisi

```
Önceki adımdan gelen temiz chunk'lar + Gemini vektörleri
        │
        ▼
┌───────────────────────────────────────┐
│            ChromaDB                   │
│  Chroma.from_documents(texts,         │
│                        embeddings)    │
│                                       │
│  Her chunk → vektör olarak saklanır   │
└───────────────────────────────────────┘
        │
        ▼
  Kullanıcı sorusu: "Autoencoder nasıl çalışır?"
        │
        ▼
┌───────────────────────────────────────┐
│         similarity_search()           │
│                                       │
│  1. Soruyu embed et (Gemini)          │
│  2. Tüm chunk vektörleriyle karşılaştır│
│  3. En yakın k chunk'ı döndür         │
└───────────────────────────────────────┘
        │
        ▼
  En alakalı chunk'lar → bir sonraki adımda LLM'e gidecek
```

| Bileşen | Görevi |
|---|---|
| `Chroma.from_documents()` | Chunk'ları ve vektörleri tek seferde store'a yükler |
| `similarity_search(query, k)` | Soruya en yakın k chunk'ı cosine similarity ile bulur |
| `k` parametresi | Kaç chunk döneceğini belirler — küçük veri setinde 2-3 yeterli |
| `doc.page_content` | Bulunan chunk'ın ham metnine erişim |

---

## Kullanılan Teknolojiler

![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=flat&logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6B35?style=flat&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=flat&logoColor=white)
![Gemini](https://img.shields.io/badge/Gemini_Embedding-4285F4?style=flat&logo=google&logoColor=white)
![Llama](https://img.shields.io/badge/Llama_3.1_8B-0467DF?style=flat&logo=meta&logoColor=white)
![Python](https://img.shields.io/badge/Python_3-3776AB?style=flat&logo=python&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)

---

## Kurulum

```bash
pip install langchain langchain-core langchain-groq
pip install langchain-text-splitters langchain-google-genai
pip install langchain-chroma chromadb
```

### API Anahtarları

Google Colab **Secrets** sekmesine ekle:
- `GROQ_API_KEY` → [console.groq.com](https://console.groq.com)
- `GOOGLE_API_KEY` → [aistudio.google.com](https://aistudio.google.com)

---

## Kullanım

```python
from langchain_chroma import Chroma

# Chunk'ları vektör store'a yükle
# texts     → önceki adımdan gelen Document listesi
# embeddings → Gemini embedding modeli
db = Chroma.from_documents(texts, embeddings)

# Doğal dil sorusuyla arama yap
query = "Autoencoder nasıl çalışır?"
docs = db.similarity_search(query, k=2)

# Sonuçları göster
print(f"Sorulan soru: {query}")
print(f"Bulunan chunk sayısı: {len(docs)}")

for i, doc in enumerate(docs):
    print(f"\n--- En alakalı chunk {i+1} ---")
    print(doc.page_content)
```

---

## Neden ChromaDB?

| | ChromaDB | FAISS | Pinecone |
|---|---|---|---|
| Kurulum | pip, sıfır config | pip | Bulut, kayıt gerekir |
| Kalıcı depolama | Opsiyonel | Hayır | Evet |
| Ücretsiz | Tamamen | Tamamen | Limit var |
| Öğrenme için | ✅ İdeal | ⚠️ Orta | ❌ Karmaşık |

Öğrenme aşaması için ChromaDB; kurulumu en kolay, LangChain entegrasyonu tek satır.

---

## Seri İçindeki Yeri

Bu notebook, LangChain ile kurulan RAG serisinin **5. adımıdır.**

```
[1] ✅ Mesaj yapısı ve LLM bağlantısı
[2] ✅ PromptTemplate ile şablonlu prompt
[3] ✅ Çoklu zincir kurma ve zincirleri bağlama
[4] ✅ Metin bölme ve embedding
[5] ✅ Vektör store ve benzerlik araması        ← bu repo
```

Her adım bir sonrakine köprü kuruyor.  
Serinin tamamını takip etmek için LinkedIn profilimi ziyaret edebilirsin 👇

---

## Bağlantı

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/yasir-alrawi-12814521a/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/yasir237)

---

## Lisans

MIT
