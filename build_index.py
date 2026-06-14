import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
import pickle

embeddings_model = HuggingFaceEmbeddings(
    # model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    # model_name="DeepPavlov/rubert-base-cased-sentence",
    model_kwargs={'device': 'cpu'}
)

embeding_size = 384 # Размер эмбеддингов: 384
chank_size = 1000
chank_overlap = 200
chunks = []
metadatas = []

# Загрузка документов
for filename in os.listdir('knowledge_base'):
    with open(f'knowledge_base/{filename}', 'r') as f:
        content = f.read()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chank_size,
            chunk_overlap=chank_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        doc_chunks = splitter.split_text(content)
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            metadatas.append({
                'source': filename,
                'chunk_id': i,
                'title': filename.replace('.txt', '')
            })

# Генерация эмбеддингов
embeddings = embeddings_model.embed_documents(chunks)
embeddings_np = np.array(embeddings).astype('float32')

# Создание FAISS индекса
index = faiss.IndexFlatL2(embeding_size)  # L2 distance
index.add(embeddings_np)

# Сохранение
faiss.write_index(index, 'faiss.index')
with open('metadata.pkl', 'wb') as f:
    pickle.dump({'chunks': chunks, 'metadatas': metadatas}, f)

print(f"Создано чанков: {len(chunks)}")
print(f"Размер индекса: {index.ntotal}")