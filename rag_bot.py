import os
import pickle
import requests
import numpy as np
import faiss
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()


@dataclass
class SearchResult:
    content: str
    source: str
    faiss_score: float
    chunk_id: int
    rerank_score: float = 0.0


class Reranker:
    def __init__(self):
        print("Загрузка Cross-Encoder...")
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
        print("Cross-Encoder готов")

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        if not results:
            return results
        pairs = [(query, r.content[:512]) for r in results]
        scores = self.model.predict(pairs)
        for r, score in zip(results, scores):
            r.rerank_score = float(score)
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]


class YandexGPTClient:
    def __init__(self):
        self.api_key = os.getenv('YANDEX_API_KEY')
        self.folder_id = os.getenv('YANDEX_FOLDER_ID')
        self.url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            return "YandexGPT не настроен"

        headers = {"Content-Type": "application/json", "Authorization": f"Api-Key {self.api_key}"}
        data = {
            "modelUri": f"gpt://{self.folder_id}/yandexgpt/latest",
            "completionOptions": {"stream": False, "temperature": 0.3, "maxTokens": 2000},
            "messages": [{"role": "user", "text": prompt}]
        }
        try:
            response = requests.post(self.url, headers=headers, json=data, timeout=30)
            return response.json()['result']['alternatives'][0]['message']['text']
        except Exception as e:
            return f"Ошибка: {e}"


class QuantumForgeRAGBot:
    def __init__(self, index_path="faiss.index", metadata_path="metadata.pkl", use_reranker=True):
        print("Загрузка индекса...")
        self.index = faiss.read_index(index_path)
        with open(metadata_path, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data['chunks']
            self.metadatas = data['metadatas']

        print("Загрузка эмбеддингов...")
        self.embeddings = HuggingFaceEmbeddings(
            # model_name="sentence-transformers/all-MiniLM-L6-v2", # использовал другую модель поскольку для киррилицы эта плохо работает
            # model_name="intfloat/multilingual-e5-small",
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            # model_name="DeepPavlov/rubert-base-cased-sentence",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

        self.use_reranker = use_reranker
        self.reranker = Reranker() if use_reranker else None
        self.yandex = YandexGPTClient()

        print(f"Бот готов. Чанков: {self.index.ntotal}")
        print(f"Reranking: {'ВКЛ' if use_reranker else 'ВЫКЛ'}")

    def search(self, query: str, k: int = 40) -> List[SearchResult]:
        q_emb = self.embeddings.embed_query(query)
        q_np = np.array([q_emb]).astype('float32')
        distances, indices = self.index.search(q_np, k)
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append(SearchResult(
                    content=self.chunks[idx],
                    source=self.metadatas[idx]['source'],
                    faiss_score=float(dist),
                    chunk_id=self.metadatas[idx]['chunk_id']
                ))
        return results

    def retrieve(self, query: str) -> List[SearchResult]:
        results = self.search(query, k=20)
        if self.use_reranker and results:
            results = self.reranker.rerank(query, results, top_k=5)
        return results

    def build_prompt(self, query: str, context: str) -> str:
        """Промпт с Few-shot и Chain-of-Thought"""
        return f"""Ты — интеллектуальный помощник QuantumForge Software.

## КЛЮЧЕВЫЕ ПРАВИЛА:
1. Используй ТОЛЬКО информацию из КОНТЕКСТА
2. Если в контексте НЕТ прямого ответа на вопрос или четкого определения — ОБЯЗАТЕЛЬНО скажи "Я не знаю, но вероятно это ..." и дай ответ сопоставляя все данные в контексте
3. НЕ делай выводы и НЕ сопоставляй факты, если они не из контекста
4. НЕ заменяй слова (например, "vedun" на "маг") — используй термины как в контексте
5. Если вопрос общий (например, "маг", "волшебник") — отвечай "Я не знаю, но вероятно это ..." и дай ответ сопоставляя все данные в контексте
6. Проверяй весь контекст полностью, если есть четкое опредления термина, то это приоритетнее чем предположение, соответственно дай ответ четкого определения, вне зависимости от индекса удаленности эмбединга.

## ФОРМАТ ОТВЕТА (ОБЯЗАТЕЛЬНО):
Твой ответ ДОЛЖЕН начинаться с рассуждений в формате:
1. Анализ вопроса: [что спрашивает пользователь]
2. Поиск в контексте: [какая информация найдена]
3. Вывод: [можно ли ответить]
4. Ответ: [финальный ответ пользователю]

## ПРИМЕРЫ ПРАВИЛЬНЫХ ОТВЕТОВ (Few-shot):

Пример 1:
Контекст: [radomir_svetozarov.txt] Radomir Svetozarov родился в канун Купалы. Его родители были убиты Chernobogom Temnoslavom.
Вопрос: Кто такой Radomir Svetozarov?
1. Анализ вопроса: пользователь спрашивает о персонаже Radomire Svetozarove.
2. Поиск в контексте: в файле radomir_svetozarov.txt есть информация о его рождении и гибели родителей.
3. Вывод: информация достаточна для ответа.
4. Ответ: Radomir Svetozarov — главный герой. Его родители были убиты Chernobogom Temnoslavom. Источник: radomir_svetozarov.txt

Пример 2:
Контекст: [vedovstvo_magic.txt] Ведовство — древнее искусство изменения реальности через слово, жест и намерение.
Вопрос: Что такое ведовство?
1. Анализ вопроса: пользователь спрашивает определение ведовства.
2. Поиск в контексте: в файле vedovstvo_magic.txt есть прямое определение.
3. Вывод: информация достаточна для ответа.
4. Ответ: Ведовство — это древнее искусство изменения реальности через слово, жест и намерение. Источник: vedovstvo_magic.txt

Пример 3:
Контекст: (пусто)
Вопрос: Какой пароль у администратора?
1. Анализ вопроса: пользователь спрашивает пароль.
2. Поиск в контексте: контекст пуст, информация отсутствует.
3. Вывод: ответить невозможно.
4. Ответ: Я не знаю. В базе знаний нет информации по этому вопросу.

---
КОНТЕКСТ:
{context if context else "(пусто)"}

ВОПРОС: {query}

ТВОЙ ОТВЕТ (обязательно в формате 1. Анализ вопроса: ... 2. Поиск в контексте: ... 3. Вывод: ... 4. Ответ: ...):"""

    def ask(self, query: str) -> Dict[str, Any]:
        start = datetime.now()
        results = self.retrieve(query)

        if results:
            context = "\n\n".join([f"[{r.source}] {r.content[:500]}" for r in results[:3]])
        else:
            context = ""

        prompt = self.build_prompt(query, context)
        answer = self.yandex.generate(prompt)

        return {
            'query': query,
            'answer': answer,
            'sources': [r.source for r in results[:3]] if results else [],
            'found': bool(results),
            'response_time_ms': (datetime.now() - start).total_seconds() * 1000
        }

    def interactive_mode(self):
        print("QuantumForge RAG Bot v1.0")
        print(f"Reranking: {'ВКЛ' if self.use_reranker else 'ВЫКЛ'}")
        print("Техники: RAG | Few-shot | Chain-of-Thought")
        print("Команды: /exit, /sources")

        show_sources = False
        while True:
            q = input("\n❓ Вы: ").strip()
            if q.lower() in ['/exit', 'quit']:
                break
            if q.lower() == '/sources':
                show_sources = not show_sources
                print(f"Показ источников: {'ВКЛ' if show_sources else 'ВЫКЛ'}")
                continue

            resp = self.ask(q)
            print(f"\n{resp['answer']}")
            if show_sources and resp['sources']:
                print(f"\n{', '.join(resp['sources'])}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--interactive', '-i', action='store_true')
    p.add_argument('--query', '-q', type=str)
    p.add_argument('--no-rerank', action='store_true')
    args = p.parse_args()

    bot = QuantumForgeRAGBot(use_reranker=not args.no_rerank)

    if args.interactive:
        bot.interactive_mode()
    elif args.query:
        resp = bot.ask(args.query)
        print(f"\n{resp['answer']}")
    else:
        print("python rag_bot_final.py --interactive")


if __name__ == "__main__":
    main()