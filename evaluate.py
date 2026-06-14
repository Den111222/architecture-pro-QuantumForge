import json
import pickle
import re
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from mawo_pymorphy3 import MorphAnalyzer
from rag_bot_secure import QuantumForgeRAGBot

morph = MorphAnalyzer()


@dataclass
class QueryResult:
    query: str
    expected_keywords: List[str]
    should_have_answer: bool
    actual_answer: str
    sources: List[str]
    found: bool
    keywords_found: List[str]
    keywords_missed: List[str]
    score: float
    response_time_ms: float
    timestamp: str


class QualityEvaluator:
    def __init__(self, bot=None, golden_file="golden_questions.json"):
        self.bot = bot
        self.results: List[QueryResult] = []
        self.golden_file = golden_file
        self.golden_questions = self._load_golden_questions()

    def _load_golden_questions(self) -> List[Dict]:
        with open(self.golden_file, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        for q in questions:
            if isinstance(q.get('should_have_answer'), str):
                q['should_have_answer'] = q['should_have_answer'].lower() == 'true'
            if q.get('source') == 'None':
                q['source'] = None
        print(f"Загружено {len(questions)} золотых вопросов из {self.golden_file}")
        return questions

    def _lemmatize_text(self, text: str) -> set:
        """Лемматизация текста через mawo-pymorphy3"""
        if not text:
            return set()

        # Разбиваем на слова, убираем пунктуацию
        words = re.findall(r'[а-яА-ЯёЁa-zA-Z]+', text.lower())

        lemmas = set()
        for word in words:
            if len(word) > 1:
                try:
                    parsed = morph.parse(word)[0]
                    lemma = parsed.normal_form
                    lemmas.add(lemma)
                except:
                    lemmas.add(word)
        return lemmas

    def check_keywords(self, answer: str, expected: List[str]) -> Tuple[List[str], List[str]]:
        """Проверка ключевых слов с лемматизацией"""
        answer_lemmas = self._lemmatize_text(answer)

        found = []
        missed = []

        for kw in expected:
            kw_lemmas = self._lemmatize_text(kw)
            if kw_lemmas & answer_lemmas:  # пересечение множеств
                found.append(kw)
            else:
                missed.append(kw)

        return found, missed

    def evaluate_answer(self, result: QueryResult) -> float:
        if not result.should_have_answer:
            if "не знаю" in result.actual_answer.lower() or "нет информации" in result.actual_answer.lower():
                return 2.0
            return 0.0

        if not result.found:
            return 0.0

        found_count = len(result.keywords_found)
        total_count = len(result.expected_keywords)
        if total_count == 0:
            return 1.0 if result.found else 0.0

        ratio = found_count / total_count
        if ratio >= 0.8:
            return 2.0
        if ratio >= 0.5:
            return 1.0
        return 0.0

    def run_single(self, item: Dict) -> QueryResult:
        start = time.time()
        response = self.bot.ask(item["query"])
        end = time.time()

        found_keywords, missed_keywords = self.check_keywords(
            response['answer'],
            item["expected_keywords"]
        )

        result = QueryResult(
            query=item["query"],
            expected_keywords=item["expected_keywords"],
            should_have_answer=item["should_have_answer"],
            actual_answer=response['answer'],
            sources=response.get('sources', []),
            found=response.get('found', False),
            keywords_found=found_keywords,
            keywords_missed=missed_keywords,
            score=0.0,
            response_time_ms=(end - start) * 1000,
            timestamp=datetime.now().isoformat()
        )
        result.score = self.evaluate_answer(result)
        return result

    def run_all(self) -> List[QueryResult]:
        print("Запуск оценки качества RAG системы")

        for i, item in enumerate(self.golden_questions):
            print(f"\n[{i + 1}/{len(self.golden_questions)}] {item['query'][:50]}...")
            result = self.run_single(item)
            self.results.append(result)

            if result.score == 2:
                print(f"   Отлично (score: {result.score})")
            elif result.score == 1:
                print(f"   Частично (score: {result.score})")
            else:
                print(f"   Плохо (score: {result.score})")

        return self.results

    def generate_report(self) -> Dict[str, Any]:
        total = len(self.results)
        if total == 0:
            return {"error": "Нет результатов"}

        excellent = sum(1 for r in self.results if r.score == 2)
        partial = sum(1 for r in self.results if r.score == 1)
        poor = sum(1 for r in self.results if r.score == 0)

        should_answer = [r for r in self.results if r.should_have_answer]
        should_not_answer = [r for r in self.results if not r.should_have_answer]

        should_answer_score = sum(r.score for r in should_answer) / len(should_answer) if should_answer else 0
        should_not_answer_score = sum(r.score for r in should_not_answer) / len(
            should_not_answer) if should_not_answer else 0

        return {
            "total_queries": total,
            "excellent": excellent,
            "partial": partial,
            "poor": poor,
            "success_rate": (excellent + partial) / total * 100,
            "should_answer_avg_score": should_answer_score,
            "should_not_answer_avg_score": should_not_answer_score,
            "avg_response_time_ms": sum(r.response_time_ms for r in self.results) / total,
            "timestamp": datetime.now().isoformat()
        }

    def save_logs(self, filepath: str = "evaluation_logs.jsonl"):
        with open(filepath, 'w', encoding='utf-8') as f:
            for r in self.results:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + '\n')
        print(f"\nЛоги сохранены в {filepath}")

    def save_csv(self, filepath: str = "evaluation_logs.csv"):
        import csv
        with open(filepath, 'w', encoding='utf-8', newline='') as f:
            if self.results:
                writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
                writer.writeheader()
                for r in self.results:
                    writer.writerow(asdict(r))
        print(f"CSV сохранён в {filepath}")

    def print_summary(self):
        report = self.generate_report()

        print("ОТЧЁТ О КАЧЕСТВЕ БАЗЫ ЗНАНИЙ")
        print(f"Всего запросов:        {report['total_queries']}")
        print(f"Отлично (score=2):     {report['excellent']}")
        print(f"Частично (score=1):    {report['partial']}")
        print(f"Плохо (score=0):       {report['poor']}")
        print(f"Успешность:            {report['success_rate']:.1f}%")
        print("-" * 60)
        print(f"Вопросы с ответом (avg score):   {report['should_answer_avg_score']:.2f}")
        print(f"Вопросы без ответа (avg score):  {report['should_not_answer_avg_score']:.2f}")
        print(f"Среднее время ответа:  {report['avg_response_time_ms']:.0f} мс")

    def find_coverage_gaps(self, logs_file: str = "evaluation_logs.jsonl") -> Dict[str, List[str]]:
        gaps = {"missing_sources": [], "low_relevance": [], "hallucinations": []}
        found_sources = set()

        try:
            with open(logs_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        result = json.loads(line)
                        for s in result.get('sources', []):
                            found_sources.add(s)
                        if result.get('should_have_answer', False) and result.get('score', 0) < 1:
                            gaps["low_relevance"].append(result['query'])
                        if not result.get('should_have_answer', False) and result.get('found', False):
                            if "не знаю" not in result.get('actual_answer', '').lower():
                                gaps["hallucinations"].append(result['query'])
        except FileNotFoundError:
            print(f"Файл {logs_file} не найден. Сначала запусти: python evaluate.py --run")
            return gaps

        try:
            with open('metadata.pkl', 'rb') as f:
                data = pickle.load(f)
                all_sources = set(m['source'] for m in data['metadatas'])
            gaps["missing_sources"] = list(all_sources - found_sources)
        except Exception as e:
            print(f"Ошибка: {e}")

        return gaps


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--gaps', action='store_true')
    parser.add_argument('--golden', type=str, default='golden_questions.json')
    args = parser.parse_args()

    if args.gaps:
        evaluator = QualityEvaluator(bot=None, golden_file=args.golden)
        gaps = evaluator.find_coverage_gaps()
        print("\nПРОБЕЛЫ В ПОКРЫТИИ")
        print(f"  Не найдены источники ({len(gaps['missing_sources'])}):")
        for s in gaps['missing_sources']:
            print(f"    - {s}")
        print(f"\n  Низкая релевантность: {gaps['low_relevance']}")
        print(f"  Галлюцинации: {gaps['hallucinations']}")
        return

    print("Загрузка RAG бота...")
    bot = QuantumForgeRAGBot()
    evaluator = QualityEvaluator(bot, golden_file=args.golden)

    if args.run:
        evaluator.run_all()
        evaluator.save_logs()
        evaluator.save_csv()
        evaluator.print_summary()
        gaps = evaluator.find_coverage_gaps()
        if gaps["missing_sources"]:
            print(f"\nФайлы, которые не были найдены ({len(gaps['missing_sources'])}):")
            for s in gaps['missing_sources']:
                print(f"    - {s}")
    else:
        print("python evaluate.py --run")


if __name__ == "__main__":
    main()