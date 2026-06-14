import re
import json
from typing import List, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass

try:
    from transformers import pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


@dataclass
class SecurityEvent:
    """Событие безопасности для аудита"""
    timestamp: datetime
    event_type: str  # 'query_blocked', 'injection_detected', 'rate_limit'
    user_id: str
    query: str
    reason: str
    score: float = 0.0


class RateLimiter:
    """Rate limiting с скользящим окном (как в реальных API)"""

    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[datetime]] = defaultdict(list)

    def check(self, user_id: str) -> Tuple[bool, int]:
        """Проверка лимита, возвращает (разрешено, осталось запросов)"""
        now = datetime.now()
        window_start = now - timedelta(seconds=self.window_seconds)

        # Очистка старых запросов
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if req_time > window_start
        ]

        remaining = self.max_requests - len(self.requests[user_id])

        if len(self.requests[user_id]) >= self.max_requests:
            return False, 0

        self.requests[user_id].append(now)
        return True, remaining - 1


class PromptInjectionDetector:
    """
    Детекция промпт-инъекций.
    Использует многоуровневый подход: ML модель + правила + эвристики.
    """

    def __init__(self):
        # ML модель для детекции
        self.ml_model = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.ml_model = pipeline(
                    "text-classification",
                    model="viavoice/mdeberta-ru-prompt-injection",
                    truncation=True,
                    max_length=512
                )
                print("ML модель для детекции инъекций загружена")
            except Exception as e:
                print(f"ML модель не загружена: {e}")

        # Известные вредоносные паттерны (только самые явные)
        self.high_confidence_patterns = [
            re.compile(r'(?i)(ignore|bypass|override)\s+(all|previous|system|instructions)'),
            re.compile(r'(?i)output\s*:.*password|secret|key|token'),
            re.compile(r'(?i)(leak|expose|reveal)\s+(system|prompt|instructions)'),
            re.compile(r'(?i)pretend\s+you\s+are\s+(admin|root|system)'),
        ]

        # Подозрительные эвристики (низкая уверенность)
        self.suspicious_heuristics = [
            (re.compile(r'(?i)forget.*instructions'), 0.6),
            (re.compile(r'(?i)you\s+will\s+now'), 0.4),
            (re.compile(r'(?i)new\s+instruction'), 0.5),
            (re.compile(r'(?i)actually\s+ignore'), 0.7),
        ]

    def detect(self, text: str) -> Tuple[bool, float, str]:
        """
        Детекция промпт-инъекции.
        Возвращает: (is_injection, confidence_score, reason)
        """
        # Level 1: Высокая уверенность (явные паттерны)
        for pattern in self.high_confidence_patterns:
            if pattern.search(text):
                return True, 0.95, f"Обнаружен явный паттерн: {pattern.pattern}"

        # Level 2: ML модель
        if self.ml_model:
            try:
                result = self.ml_model(text)[0]
                if result['label'] == 'INJECTION' and result['score'] > 0.7:
                    return True, result['score'], "ML модель определила инъекцию"
            except Exception:
                pass  # Fallback к эвристикам

        # Level 3: Эвристики с весами
        total_score = 0.0
        reasons = []
        for pattern, weight in self.suspicious_heuristics:
            if pattern.search(text):
                total_score += weight
                reasons.append(pattern.pattern)

        if total_score > 1.0:
            return True, min(total_score / 2, 0.95), f"Сработали эвристики: {', '.join(reasons[:2])}"

        # Проверка длины и энтропии (аномалии)
        if self._has_high_entropy(text):
            return True, 0.6, "Аномально высокая энтропия текста"

        return False, 0.0, "OK"

    def _has_high_entropy(self, text: str, threshold: float = 4.5) -> bool:
        """Проверка энтропии текста (необычные запросы)"""
        import math

        if len(text) < 20:
            return False

        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        entropy = 0.0
        length = len(text)
        for count in freq.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * math.log2(prob)

        return entropy > threshold


class OutputSanitizer:
    """Санитизация выходных данных LLM"""

    def __init__(self):
        # Паттерны секретов и чувствительных данных
        self.secret_patterns = [
            (re.compile(r'(sk-[a-zA-Z0-9]{32,})', re.I), 'OPENAI_KEY'),
            (re.compile(r'(AQVN[a-zA-Z0-9]{30,})', re.I), 'YANDEX_API_KEY'),
            (re.compile(r'(Bearer\s+[a-zA-Z0-9._-]{20,})', re.I), 'BEARER_TOKEN'),
            (re.compile(r'(?i)(password|passwd|пароль)[\s:]*\S+'), 'PASSWORD'),
            (re.compile(r'(?i)(api[_-]?key|токен)[\s:]*\S+'), 'API_KEY'),
        ]

        # Системные промпты, которые нельзя раскрывать
        self.system_prompt_indicators = [
            r'(?i)system\s+prompt',
            r'(?i)you are a helpful assistant',
            r'(?i)правила.*для.*ассистента',
            r'(?i)твоя.*задача',
        ]

    def sanitize(self, text: str) -> Tuple[str, List[str]]:
        """
        Санитизация ответа.
        Возвращает: (очищенный_текст, список_сработавших_правил)
        """
        original = text
        triggered_rules = []

        # Удаление секретов
        for pattern, secret_type in self.secret_patterns:
            if pattern.search(text):
                text = pattern.sub(f'[REDACTED_{secret_type}]', text)
                triggered_rules.append(f'redacted_{secret_type}')

        # Проверка на утечку системного промпта
        for indicator in self.system_prompt_indicators:
            if re.search(indicator, text, re.I):
                triggered_rules.append('system_prompt_leak_attempt')
                text = "Ответ содержал системную информацию и был отфильтрован."
                break

        # Защита от повторяющихся паттернов (атаки loop)
        if self._has_repetition_attack(text):
            triggered_rules.append('repetition_attack')
            text = text[:500] + "... [ответ сокращён из-за повторяющегося паттерна]"

        return text, triggered_rules

    def _has_repetition_attack(self, text: str, max_repetitions: int = 10) -> bool:
        """Детекция loop атак (повторяющийся текст)"""
        words = text.split()
        if len(words) < 50:
            return False

        # Поиск повторяющихся последовательностей
        for length in [3, 4, 5]:
            patterns = {}
            for i in range(len(words) - length):
                pattern = ' '.join(words[i:i + length])
                patterns[pattern] = patterns.get(pattern, 0) + 1
                if patterns[pattern] > max_repetitions:
                    return True
        return False


class SecurityAuditor:
    """Аудит всех событий безопасности (для compliance)"""

    def __init__(self, log_path: str = "security_audit.log"):
        self.log_path = log_path
        self.events: List[SecurityEvent] = []

    def log(self, event_type: str, user_id: str, query: str, reason: str, score: float = 0.0):
        """Логирование события безопасности"""
        event = SecurityEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            user_id=user_id,
            query=query[:200],  # Ограничение длины
            reason=reason,
            score=score
        )
        self.events.append(event)
        self._write_to_file(event)

        # Консольное уведомление для админа
        print(f"[SECURITY] {event_type}: user={user_id}, reason={reason}")

    def _write_to_file(self, event: SecurityEvent):
        """Запись в JSONL файл для аналитики"""
        try:
            with open(self.log_path, 'a') as f:
                f.write(json.dumps({
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type,
                    'user_id': event.user_id,
                    'query': event.query,
                    'reason': event.reason,
                    'score': event.score
                }) + '\n')
        except Exception:
            pass  # Fail silently для логирования

    def get_stats(self) -> dict:
        """Получение статистики по событиям"""
        stats = defaultdict(int)
        for event in self.events:
            stats[event.event_type] += 1
        return dict(stats)


class ProductionSecurityFilter:
    """
    Главный класс безопасности для production RAG системы.
    Объединяет все уровни защиты.
    """

    def __init__(self):
        self.rate_limiter = RateLimiter(max_requests=60, window_seconds=60)
        self.injection_detector = PromptInjectionDetector()
        self.output_sanitizer = OutputSanitizer()
        self.auditor = SecurityAuditor()

        # Белый список разрешённых тем (опционально)
        self.allowed_topics = None  # None = всё разрешено, кроме запрещённого

        # Конфигурация
        self.max_query_length = 500
        self.min_query_length = 1

    def validate_query(self, query: str, user_id: str = "anonymous") -> Tuple[bool, str, float]:
        """
        Валидация входящего запроса.
        Возвращает: (разрешено, причина, confidence_score)
        """
        # Базовые проверки
        if not query or len(query) < self.min_query_length:
            return False, "Empty query", 0.0

        if len(query) > self.max_query_length:
            self.auditor.log('query_blocked', user_id, query, "Query too long")
            return False, f"Query too long (max {self.max_query_length} chars)", 0.0

        # Rate limiting
        allowed, remaining = self.rate_limiter.check(user_id)
        if not allowed:
            self.auditor.log('rate_limit', user_id, query, "Rate limit exceeded")
            return False, "Rate limit exceeded. Please try again later.", 0.0

        # Детекция инъекций
        is_injection, confidence, reason = self.injection_detector.detect(query)
        if is_injection:
            self.auditor.log('injection_detected', user_id, query, reason, confidence)
            return False, f"Security policy violation: {reason}", confidence

        return True, "OK", 1.0

    def sanitize_response(self, response: str, query: str, user_id: str = "anonymous") -> str:
        """Санитизация ответа перед отправкой пользователю"""
        sanitized, rules = self.output_sanitizer.sanitize(response)

        if rules:
            self.auditor.log('response_sanitized', user_id, query, f"Rules triggered: {rules}")

        return sanitized

    def get_report(self) -> dict:
        """Получение отчёта о безопасности для аудита (SOC2, ISO27001)"""
        return {
            'stats': self.auditor.get_stats(),
            'config': {
                'max_query_length': self.max_query_length,
                'rate_limit_max': 60,
                'rate_limit_window': 60,
                'ml_model_available': TRANSFORMERS_AVAILABLE
            },
            'timestamp': datetime.now().isoformat()
        }


# ============================================================
# Интеграция в RAG бота
# ============================================================

class SecureRAGBot:
    """RAG бот с production-безопасностью"""

    def __init__(self, bot):
        self.bot = bot
        self.security = ProductionSecurityFilter()

    def ask(self, query: str, user_id: str = "anonymous") -> dict:
        """Безопасный метод ask с защитой"""

        # Шаг 1: Валидация запроса
        is_safe, reason, confidence = self.security.validate_query(query, user_id)
        if not is_safe:
            return {
                'answer': f"Запрос отклонён: {reason}",
                'sources': [],
                'blocked': True,
                'security_confidence': confidence
            }

        # Шаг 2: Нормальная обработка (RAG пайплайн)
        response = self.bot.ask(query)

        # Шаг 3: Санитизация ответа
        response['answer'] = self.security.sanitize_response(
            response['answer'],
            query,
            user_id
        )

        return response

    def get_security_report(self) -> dict:
        """Отчёт для аудита"""
        return self.security.get_report()
