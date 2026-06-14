import json
import os
import pickle
import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('index_update.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class KnowledgeBaseUpdater:
    """Автоматическое обновление базы знаний и FAISS индекса"""

    def __init__(
            self,
            knowledge_base_dir: str = "knowledge_base",
            index_path: str = "faiss.index",
            metadata_path: str = "metadata.pkl",
            chunk_size: int = 1000,
            chunk_overlap: int = 200
    ):
        self.kb_dir = Path(knowledge_base_dir)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Хэши файлов для отслеживания изменений
        self.file_hashes: Dict[str, str] = {}
        self._load_hashes()

        # Модель эмбеддингов
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

    def _load_hashes(self):
        """Загрузка сохранённых хэшей файлов"""
        hash_path = self.kb_dir / ".file_hashes.json"
        if hash_path.exists():
            with open(hash_path, 'r') as f:
                self.file_hashes = json.load(f)

    def _save_hashes(self):
        """Сохранение хэшей файлов"""
        hash_path = self.kb_dir / ".file_hashes.json"
        with open(hash_path, 'w') as f:
            json.dump(self.file_hashes, f, indent=2)

    def _get_file_hash(self, filepath: Path) -> str:
        """Вычисление MD5 хэша файла"""
        with open(filepath, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _get_text_files(self) -> List[Path]:
        """Получение списка всех .txt файлов в папке"""
        return [f for f in self.kb_dir.glob("*.txt") if f.name != ".file_hashes.json"]

    def get_changes(self) -> Dict[str, List[Path]]:
        """
        Определение изменений в базе знаний.
        Возвращает: {'new': [...], 'modified': [...], 'deleted': [...]}
        """
        current_files = self._get_text_files()
        current_hashes = {}

        for filepath in current_files:
            current_hashes[str(filepath.name)] = self._get_file_hash(filepath)

        changes = {'new': [], 'modified': [], 'deleted': []}

        # Новые и изменённые файлы
        for filename, hash_val in current_hashes.items():
            if filename not in self.file_hashes:
                changes['new'].append(self.kb_dir / filename)
                logger.info(f"Новый файл: {filename}")
            elif self.file_hashes[filename] != hash_val:
                changes['modified'].append(self.kb_dir / filename)
                logger.info(f"Изменён файл: {filename}")

        # Удалённые файлы
        for filename in self.file_hashes:
            if filename not in current_hashes:
                changes['deleted'].append(filename)
                logger.info(f"Удалён файл: {filename}")

        return changes

    def _chunk_document(self, content: str, source: str) -> List[Dict[str, Any]]:
        """Разбивка документа на чанки"""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(content)
        return [
            {
                'content': chunk,
                'source': source,
                'chunk_id': i,
                'title': source.replace('.txt', '')
            }
            for i, chunk in enumerate(chunks)
        ]

    def rebuild_index(self):
        """Полная перестройка индекса (при больших изменениях)"""
        logger.info("Полная перестройка индекса...")

        all_chunks = []
        all_metadatas = []

        for filepath in self._get_text_files():
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks_data = self._chunk_document(content, filepath.name)
            for chunk in chunks_data:
                all_chunks.append(chunk['content'])
                all_metadatas.append({
                    'source': chunk['source'],
                    'chunk_id': chunk['chunk_id'],
                    'title': chunk['title']
                })

        if not all_chunks:
            logger.error("❌ Нет чанков для индексации!")
            return False

        # Генерация эмбеддингов
        logger.info(f"Генерация эмбеддингов для {len(all_chunks)} чанков...")
        embeddings = self.embeddings.embed_documents(all_chunks)
        embedding_size = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')

        # Создание FAISS индекса
        index = faiss.IndexFlatL2(embedding_size)
        index.add(embeddings_np)

        # Сохранение
        faiss.write_index(index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump({'chunks': all_chunks, 'metadatas': all_metadatas}, f)

        # Обновление хэшей
        for filepath in self._get_text_files():
            self.file_hashes[filepath.name] = self._get_file_hash(filepath)
        self._save_hashes()

        logger.info(f"Индекс обновлён: {len(all_chunks)} чанков, размерность {embedding_size}")
        return True

    def incremental_update(self):
        """Инкрементальное обновление (только изменённые файлы)"""
        changes = self.get_changes()

        if not any(changes.values()):
            logger.info("Нет изменений в базе знаний")
            return True

        # При удалении или большом количестве изменений — полная перестройка
        if len(changes['deleted']) > 0 or len(changes['new']) + len(changes['modified']) > 5:
            logger.info("Обнаружены удаления или много изменений — выполняем полную перестройку")
            return self.rebuild_index()

        if not changes['new'] and not changes['modified']:
            return True

        # Загрузка существующего индекса
        if not os.path.exists(self.index_path):
            logger.info("Индекс не найден — полная перестройка")
            return self.rebuild_index()

        index = faiss.read_index(self.index_path)
        with open(self.metadata_path, 'rb') as f:
            data = pickle.load(f)
            chunks = data['chunks']
            metadatas = data['metadatas']

        # Добавление новых и изменённых файлов
        new_chunks = []
        new_metadatas = []

        for filepath in changes['new'] + changes['modified']:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            chunks_data = self._chunk_document(content, filepath.name)
            for chunk in chunks_data:
                new_chunks.append(chunk['content'])
                new_metadatas.append({
                    'source': chunk['source'],
                    'chunk_id': chunk['chunk_id'],
                    'title': chunk['title']
                })

            # Обновление хэша
            self.file_hashes[filepath.name] = self._get_file_hash(filepath)

        # Генерация эмбеддингов для новых чанков
        if new_chunks:
            logger.info(f"Добавление {len(new_chunks)} новых чанков...")
            embeddings = self.embeddings.embed_documents(new_chunks)
            embeddings_np = np.array(embeddings).astype('float32')

            # Добавление в существующий индекс
            index.add(embeddings_np)
            chunks.extend(new_chunks)
            metadatas.extend(new_metadatas)

            # Сохранение
            faiss.write_index(index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump({'chunks': chunks, 'metadatas': metadatas}, f)

            logger.info(f"Добавлено {len(new_chunks)} чанков")

        self._save_hashes()
        return True

    def run(self, force_rebuild: bool = False):
        """Запуск обновления"""
        start_time = datetime.now()
        logger.info(f"Запуск обновления индекса в {start_time}")

        try:
            if force_rebuild:
                success = self.rebuild_index()
            else:
                success = self.incremental_update()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            if success:
                logger.info(f"Обновление завершено за {duration:.2f} секунд")
            else:
                logger.error(f"Обновление завершилось с ошибкой за {duration:.2f} секунд")

            return success

        except Exception as e:
            logger.error(f"Критическая ошибка: {e}")
            return False


def setup_cron():
    """Инструкция по настройке cron для ежедневного обновления"""

    project_path = Path.cwd()
    python_path = project_path / ".venv" / "bin" / "python"
    cron_line = f"0 6 * * * cd {Path.cwd()} && {python_path} update_index.py >> /var/log/kb_update.log 2>&1"

    print("\nДля настройки ежедневного обновления выполните:")
    print(f"   crontab -e")
    print(f"   Добавьте строку:")
    print(f"   {cron_line}")
    print("\n   Или одной командой:")
    print(f'   (crontab -l 2>/dev/null; echo "{cron_line}") | crontab -')
    print("\nЛоги будут в /tmp/kb_update.log")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Обновление базы знаний RAG бота')
    parser.add_argument('--rebuild', action='store_true', help='Полная перестройка индекса')
    parser.add_argument('--check-only', action='store_true', help='Только проверить изменения')
    parser.add_argument('--setup-cron', action='store_true', help='Показать инструкцию по настройке cron')

    args = parser.parse_args()

    if args.setup_cron:
        setup_cron()
        return

    updater = KnowledgeBaseUpdater()

    if args.check_only:
        changes = updater.get_changes()
        print("\nИзменения в базе знаний:")
        print(f"  Новые файлы: {len(changes['new'])}")
        print(f"  Изменённые: {len(changes['modified'])}")
        print(f"  Удалённые: {len(changes['deleted'])}")

        for f in changes['new']:
            print(f"    + {f.name}")
        for f in changes['modified']:
            print(f"    * {f.name}")
        for f in changes['deleted']:
            print(f"    - {f}")
        return

    success = updater.run(force_rebuild=args.rebuild)

    if not success:
        exit(1)


if __name__ == "__main__":
    main()