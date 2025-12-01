import sqlite3
from typing import List, Dict
import tiktoken

llama_tokenizer = tiktoken.get_encoding("cl100k_base")

class Memory_Manager:
    def __init__(self, db_path: str, max_token: int = 6000):
        self.db_path = db_path
        self.max_token = max_token
        self.init_database()

    def init_database(self):
        # Создаем таблицы если их нет
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        conn.close()

    def add_message(self, user_id: int, role: str, content: str):
        # Добавляем сообщение в базу данных
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO messages (user_id, role, content)
            VALUES (?, ?, ?)
        """,
            (user_id, role, content),
        )
        conn.commit()
        conn.close()

    def count_token(self, text: str) -> int:
        # подсчет токенов
        return len(llama_tokenizer.encode(text))

    def get_conversation_context(self, user_id: int, system_prompt: str) -> List[Dict]:
        # Получаем историю диалога с учетом лимита токенов
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE user_id = ? ORDER BY timestamp ASC",
            (user_id,),
        )
        all_messages = [
            {"role": row[0], "content": row[1]} for row in cursor.fetchall()
        ]
        conn.close()
        # Токены системного промпта (должны всегда оставаться)
        system_tokens = self.count_token(system_prompt)
        available_tokens = self.max_token - system_tokens

        if available_tokens <= 0:
            return []
        # Начинаем с конца (самых новых сообщений) и добавляем пока хватает токенов
        result_messages = []
        current_tokens = 0
        # Идем с конца к началу (от новых к старым)
        for msg in reversed(all_messages):
            msg_tokens = self.count_token(msg["content"])
            if current_tokens + msg_tokens <= available_tokens:
                result_messages.append(msg)
                current_tokens += msg_tokens
            else:
                break  # Прекращаем когда достигли лимита

        # Возвращаем в правильном порядке (от старых к новым)
        return list(reversed(result_messages))

    def clear_user_memory(self, user_id: int):
        #Очищаем всю историю пользователя
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM messages WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()