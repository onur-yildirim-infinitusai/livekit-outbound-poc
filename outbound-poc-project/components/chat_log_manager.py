import asyncio
import aiofiles
from datetime import datetime

class ChatLogManager:
    def __init__(self):
        self.log_queue = asyncio.Queue()
        self.chat_log = []
        self.file_path = "transcriptions.log"

    def add_message(self, message: str, role: str):
        line = f"[{datetime.now()}] {role}:\n{message}\n\n"
        print(line)
        self.log_queue.put_nowait(line)
        self.chat_log.append(line)

    async def write_to_file(self):
        async with aiofiles.open(self.file_path, "w") as f:
             while True:
                msg = await self.log_queue.get()
                if msg is None:
                    break
                await f.write(msg)

    def print_chat_log(self):
        for line in self.chat_log:
            print(line)
