import time
import asyncio
import subprocess
from typing import List, Callable, Awaitable, Union
from icecream import ic

async def run_command(cmd: List[str]) -> int | None:
    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        await process.communicate()
        return process.returncode
    except subprocess.CalledProcessError as e:
        ic(str(e.stderr))
    except Exception as e:
        ic(f"Unexpected error during command execution: {e}")

def check_phrases(text: str, phrases: List[str]) -> bool:
    text = text.lower()
    return any(phrase in text for phrase in phrases)

async def handle_query(
    text: str,
    phrases: List[str],
    handler: Union[Callable[[], str], Callable[[], Awaitable[str]]]
) -> str | Awaitable[str] | None:
    if check_phrases(text, phrases):
        if asyncio.iscoroutinefunction(handler):
            return await handler()
        return handler()
    return None

def get_time() -> str:
    return f"It's {time.strftime('%I:%M %p')}."

PACMAN_PHRASES = [
    "pacman update", "update pacman", "pacman upgrade", "upgrade pacman",
    "system update with pacman", "system upgrade with pacman",
]

YAY_PHRASES = [
    "yay update", "update yay", "yay upgrade", "upgrade yay",
    "system update with yay", "system upgrade with yay",
    "aur update", "update aur", "aur upgrade", "upgrade aur",
]

GENERIC_UPDATE_PHRASES = [
    "update", "upgrade", "update system", "upgrade system",
    "system update", "system upgrade", "system is outdated",
    "system needs update", "system needs upgrade",
    "system needs to be updated", "system needs to be upgraded",
    "system needs updating", "system needs upgrading",
]

async def update_system(text: str) -> tuple[int | None, str] | None:
    if any(phrase in text.lower() for phrase in PACMAN_PHRASES):
        return await run_command(["sudo", "pacman", "--noconfirm", "-Syu"]), "pacman"
    elif any(phrase in text.lower() for phrase in YAY_PHRASES):
        return await run_command(["yay", "--noconfirm", "-Syu"]), "yay"

TIME_PHRASES = [
    "what time", "what's the time", "whats the time",
    "tell me the time", "current time", "time is it",
    "time right now", "got the time", "have the time",
    "clock say", "what hour", "tell time",
]

async def check_time_query(text: str) -> str | Awaitable[str] | None:
    return await handle_query(text, TIME_PHRASES, get_time)

async def check_update_query(text: str) -> str | Awaitable[str] | None:
    for phrases in [PACMAN_PHRASES, YAY_PHRASES, GENERIC_UPDATE_PHRASES]:
        if check_phrases(text, phrases):
            result = await update_system(text) or "Command"
            return f"{result[1]} failed." if result[0] == 1 else f"{result[1]} successful."
