"""A small dependency-free interactive adapter over Weft application services.

This is intentionally a menu-style TUI.  It can grow into a persistent Textual
application later without changing the service boundary introduced here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from weft import service
from weft.chat import run_repl
from weft.llm import LLMRequestError
from weft.providers import ProviderUnavailable
from weft.security import WeftSecurityError, terminal_safe

Read = Callable[[str], str]
Emit = Callable[[str], None]


@dataclass
class TUIState:
    store_path: Path
    mode: str


def _read_value(read: Read, prompt: str) -> str:
    return read(prompt).strip()


def _show_settings(state: TUIState, emit: Emit) -> dict:
    config = service.service_config(state.store_path)
    emit("\nSettings")
    emit(f"  Mode: {state.mode}")
    emit(f"  Provider: {config['provider']}")
    emit(f"  Model: {config['model']}")
    emit(f"  Store: {state.store_path}")
    return config


def _settings(state: TUIState, read: Read, emit: Emit) -> None:
    while True:
        _show_settings(state, emit)
        emit("\n1. Mode\n2. Provider\n3. Model\n4. Store path\n5. Back")
        choice = _read_value(read, "Select: ").lower()
        if choice in {"5", "back", "b", ""}:
            return
        try:
            if choice in {"1", "mode"}:
                value = _read_value(read, "Mode (fast/balanced/best): ").lower()
                service.service_config_set(state.store_path, "mode", value)
                state.mode = value
            elif choice in {"2", "provider"}:
                value = _read_value(read, "Provider (ollama/anthropic/openai/fake): ").lower()
                service.service_config_set(state.store_path, "provider", value)
            elif choice in {"3", "model"}:
                value = _read_value(read, "Model tag: ")
                if not value:
                    raise ValueError("model must not be empty")
                service.service_config_set(state.store_path, "model", value)
            elif choice in {"4", "store", "store path"}:
                value = _read_value(read, "Store path: ")
                if not value:
                    raise ValueError("store path must not be empty")
                state.store_path = Path(value).expanduser()
                state.mode = service.service_config(state.store_path)["mode"]
            else:
                emit("Unknown selection.")
        except (ValueError, WeftSecurityError) as exc:
            emit(f"Error: {terminal_safe(exc)}")


def _ask(state: TUIState, read: Read, emit: Emit) -> None:
    question = _read_value(read, "Question: ")
    if not question:
        emit("Question cannot be empty.")
        return
    result = service.service_ask(
        state.store_path,
        question,
        mode=state.mode,
    )
    emit("\nAnswer\n")
    emit(terminal_safe(result["answer"]))
    if result["sources"]:
        emit("\nSources")
        for source in result["sources"]:
            emit(f"  - {terminal_safe(source)}")


def _chat(state: TUIState, read: Read, emit: Emit) -> None:
    session = service.service_chat_session(state.store_path, mode=state.mode)
    run_repl(session, read=read, emit=emit)


def _index(state: TUIState, read: Read, emit: Emit) -> None:
    vault = _read_value(read, "Vault path: ")
    if not vault:
        emit("Vault path cannot be empty.")
        return
    result = service.service_index(
        Path(vault).expanduser(),
        state.store_path,
    )
    emit(
        f"Indexed {result['chunks']} chunks and {result['edges']} link edges "
        f"into {terminal_safe(state.store_path)}"
    )


def _suggest(state: TUIState, read: Read, emit: Emit) -> None:
    vault = _read_value(read, "Vault path: ")
    if not vault:
        emit("Vault path cannot be empty.")
        return
    result = service.service_suggest(
        Path(vault).expanduser(),
        state.store_path,
    )
    if result["count"]:
        emit(f"{result['count']} suggestions -> {terminal_safe(result['inbox'])}")
    else:
        emit("0 suggestions; existing inbox left unchanged")


def run_tui(
    *,
    store_path: str | Path = ".weft/index",
    read: Read | None = None,
    emit: Emit | None = None,
) -> int:
    """Run the minimal interactive Weft menu without subprocess boundaries."""
    read = read or input
    emit = emit or print
    path = Path(store_path).expanduser()
    state = TUIState(path, service.service_config(path)["mode"])
    emit("Weft")
    while True:
        emit(
            f"\nMode: {state.mode}\n"
            "1. Ask vault\n"
            "2. Chat\n"
            "3. Index vault\n"
            "4. Discover links\n"
            "5. Settings\n"
            "6. Exit"
        )
        try:
            choice = _read_value(read, "Select: ").lower()
        except (EOFError, KeyboardInterrupt):
            emit("\nGoodbye.")
            return 0
        if choice in {"6", "exit", "quit", "q"}:
            emit("Goodbye.")
            return 0
        action = {
            "1": _ask,
            "ask": _ask,
            "2": _chat,
            "chat": _chat,
            "3": _index,
            "index": _index,
            "4": _suggest,
            "suggest": _suggest,
            "discover": _suggest,
        }.get(choice)
        try:
            if choice in {"5", "settings"}:
                _settings(state, read, emit)
            elif action is not None:
                action(state, read, emit)
            else:
                emit("Unknown selection.")
        except (EOFError, KeyboardInterrupt):
            emit("\nCancelled.")
        except (
            service.IndexNotFoundError,
            service.IndexVaultMismatchError,
            service.InboxExistsError,
            LLMRequestError,
            ProviderUnavailable,
            WeftSecurityError,
            OSError,
            ValueError,
        ) as exc:
            emit(f"Error: {terminal_safe(exc)}")


__all__ = ["TUIState", "run_tui"]
