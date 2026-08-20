import os
import stat
from datetime import datetime

import pytest

from weft.memory import MemoryItem
from weft.memory_mirror import render_mirror, write_mirror
from weft.proposals import Proposal
from weft.security import UnsafeWriteError


def _item(t, text):
    return MemoryItem(id="mem_1", type=t, text=text)


def _prop(text):
    return Proposal(id="prop_1", type="fact", text=text, status="pending",
                    signature="1", source="heuristic")


def test_render_shows_active_and_pending():
    md = render_mirror([_item("preference", "Answer concisely")],
                       [_prop("Recurring interest: retrieval")], datetime(2026, 8, 19))
    assert "Answer concisely" in md
    assert "Recurring interest: retrieval" in md
    assert "weft memory accept" in md
    assert "prop_1" in md


def test_write_mirror_is_private(tmp_path):
    path = write_mirror(tmp_path, render_mirror([_item("fact", "x")], [], datetime.now()))
    assert path.name == "_memory.md"
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600


def test_write_mirror_refuses_symlink(tmp_path):
    target = tmp_path / "_memory.md"
    target.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(UnsafeWriteError):
        write_mirror(tmp_path, "content")
