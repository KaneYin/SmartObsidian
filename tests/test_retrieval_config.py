import pytest

from weft.config import ResolvedConfig, config_path_for, save_config
from weft.retrieval_config import (
    RetrievalOverrides,
    build_retrieval_config,
    resolve_retrieval_config,
)


def test_mode_presets_express_user_intent():
    fast = build_retrieval_config("fast")
    balanced = build_retrieval_config("balanced")
    best = build_retrieval_config("best")

    assert (fast.k, fast.hybrid, fast.graph, fast.rerank, fast.memory_query) == (
        5, False, True, False, False
    )
    assert (balanced.k, balanced.hybrid, balanced.graph,
            balanced.rerank, balanced.memory_query) == (8, True, True, False, True)
    assert (best.k, best.hybrid, best.graph, best.rerank,
            best.rerank_pool, best.memory_query) == (8, True, True, True, 30, True)


def test_explicit_options_override_mode():
    config = build_retrieval_config(
        "best",
        RetrievalOverrides(k=12, hybrid=False, rerank=False, memory_query=False),
    )
    assert config.k == 12
    assert config.hybrid is False
    assert config.rerank is False
    assert config.memory_query is False


def test_mode_precedence_cli_over_env_over_file(tmp_path):
    store = tmp_path / ".weft" / "index"
    save_config(config_path_for(store), ResolvedConfig(mode="fast"))

    from_file = resolve_retrieval_config(store, env={})
    from_env = resolve_retrieval_config(store, env={"WEFT_MODE": "best"})
    from_cli = resolve_retrieval_config(
        store, mode="balanced", env={"WEFT_MODE": "best"}
    )

    assert from_file.mode == "fast"
    assert from_env.mode == "best"
    assert from_cli.mode == "balanced"


def test_invalid_mode_and_bounds_are_rejected():
    with pytest.raises(ValueError, match="mode must be"):
        build_retrieval_config("turbo")
    with pytest.raises(ValueError, match="k must be"):
        build_retrieval_config("fast", RetrievalOverrides(k=0))
    with pytest.raises(ValueError, match="rerank pool"):
        build_retrieval_config("best", RetrievalOverrides(rerank_pool=0))
