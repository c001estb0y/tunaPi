from pathlib import Path

import pytest
import tunapi.channel_workspaces as channel_workspaces

from tunapi.channel_workspaces import (
    ChannelWorkspaceStore,
    WorkspaceResolutionError,
)


def test_workspace_store_adds_and_loads_binding(tmp_path: Path) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")

    store.add_workspace(
        channel_id="channel-1",
        name="agent-mem",
        path=tmp_path / "agent-runtime" / "workspaces" / "agent-mem",
        repo="https://github.com/example/agent-mem",
    )
    store.bind_agent("channel-1", agent_id="codeview", workspace_name="agent-mem")

    loaded = store.load("channel-1")

    assert loaded.channel_id == "channel-1"
    assert loaded.workspaces["agent-mem"].path == (
        tmp_path / "agent-runtime" / "workspaces" / "agent-mem"
    )
    assert loaded.workspaces["agent-mem"].repo == "https://github.com/example/agent-mem"
    assert loaded.agents["codeview"].default_workspace == "agent-mem"


def test_workspace_store_load_missing_file_returns_empty_binding(
    tmp_path: Path,
) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")

    loaded = store.load("channel-1")

    assert loaded.channel_id == "channel-1"
    assert loaded.workspaces == {}
    assert loaded.agents == {}


def test_workspace_resolution_prefers_explicit_then_agent_then_channel_default(
    tmp_path: Path,
) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")
    store.add_workspace(
        channel_id="channel-1",
        name="agent-mem",
        path=tmp_path / "agent-runtime" / "workspaces" / "agent-mem",
        repo=None,
    )
    store.add_workspace(
        channel_id="channel-1",
        name="review-repo",
        path=tmp_path / "agent-runtime" / "workspaces" / "review-repo",
        repo=None,
    )
    store.set_default("channel-1", "agent-mem")
    store.bind_agent("channel-1", agent_id="codeview", workspace_name="review-repo")

    explicit = store.resolve(
        channel_id="channel-1",
        agent_id="codeview",
        explicit_workspace="agent-mem",
        fallback_workspace=None,
    )
    agent_default = store.resolve(
        channel_id="channel-1",
        agent_id="codeview",
        explicit_workspace=None,
        fallback_workspace=None,
    )
    channel_default = store.resolve(
        channel_id="channel-1",
        agent_id="kaixing",
        explicit_workspace=None,
        fallback_workspace=None,
    )

    assert explicit.binding_source == "explicit"
    assert explicit.workspace.name == "agent-mem"
    assert agent_default.binding_source == "agent-default"
    assert agent_default.workspace.name == "review-repo"
    assert channel_default.binding_source == "channel-default"
    assert channel_default.workspace.name == "agent-mem"


def test_workspace_store_rejects_path_outside_runtime_root(tmp_path: Path) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")

    with pytest.raises(WorkspaceResolutionError, match="inside runtime root"):
        store.add_workspace(
            channel_id="channel-1",
            name="escape",
            path=tmp_path / "elsewhere",
            repo=None,
        )


@pytest.mark.parametrize("payload", [b"\xff\xfe", b"[channel\n"])
def test_workspace_store_wraps_invalid_bindings_file(
    tmp_path: Path,
    payload: bytes,
) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")
    bindings_path = store.bindings_path("channel-1")
    bindings_path.parent.mkdir(parents=True)
    bindings_path.write_bytes(payload)

    with pytest.raises(WorkspaceResolutionError, match="invalid workspace bindings"):
        store.load("channel-1")


def test_workspace_store_uses_fallback_workspace(tmp_path: Path) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")
    fallback = tmp_path / "agent-runtime" / "workspaces" / "fallback"

    resolved = store.resolve(
        channel_id="channel-1",
        agent_id="codeview",
        explicit_workspace=None,
        fallback_workspace=fallback,
    )

    assert resolved is not None
    assert resolved.binding_source == "project-binding"
    assert resolved.workspace.name == "fallback"
    assert resolved.workspace.path == fallback
    assert resolved.channel_context_dir == store.channel_dir("channel-1")


def test_workspace_store_rejects_unknown_explicit_workspace(tmp_path: Path) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")

    with pytest.raises(WorkspaceResolutionError, match="unknown workspace"):
        store.resolve(
            channel_id="channel-1",
            agent_id="codeview",
            explicit_workspace="missing",
            fallback_workspace=None,
        )


def test_workspace_store_rejects_concurrent_mutation_lock(tmp_path: Path) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")
    lock = store.channel_dir("channel-1") / ".bindings.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("other-holder\n", encoding="utf-8")

    with pytest.raises(WorkspaceResolutionError, match="workspace bindings are locked"):
        store.add_workspace(
            channel_id="channel-1",
            name="agent-mem",
            path=tmp_path / "agent-runtime" / "workspaces" / "agent-mem",
            repo=None,
        )


def test_workspace_store_recovers_stale_mutation_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")
    lock = store.channel_dir("channel-1") / ".bindings.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("999999999:0123456789abcdef0123456789abcdef\n", encoding="utf-8")
    monkeypatch.setattr(channel_workspaces, "_pid_exists", lambda _pid: False)

    store.add_workspace(
        channel_id="channel-1",
        name="agent-mem",
        path=tmp_path / "agent-runtime" / "workspaces" / "agent-mem",
        repo=None,
    )

    loaded = store.load("channel-1")
    assert loaded.workspaces["agent-mem"].path == (
        tmp_path / "agent-runtime" / "workspaces" / "agent-mem"
    )
    assert not lock.exists()


def test_workspace_store_rejects_unknown_lock_token_format(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = ChannelWorkspaceStore(tmp_path / "agent-runtime")
    lock = store.channel_dir("channel-1") / ".bindings.lock"
    lock.parent.mkdir(parents=True)
    lock.write_text("999999:not-a-uuid\n", encoding="utf-8")
    monkeypatch.setattr(channel_workspaces, "_pid_exists", lambda _pid: False)

    with pytest.raises(WorkspaceResolutionError, match="workspace bindings are locked"):
        store.add_workspace(
            channel_id="channel-1",
            name="agent-mem",
            path=tmp_path / "agent-runtime" / "workspaces" / "agent-mem",
            repo=None,
        )

    assert lock.read_text(encoding="utf-8") == "999999:not-a-uuid\n"
