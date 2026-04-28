from pathlib import Path

import pytest

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
