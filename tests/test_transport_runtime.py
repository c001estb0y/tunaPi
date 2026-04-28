from pathlib import Path

from tunapi.agent_runtime import AgentRuntimeConfig
from tunapi.config import ProjectConfig, ProjectsConfig
from tunapi.context import RunContext
from tunapi.router import AutoRouter, RunnerEntry
from tunapi.runners.mock import Return, ScriptRunner
from tunapi.transport_runtime import TransportRuntime


def _make_runtime(*, project_default_engine: str | None = None) -> TransportRuntime:
    codex = ScriptRunner([Return(answer="ok")], engine="codex")
    pi = ScriptRunner([Return(answer="ok")], engine="pi")
    router = AutoRouter(
        entries=[
            RunnerEntry(engine=codex.engine, runner=codex),
            RunnerEntry(engine=pi.engine, runner=pi),
        ],
        default_engine=codex.engine,
    )
    project = ProjectConfig(
        alias="proj",
        path=Path("."),
        worktrees_dir=Path(".worktrees"),
        default_engine=project_default_engine,
    )
    projects = ProjectsConfig(projects={"proj": project}, default_project=None)
    return TransportRuntime(router=router, projects=projects)


def test_resolve_engine_uses_project_default() -> None:
    runtime = _make_runtime(project_default_engine="pi")
    engine = runtime.resolve_engine(
        engine_override=None,
        context=RunContext(project="proj"),
    )
    assert engine == "pi"


def test_resolve_engine_prefers_override() -> None:
    runtime = _make_runtime(project_default_engine="pi")
    engine = runtime.resolve_engine(
        engine_override="codex",
        context=RunContext(project="proj"),
    )
    assert engine == "codex"


def test_resolve_message_defaults_to_chat_project() -> None:
    codex = ScriptRunner([Return(answer="ok")], engine="codex")
    router = AutoRouter(
        entries=[RunnerEntry(engine=codex.engine, runner=codex)],
        default_engine=codex.engine,
    )
    project = ProjectConfig(
        alias="proj",
        path=Path("."),
        worktrees_dir=Path(".worktrees"),
        chat_id=-42,
    )
    projects = ProjectsConfig(
        projects={"proj": project},
        default_project=None,
        chat_map={-42: "proj"},
    )
    runtime = TransportRuntime(router=router, projects=projects)

    resolved = runtime.resolve_message(
        text="hello",
        reply_text=None,
        chat_id=-42,
    )

    assert resolved.context == RunContext(project="proj", branch=None)


def test_resolve_message_uses_ambient_context() -> None:
    runtime = _make_runtime()
    ambient = RunContext(project="proj", branch="feat/ambient")

    resolved = runtime.resolve_message(
        text="hello",
        reply_text=None,
        ambient_context=ambient,
    )

    assert resolved.context == ambient
    assert resolved.context_source == "ambient"


def test_resolve_message_reply_ctx_overrides_ambient() -> None:
    runtime = _make_runtime()
    ambient = RunContext(project="proj", branch="feat/ambient")

    resolved = runtime.resolve_message(
        text="hello",
        reply_text="`ctx: proj @reply`",
        ambient_context=ambient,
    )

    assert resolved.context == RunContext(project="proj", branch="reply")
    assert resolved.context_source == "reply_ctx"


def test_resolve_message_directives_override_ambient() -> None:
    runtime = _make_runtime()
    ambient = RunContext(project="proj", branch="feat/ambient")

    resolved = runtime.resolve_message(
        text="/proj @main do it",
        reply_text=None,
        ambient_context=ambient,
    )

    assert resolved.context == RunContext(project="proj", branch="main")
    assert resolved.context_source == "directives"


def test_resolve_message_branch_directive_merges_with_ambient_project() -> None:
    runtime = _make_runtime()
    ambient = RunContext(project="proj", branch="feat/ambient")

    resolved = runtime.resolve_message(
        text="@hotfix do it",
        reply_text=None,
        ambient_context=ambient,
    )

    assert resolved.context == RunContext(project="proj", branch="hotfix")
    assert resolved.context_source == "directives"


def test_resolve_message_project_directive_clears_ambient_branch() -> None:
    codex = ScriptRunner([Return(answer="ok")], engine="codex")
    router = AutoRouter(
        entries=[RunnerEntry(engine=codex.engine, runner=codex)],
        default_engine=codex.engine,
    )
    projects = ProjectsConfig(
        projects={
            "proj": ProjectConfig(
                alias="proj",
                path=Path("."),
                worktrees_dir=Path(".worktrees"),
            ),
            "other": ProjectConfig(
                alias="other",
                path=Path("."),
                worktrees_dir=Path(".worktrees"),
            ),
        },
        default_project=None,
    )
    runtime = TransportRuntime(router=router, projects=projects)
    ambient = RunContext(project="proj", branch="feat/ambient")

    resolved = runtime.resolve_message(
        text="/other do it",
        reply_text=None,
        ambient_context=ambient,
    )

    assert resolved.context == RunContext(project="other", branch=None)
    assert resolved.context_source == "directives"


def test_resolve_run_environment_from_runtime(tmp_path: Path) -> None:
    runtime = _make_runtime()
    runtime.set_agent_runtime(
        AgentRuntimeConfig(root=tmp_path / "agent-runtime", enabled=True)
    )

    env = runtime.resolve_run_environment(
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )

    assert env is not None
    assert env.runtime_home == tmp_path / "agent-runtime" / "runtime-homes" / "kaixing"


def test_resolve_run_environment_returns_none_when_disabled(tmp_path: Path) -> None:
    runtime = _make_runtime()
    runtime.set_agent_runtime(
        AgentRuntimeConfig(root=tmp_path / "agent-runtime", enabled=False)
    )

    assert (
        runtime.resolve_run_environment(
            agent_id="kaixing",
            workspace_dir=tmp_path / "workspace",
        )
        is None
    )


def test_transport_runtime_resolves_bound_channel_workspace(tmp_path: Path) -> None:
    runtime = _make_runtime()
    runtime.set_agent_runtime(
        AgentRuntimeConfig(root=tmp_path / "agent-runtime", enabled=True)
    )
    workspace = tmp_path / "agent-runtime" / "workspaces" / "agent-mem"
    workspace.mkdir(parents=True)
    runtime.add_channel_workspace(
        channel_id="channel-1",
        name="agent-mem",
        path=workspace,
        repo="https://github.com/example/agent-mem",
    )
    runtime.bind_channel_workspace_agent(
        channel_id="channel-1",
        agent_id="codeview",
        workspace_name="agent-mem",
    )

    resolved = runtime.resolve_channel_workspace(
        channel_id="channel-1",
        agent_id="codeview",
        explicit_workspace=None,
        fallback_workspace=None,
    )

    assert resolved is not None
    assert resolved.workspace.name == "agent-mem"
    assert resolved.workspace.path == workspace
    assert resolved.binding_source == "agent-default"


def test_transport_runtime_lists_channel_workspaces(tmp_path: Path) -> None:
    runtime = _make_runtime()
    runtime.set_agent_runtime(
        AgentRuntimeConfig(root=tmp_path / "agent-runtime", enabled=True)
    )
    workspace = tmp_path / "agent-runtime" / "workspaces" / "agent-mem"
    workspace.mkdir(parents=True)
    runtime.add_channel_workspace(
        channel_id="channel-1",
        name="agent-mem",
        path=workspace,
        repo="https://github.com/example/agent-mem",
    )
    runtime.bind_channel_workspace_agent(
        channel_id="channel-1",
        agent_id="codeview",
        workspace_name="agent-mem",
    )
    runtime.set_channel_default_workspace(
        channel_id="channel-1",
        workspace_name="agent-mem",
    )

    binding = runtime.list_channel_workspaces("channel-1")

    assert binding is not None
    assert binding.workspaces["agent-mem"].repo == "https://github.com/example/agent-mem"
    assert binding.agents["codeview"].default_workspace == "agent-mem"
    assert binding.default_workspace == "agent-mem"
