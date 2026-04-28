import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

import tunapi.runner as runner_module
from tunapi.agent_runtime import (
    AgentRuntimeConfig,
    AgentRuntimeError,
    RunManifest,
    activate_run_environment,
    git_commit,
    get_active_run_environment,
    prepare_runtime_home,
    resolve_run_environment,
    workspace_lock,
    write_manifest,
)
from tunapi.model import CompletedEvent, ResumeToken, TunapiEvent
from tunapi.runner import JsonlSubprocessRunner


def test_agent_runtime_config_exposes_runtime_dirs(tmp_path: Path) -> None:
    cfg = AgentRuntimeConfig(
        root=tmp_path / "agent-runtime",
        enabled=True,
        default_agent_id="kaixing",
    )

    assert cfg.enabled is True
    assert cfg.default_agent_id == "kaixing"
    assert cfg.agent_envs_dir == cfg.root / "agent-envs"
    assert cfg.runtime_homes_dir == cfg.root / "runtime-homes"
    assert cfg.workspaces_dir == cfg.root / "workspaces"
    assert cfg.locks_dir == cfg.root / "locks"
    assert cfg.runs_dir == cfg.root / "runs"
    assert cfg.logs_dir == cfg.root / "logs"


def test_resolve_run_environment_uses_agent_specific_home(tmp_path: Path) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")

    env = resolve_run_environment(
        cfg,
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )

    assert env.agent_id == "kaixing"
    assert env.runtime_root == tmp_path / "agent-runtime"
    assert env.runtime_home == tmp_path / "agent-runtime" / "runtime-homes" / "kaixing"
    assert env.agent_env_dir == tmp_path / "agent-runtime" / "agent-envs" / "kaixing"
    assert env.workspace_dir == tmp_path / "workspaces" / "calculator-development"
    assert (
        env.lock_path
        == tmp_path / "agent-runtime" / "locks" / "workspace-calculator-development.lock"
    )


def test_resolve_run_environment_rejects_path_escape(tmp_path: Path) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")

    with pytest.raises(AgentRuntimeError, match="workspace"):
        resolve_run_environment(
            cfg,
            agent_id="kaixing",
            workspace_dir=tmp_path / "agent-runtime" / "workspaces" / ".." / "escape",
        )


def test_resolve_run_environment_rejects_runtime_root_escape(
    tmp_path: Path,
) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")

    with pytest.raises(AgentRuntimeError, match="workspace"):
        resolve_run_environment(
            cfg,
            agent_id="kaixing",
            workspace_dir=tmp_path / "agent-runtime" / "escape",
        )


def test_prepare_runtime_home_creates_codex_dir(tmp_path: Path) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )

    prepare_runtime_home(env)

    assert (env.runtime_home / ".codex").is_dir()


def test_workspace_lock_rejects_second_holder(tmp_path: Path) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )
    env.workspace_dir.mkdir(parents=True)

    with workspace_lock(env):
        with pytest.raises(AgentRuntimeError, match="locked"):
            with workspace_lock(env):
                pass

    assert not env.lock_path.exists()


def test_workspace_lock_does_not_remove_another_holder(
    tmp_path: Path,
) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )
    env.workspace_dir.mkdir(parents=True)

    with workspace_lock(env):
        env.lock_path.unlink()
        env.lock_path.write_text("other-holder\n", encoding="utf-8")

    assert env.lock_path.read_text(encoding="utf-8") == "other-holder\n"


def test_write_manifest_writes_runtime_and_workspace_copy(tmp_path: Path) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )
    env.workspace_dir.mkdir(parents=True)
    manifest = RunManifest(
        version=1,
        run_id="run-1",
        agent_id="kaixing",
        agent_env_dir=str(env.agent_env_dir),
        agent_env_commit=None,
        workspace_dir=str(env.workspace_dir),
        workspace_commit_before=None,
        workspace_commit_after=None,
        sandbox_policy="workspace-write",
        started_at="2026-04-27T21:00:00",
        finished_at="2026-04-27T21:01:00",
        status="completed",
        engine="codex",
        channel_id="channel-1",
        message_id="message-1",
    )

    runtime_manifest = write_manifest(env, manifest)

    assert (
        runtime_manifest
        == tmp_path / "agent-runtime" / "runs" / "run-1" / "manifest.json"
    )
    runtime_payload = json.loads(runtime_manifest.read_text(encoding="utf-8"))
    assert runtime_payload["run_id"] == "run-1"
    assert runtime_payload["channel_id"] == "channel-1"
    assert runtime_payload["message_id"] == "message-1"
    workspace_manifest = env.workspace_dir / ".agent" / "run-manifests" / "run-1.json"
    assert json.loads(workspace_manifest.read_text(encoding="utf-8"))[
        "agent_id"
    ] == "kaixing"


@pytest.mark.parametrize("run_id", ["", "..", "../x", "a/b", r"a\b"])
def test_write_manifest_rejects_invalid_run_id(tmp_path: Path, run_id: str) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )
    env.workspace_dir.mkdir(parents=True)
    manifest = RunManifest(
        version=1,
        run_id=run_id,
        agent_id="kaixing",
        agent_env_dir=str(env.agent_env_dir),
        agent_env_commit=None,
        workspace_dir=str(env.workspace_dir),
        workspace_commit_before=None,
        workspace_commit_after=None,
        sandbox_policy="workspace-write",
        started_at="2026-04-27T21:00:00",
        finished_at=None,
        status="failed",
        engine="codex",
    )

    with pytest.raises(AgentRuntimeError, match="run id"):
        write_manifest(env, manifest)


def test_write_manifest_uses_atomic_replace(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )
    env.workspace_dir.mkdir(parents=True)
    manifest = RunManifest(
        version=1,
        run_id="run-1",
        agent_id="kaixing",
        agent_env_dir=str(env.agent_env_dir),
        agent_env_commit=None,
        workspace_dir=str(env.workspace_dir),
        workspace_commit_before=None,
        workspace_commit_after=None,
        sandbox_policy="workspace-write",
        started_at="2026-04-27T21:00:00",
        finished_at=None,
        status="completed",
        engine="codex",
    )
    original_replace = os.replace
    replaced: list[Path] = []

    def _record_replace(src: str | Path, dst: str | Path) -> None:
        replaced.append(Path(dst))
        original_replace(src, dst)

    monkeypatch.setattr("tunapi.agent_runtime.os.replace", _record_replace)

    write_manifest(env, manifest)

    assert replaced == [
        tmp_path / "agent-runtime" / "runs" / "run-1" / "manifest.json",
        env.workspace_dir / ".agent" / "run-manifests" / "run-1.json",
    ]


def test_activate_run_environment_restores_previous_env(tmp_path: Path) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="kaixing",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )

    assert get_active_run_environment() is None
    with activate_run_environment(env):
        assert get_active_run_environment() == env
    assert get_active_run_environment() is None


def test_run_environment_subprocess_env_sets_home(tmp_path: Path) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="codeview",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )

    process_env = env.subprocess_env({"PATH": "/bin"})

    assert process_env["HOME"] == str(env.runtime_home)
    assert process_env["USERPROFILE"] == str(env.runtime_home)
    assert process_env["PATH"] == "/bin"
    assert process_env["TUNAPI_AGENT_ID"] == "codeview"
    assert process_env["TUNAPI_AGENT_ENV_DIR"] == str(env.agent_env_dir)
    assert process_env["TUNAPI_WORKSPACE_DIR"] == str(env.workspace_dir)


def test_run_environment_subprocess_env_defaults_to_current_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("TUNAPI_TEST_KEEP", "yes")
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="codeview",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )

    process_env = env.subprocess_env()

    assert process_env["TUNAPI_TEST_KEEP"] == "yes"
    assert process_env["HOME"] == str(env.runtime_home)
    assert process_env["USERPROFILE"] == str(env.runtime_home)
    assert process_env["TUNAPI_AGENT_ID"] == "codeview"


@pytest.mark.anyio
async def test_jsonl_subprocess_runner_uses_active_run_environment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = AgentRuntimeConfig(root=tmp_path / "agent-runtime")
    env = resolve_run_environment(
        cfg,
        agent_id="codeview",
        workspace_dir=tmp_path / "workspaces" / "calculator-development",
    )
    captured_env: dict[str, str] | None = None

    class _FakeProc:
        stdout = object()
        stderr = object()
        stdin = None
        pid = 123

        async def wait(self) -> int:
            return 0

    class _FakeManager:
        async def __aenter__(self) -> _FakeProc:
            return _FakeProc()

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

    def fake_manage_subprocess(*args: Any, **kwargs: Any) -> _FakeManager:
        _ = args
        nonlocal captured_env
        captured_env = kwargs["env"]
        return _FakeManager()

    async def fake_drain_stderr(*args: Any, **kwargs: Any) -> None:
        _ = args, kwargs
        return None

    class _EnvRunner(JsonlSubprocessRunner):
        engine = "env-jsonl"

        def command(self) -> str:
            return "dummy"

        def build_args(
            self,
            prompt: str,
            resume: ResumeToken | None,
            *,
            state: object,
        ) -> list[str]:
            _ = prompt, resume, state
            return []

        def stdin_payload(
            self,
            prompt: str,
            resume: ResumeToken | None,
            *,
            state: object,
        ) -> bytes | None:
            _ = prompt, resume, state
            return None

        def env(self, *, state: object) -> dict[str, str]:
            _ = state
            return {"PATH": "/bin", "HOME": "caller-home"}

        async def iter_json_lines(self, stream: object):
            _ = stream
            yield b'{"type": "completed", "resume": "sid"}'

        def translate(
            self,
            data: Any,
            *,
            state: Any,
            resume: ResumeToken | None,
            found_session: ResumeToken | None,
        ) -> list[TunapiEvent]:
            _ = data, state, resume, found_session
            token = ResumeToken(engine=self.engine, value="sid")
            return [CompletedEvent(engine=self.engine, ok=True, answer="done", resume=token)]

    monkeypatch.setattr(runner_module, "manage_subprocess", fake_manage_subprocess)
    monkeypatch.setattr(runner_module, "drain_stderr", fake_drain_stderr)

    with activate_run_environment(env):
        events = [evt async for evt in _EnvRunner().run_impl("hello", None)]

    assert any(isinstance(evt, CompletedEvent) for evt in events)
    assert captured_env is not None
    assert captured_env["PATH"] == "/bin"
    assert captured_env["HOME"] == str(env.runtime_home)
    assert captured_env["USERPROFILE"] == str(env.runtime_home)
    assert captured_env["TUNAPI_AGENT_ID"] == "codeview"
    assert captured_env["TUNAPI_AGENT_ENV_DIR"] == str(env.agent_env_dir)
    assert captured_env["TUNAPI_WORKSPACE_DIR"] == str(env.workspace_dir)


def test_git_commit_returns_head_hash_for_repo(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "Tunapi Tests"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tunapi-tests@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "README.md").write_text("hello\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert git_commit(repo) == expected


def test_git_commit_returns_none_for_non_git_dir(tmp_path: Path) -> None:
    assert git_commit(tmp_path) is None
