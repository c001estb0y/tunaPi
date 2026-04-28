from __future__ import annotations

import dataclasses
import json
import os
import shutil
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path

from .utils.git import git_stdout


class AgentRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class AgentRuntimeConfig:
    root: Path
    enabled: bool = False
    default_agent_id: str | None = None

    @property
    def agent_envs_dir(self) -> Path:
        return self.root / "agent-envs"

    @property
    def runtime_homes_dir(self) -> Path:
        return self.root / "runtime-homes"

    @property
    def workspaces_dir(self) -> Path:
        return self.root / "workspaces"

    @property
    def locks_dir(self) -> Path:
        return self.root / "locks"

    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.root / "logs"


@dataclass(frozen=True, slots=True)
class RunEnvironment:
    agent_id: str
    runtime_root: Path
    runtime_home: Path
    agent_env_dir: Path
    workspace_dir: Path
    lock_path: Path

    def subprocess_env(self, base_env: Mapping[str, str] | None = None) -> dict[str, str]:
        env = dict(base_env or os.environ)
        env["HOME"] = str(self.runtime_home)
        env["USERPROFILE"] = str(self.runtime_home)
        env["TUNAPI_AGENT_ID"] = self.agent_id
        env["TUNAPI_AGENT_ENV_DIR"] = str(self.agent_env_dir)
        env["TUNAPI_WORKSPACE_DIR"] = str(self.workspace_dir)
        return env


@dataclass(frozen=True, slots=True)
class RunManifest:
    version: int
    run_id: str
    agent_id: str
    agent_env_dir: str
    agent_env_commit: str | None
    workspace_dir: str
    workspace_commit_before: str | None
    workspace_commit_after: str | None
    sandbox_policy: str
    started_at: str
    finished_at: str | None
    status: str
    engine: str
    channel_id: str | None = None
    message_id: str | None = None


_ACTIVE_RUN_ENVIRONMENT: ContextVar[RunEnvironment | None] = ContextVar(
    "tunapi.active_run_environment", default=None
)


def resolve_run_environment(
    cfg: AgentRuntimeConfig,
    *,
    agent_id: str,
    workspace_dir: Path,
) -> RunEnvironment:
    _validate_path_component(agent_id, label="agent id")
    _validate_workspace_dir(workspace_dir, root=cfg.root)

    root = cfg.root
    workspace_name = workspace_dir.name
    _validate_path_component(workspace_name, label="workspace")

    return RunEnvironment(
        agent_id=agent_id,
        runtime_root=root,
        runtime_home=cfg.runtime_homes_dir / agent_id,
        agent_env_dir=cfg.agent_envs_dir / agent_id,
        workspace_dir=workspace_dir,
        lock_path=cfg.locks_dir / f"workspace-{workspace_name}.lock",
    )


def prepare_runtime_home(env: RunEnvironment) -> None:
    (env.runtime_home / ".codex").mkdir(parents=True, exist_ok=True)
    (env.runtime_home / ".agents").mkdir(parents=True, exist_ok=True)
    env.agent_env_dir.mkdir(parents=True, exist_ok=True)
    _map_agent_env_to_runtime_home(env)


def _map_agent_env_to_runtime_home(env: RunEnvironment) -> None:
    for target in (
        env.runtime_home / ".codex" / "AGENTS.md",
        env.runtime_home / ".codex" / "AGENTS.override.md",
        env.runtime_home / ".codex" / "config.toml",
        env.runtime_home / ".codex" / "rules",
        env.runtime_home / ".agents" / "skills",
    ):
        _reset_managed_target(target)

    _copy_file_if_exists(
        env.agent_env_dir / "AGENTS.md",
        env.runtime_home / ".codex" / "AGENTS.md",
    )
    _copy_file_if_exists(
        env.agent_env_dir / "AGENTS.override.md",
        env.runtime_home / ".codex" / "AGENTS.override.md",
    )
    _copy_file_if_exists(
        env.agent_env_dir / ".codex" / "config.toml",
        env.runtime_home / ".codex" / "config.toml",
    )
    _copy_dir_if_exists(
        env.agent_env_dir / ".codex" / "rules",
        env.runtime_home / ".codex" / "rules",
    )
    _copy_dir_if_exists(
        env.agent_env_dir / ".agents" / "skills",
        env.runtime_home / ".agents" / "skills",
    )


def _copy_file_if_exists(src: Path, dst: Path) -> None:
    if not src.is_file():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_dir_if_exists(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    shutil.copytree(src, dst, dirs_exist_ok=True)


def _reset_managed_target(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def git_commit(path: Path) -> str | None:
    return git_stdout(["rev-parse", "HEAD"], cwd=path)


def write_manifest(env: RunEnvironment, manifest: RunManifest) -> Path:
    _validate_path_component(manifest.run_id, label="run id")

    payload = dataclasses.asdict(manifest)
    content = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

    runtime_manifest = env.runtime_root / "runs" / manifest.run_id / "manifest.json"
    workspace_manifest = (
        env.workspace_dir / ".agent" / "run-manifests" / f"{manifest.run_id}.json"
    )

    for path in (runtime_manifest, workspace_manifest):
        _atomic_write_text(path, content)

    return runtime_manifest


@contextmanager
def workspace_lock(env: RunEnvironment) -> Iterator[None]:
    env.lock_path.parent.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}:{uuid.uuid4().hex}\n"
    try:
        fd = os.open(str(env.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise AgentRuntimeError(f"workspace is locked: {env.workspace_dir}") from exc

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
            lock_file.write(token)
        yield
    finally:
        try:
            current_token = env.lock_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            pass
        else:
            if current_token == token:
                env.lock_path.unlink()


@contextmanager
def activate_run_environment(env: RunEnvironment) -> Iterator[None]:
    token = set_active_run_environment(env)
    try:
        yield
    finally:
        reset_active_run_environment(token)


def get_active_run_environment() -> RunEnvironment | None:
    return _ACTIVE_RUN_ENVIRONMENT.get()


def set_active_run_environment(env: RunEnvironment | None) -> Token[RunEnvironment | None]:
    return _ACTIVE_RUN_ENVIRONMENT.set(env)


def reset_active_run_environment(token: Token[RunEnvironment | None]) -> None:
    _ACTIVE_RUN_ENVIRONMENT.reset(token)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _validate_workspace_dir(workspace_dir: Path, *, root: Path) -> None:
    if any(part == ".." for part in workspace_dir.parts):
        raise AgentRuntimeError("workspace path cannot contain '..'")

    root_resolved = root.resolve(strict=False)
    workspace_resolved = workspace_dir.resolve(strict=False)
    if not workspace_resolved.is_relative_to(root_resolved):
        return

    runtime_workspaces = root_resolved / "workspaces"
    if not workspace_resolved.is_relative_to(runtime_workspaces):
        raise AgentRuntimeError("workspace path inside runtime root must use workspaces/")


def _validate_path_component(value: str, *, label: str) -> None:
    if not value or value in {".", ".."}:
        raise AgentRuntimeError(f"{label} cannot be empty or relative")
    if any(separator in value for separator in ("/", "\\")):
        raise AgentRuntimeError(f"{label} cannot contain path separators")
