from __future__ import annotations

import os
import re
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import tomli_w

_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9_.-]+$")


class WorkspaceResolutionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class WorkspaceRecord:
    name: str
    path: Path
    repo: str | None = None


@dataclass(frozen=True, slots=True)
class AgentWorkspaceBinding:
    default_workspace: str


@dataclass(slots=True)
class ChannelWorkspaceBinding:
    channel_id: str
    default_workspace: str | None = None
    workspaces: dict[str, WorkspaceRecord] = field(default_factory=dict)
    agents: dict[str, AgentWorkspaceBinding] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedWorkspace:
    channel_context_dir: Path
    workspace: WorkspaceRecord
    binding_source: str


class ChannelWorkspaceStore:
    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root.expanduser().resolve(strict=False)

    def channel_dir(self, channel_id: str) -> Path:
        return self.runtime_root / "channels" / f"mattermost-{_safe_channel_id(channel_id)}"

    def bindings_path(self, channel_id: str) -> Path:
        return self.channel_dir(channel_id) / "bindings.toml"

    def load(self, channel_id: str) -> ChannelWorkspaceBinding:
        binding_path = self.bindings_path(channel_id)
        try:
            raw = binding_path.read_bytes()
        except FileNotFoundError:
            return ChannelWorkspaceBinding(channel_id=channel_id)
        except OSError as e:
            raise WorkspaceResolutionError(f"failed to read workspace bindings: {e}") from e

        try:
            data = tomllib.loads(raw.decode("utf-8"))
        except tomllib.TOMLDecodeError as e:
            raise WorkspaceResolutionError(f"malformed workspace bindings: {e}") from e

        return self._binding_from_data(channel_id, data)

    def save(self, binding: ChannelWorkspaceBinding) -> None:
        _safe_channel_id(binding.channel_id)
        self._validate_binding(binding)

        binding_path = self.bindings_path(binding.channel_id)
        binding_path.parent.mkdir(parents=True, exist_ok=True)
        payload = tomli_w.dumps(_binding_to_data(binding))
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=binding_path.parent,
                prefix=f".{binding_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp:
                tmp.write(payload)
                tmp.flush()
                os.fsync(tmp.fileno())
                tmp_path = Path(tmp.name)
            os.replace(tmp_path, binding_path)
        except OSError as e:
            raise WorkspaceResolutionError(f"failed to write workspace bindings: {e}") from e
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
                except OSError:
                    pass

    def add_workspace(
        self,
        channel_id: str,
        name: str,
        path: Path,
        repo: str | None,
    ) -> None:
        _validate_workspace_name(name)
        workspace_path = _ensure_inside_runtime_root(self.runtime_root, path)
        binding = self.load(channel_id)
        binding.workspaces[name] = WorkspaceRecord(
            name=name,
            path=workspace_path,
            repo=repo,
        )
        self.save(binding)

    def set_default(self, channel_id: str, workspace_name: str) -> None:
        _validate_workspace_name(workspace_name)
        binding = self.load(channel_id)
        if workspace_name not in binding.workspaces:
            raise WorkspaceResolutionError(f"unknown workspace: {workspace_name}")
        binding.default_workspace = workspace_name
        self.save(binding)

    def bind_agent(
        self,
        channel_id: str,
        agent_id: str,
        workspace_name: str,
    ) -> None:
        _validate_workspace_name(agent_id)
        _validate_workspace_name(workspace_name)
        binding = self.load(channel_id)
        if workspace_name not in binding.workspaces:
            raise WorkspaceResolutionError(f"unknown workspace: {workspace_name}")
        binding.agents[agent_id] = AgentWorkspaceBinding(default_workspace=workspace_name)
        self.save(binding)

    def resolve(
        self,
        channel_id: str,
        agent_id: str,
        explicit_workspace: str | None,
        fallback_workspace: Path | None,
    ) -> ResolvedWorkspace | None:
        binding = self.load(channel_id)
        channel_context_dir = self.channel_dir(channel_id)

        if explicit_workspace is not None:
            return ResolvedWorkspace(
                channel_context_dir=channel_context_dir,
                workspace=self._require_workspace(binding, explicit_workspace),
                binding_source="explicit",
            )

        _validate_workspace_name(agent_id)
        agent_binding = binding.agents.get(agent_id)
        if agent_binding is not None:
            return ResolvedWorkspace(
                channel_context_dir=channel_context_dir,
                workspace=self._require_workspace(
                    binding,
                    agent_binding.default_workspace,
                ),
                binding_source="agent-default",
            )

        if binding.default_workspace is not None:
            return ResolvedWorkspace(
                channel_context_dir=channel_context_dir,
                workspace=self._require_workspace(binding, binding.default_workspace),
                binding_source="channel-default",
            )

        if fallback_workspace is not None:
            fallback_path = _ensure_inside_runtime_root(
                self.runtime_root,
                fallback_workspace,
            )
            return ResolvedWorkspace(
                channel_context_dir=channel_context_dir,
                workspace=WorkspaceRecord(
                    name=fallback_path.name,
                    path=fallback_path,
                    repo=None,
                ),
                binding_source="project-binding",
            )

        return None

    def _binding_from_data(
        self,
        channel_id: str,
        data: dict[str, Any],
    ) -> ChannelWorkspaceBinding:
        default_workspace = _optional_string(data.get("default_workspace"))
        binding = ChannelWorkspaceBinding(
            channel_id=channel_id,
            default_workspace=default_workspace,
        )

        workspaces = data.get("workspaces", {})
        if not isinstance(workspaces, dict):
            raise WorkspaceResolutionError("workspaces must be a table")
        for name, workspace_data in workspaces.items():
            _validate_workspace_name(name)
            if not isinstance(workspace_data, dict):
                raise WorkspaceResolutionError(f"workspace {name} must be a table")
            path_value = workspace_data.get("path")
            if not isinstance(path_value, str):
                raise WorkspaceResolutionError(f"workspace {name} path must be a string")
            repo = _optional_string(workspace_data.get("repo"))
            binding.workspaces[name] = WorkspaceRecord(
                name=name,
                path=_ensure_inside_runtime_root(self.runtime_root, Path(path_value)),
                repo=repo,
            )

        agents = data.get("agents", {})
        if not isinstance(agents, dict):
            raise WorkspaceResolutionError("agents must be a table")
        for agent_id, agent_data in agents.items():
            _validate_workspace_name(agent_id)
            if not isinstance(agent_data, dict):
                raise WorkspaceResolutionError(f"agent {agent_id} must be a table")
            workspace_name = agent_data.get("default_workspace")
            if not isinstance(workspace_name, str):
                raise WorkspaceResolutionError(
                    f"agent {agent_id} default_workspace must be a string",
                )
            _validate_workspace_name(workspace_name)
            binding.agents[agent_id] = AgentWorkspaceBinding(
                default_workspace=workspace_name,
            )

        self._validate_binding(binding)
        return binding

    def _validate_binding(self, binding: ChannelWorkspaceBinding) -> None:
        _safe_channel_id(binding.channel_id)
        if binding.default_workspace is not None:
            _validate_workspace_name(binding.default_workspace)
            if binding.default_workspace not in binding.workspaces:
                raise WorkspaceResolutionError(
                    f"unknown workspace: {binding.default_workspace}",
                )
        for name, workspace in binding.workspaces.items():
            _validate_workspace_name(name)
            if workspace.name != name:
                raise WorkspaceResolutionError("workspace name mismatch")
            _ensure_inside_runtime_root(self.runtime_root, workspace.path)
        for agent_id, agent_binding in binding.agents.items():
            _validate_workspace_name(agent_id)
            _validate_workspace_name(agent_binding.default_workspace)
            if agent_binding.default_workspace not in binding.workspaces:
                raise WorkspaceResolutionError(
                    f"unknown workspace: {agent_binding.default_workspace}",
                )

    @staticmethod
    def _require_workspace(
        binding: ChannelWorkspaceBinding,
        workspace_name: str,
    ) -> WorkspaceRecord:
        _validate_workspace_name(workspace_name)
        try:
            return binding.workspaces[workspace_name]
        except KeyError:
            raise WorkspaceResolutionError(f"unknown workspace: {workspace_name}") from None


def _safe_channel_id(channel_id: str) -> str:
    if not _SAFE_COMPONENT.fullmatch(channel_id):
        raise WorkspaceResolutionError("channel id must be a safe path component")
    return channel_id


def _validate_workspace_name(name: str) -> None:
    if not _SAFE_COMPONENT.fullmatch(name):
        raise WorkspaceResolutionError("workspace name must be a safe path component")


def _ensure_inside_runtime_root(runtime_root: Path, path: Path) -> Path:
    resolved_root = runtime_root.expanduser().resolve(strict=False)
    resolved_path = path.expanduser().resolve(strict=False)
    if not resolved_path.is_relative_to(resolved_root):
        raise WorkspaceResolutionError("workspace path must be inside runtime root")
    return resolved_path


def _binding_to_data(binding: ChannelWorkspaceBinding) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if binding.default_workspace is not None:
        data["default_workspace"] = binding.default_workspace
    data["workspaces"] = {
        name: _workspace_to_data(workspace)
        for name, workspace in binding.workspaces.items()
    }
    data["agents"] = {
        agent_id: {"default_workspace": agent.default_workspace}
        for agent_id, agent in binding.agents.items()
    }
    return data


def _workspace_to_data(workspace: WorkspaceRecord) -> dict[str, str]:
    data = {"path": str(workspace.path)}
    if workspace.repo is not None:
        data["repo"] = workspace.repo
    return data


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise WorkspaceResolutionError("expected string value")
    return value
