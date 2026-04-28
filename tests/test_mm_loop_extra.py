"""Extra tests for tunapi.mattermost.loop — command dispatch, prompt resolution,
file command handling, roundtable archiving, and startup helpers."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tunapi.agent_runtime import (
    AgentRuntimeConfig,
    AgentRuntimeError,
    resolve_run_environment,
)
from tunapi.core.commands import parse_command
from tunapi.core.roundtable import RoundtableSession, RoundtableStore
from tunapi.mattermost.api_models import Post, PostList, User
import tunapi.mattermost.loop as loop
from tunapi.mattermost.loop import (
    _ResolvedPrompt,
    _archive_roundtable,
    _dispatch_message,
    _handle_cancel_reaction,
    _handle_file_command,
    _resolve_prompt,
    _run_engine,
    _send_startup,
    _try_dispatch_command,
)
from tunapi.mattermost.types import (
    MattermostIncomingMessage,
    MattermostReactionEvent,
)
from tunapi.transport import MessageRef, RenderedMessage


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_msg(
    text: str = "",
    channel_id: str = "ch1",
    post_id: str = "p1",
    root_id: str = "",
    channel_type: str = "O",
    file_ids: tuple[str, ...] = (),
    sender_id: str = "u1",
    sender_username: str = "alice",
) -> MattermostIncomingMessage:
    return MattermostIncomingMessage(
        channel_id=channel_id,
        post_id=post_id,
        text=text,
        root_id=root_id,
        sender_id=sender_id,
        sender_username=sender_username,
        channel_type=channel_type,
        file_ids=file_ids,
    )


def _make_cfg(
    *,
    files_enabled: bool = False,
    voice_enabled: bool = False,
    bot_username: str = "tunabot",
    channel_id: str = "ch1",
    session_mode: str = "stateless",
) -> MagicMock:
    """Build a minimal MattermostBridgeConfig mock."""
    cfg = MagicMock()
    cfg.files_enabled = files_enabled
    cfg.voice_enabled = voice_enabled
    cfg.bot_username = bot_username
    cfg.bot_user_id = "bot_uid"
    cfg.channel_id = channel_id
    cfg.session_mode = session_mode
    cfg.startup_msg = "Bot started"
    cfg.files_deny_globs = ()
    cfg.files_max_upload_bytes = 20 * 1024 * 1024
    cfg.files_max_download_bytes = 50 * 1024 * 1024
    cfg.voice_max_bytes = 10 * 1024 * 1024
    cfg.voice_model = "gpt-4o-mini-transcribe"
    cfg.voice_base_url = None
    cfg.voice_api_key = None
    cfg.projects_root = None

    cfg.runtime = MagicMock()
    cfg.runtime.projects_root = None
    cfg.runtime.default_engine = "claude"
    cfg.trigger_mode = "all"

    cfg.exec_cfg = MagicMock()
    cfg.exec_cfg.transport = AsyncMock()
    cfg.exec_cfg.transport.send = AsyncMock(
        return_value=MessageRef(channel_id="ch1", message_id="200")
    )
    return cfg


def _make_resolved_message(
    prompt: str = "hello",
    engine_override: str | None = None,
) -> MagicMock:
    resolved = MagicMock()
    resolved.prompt = prompt
    resolved.resume_token = None
    resolved.engine_override = engine_override
    resolved.context = None
    return resolved


def _make_resolved_runner(*, issue: str | None = None) -> MagicMock:
    resolved = MagicMock()
    resolved.issue = issue
    resolved.runner = MagicMock()
    return resolved


@pytest.mark.anyio()
async def test_run_engine_uses_agent_runtime_lock_and_manifest(tmp_path, monkeypatch):
    cfg = _make_cfg(bot_username="kaixing")
    workspace = tmp_path / "agent-runtime" / "workspaces" / "calculator-development"
    workspace.mkdir(parents=True)
    runtime_config = AgentRuntimeConfig(root=tmp_path / "agent-runtime", enabled=True)

    cfg.runtime.resolve_message.return_value = _make_resolved_message()
    cfg.runtime.resolve_engine.return_value = "codex"
    cfg.runtime.format_context_line.return_value = None
    cfg.runtime.resolve_run_cwd.return_value = workspace
    cfg.runtime.resolve_runner.return_value = _make_resolved_runner()
    cfg.runtime.is_resume_line = MagicMock()
    cfg.runtime.resolve_run_environment.side_effect = (
        lambda *, agent_id, workspace_dir: resolve_run_environment(
            runtime_config,
            agent_id=agent_id,
            workspace_dir=workspace_dir,
        )
    )

    async def fake_handle_message(*args, **kwargs):
        return "ok"

    monkeypatch.setattr(loop, "handle_message", fake_handle_message)

    await _run_engine(
        _ResolvedPrompt(text="hello", file_context=""),
        _make_msg(text="hello"),
        cfg,
        {},
        AsyncMock(),
        None,
        AsyncMock(),
    )

    manifests = list((workspace / ".agent" / "run-manifests").glob("*.json"))
    assert len(manifests) == 1
    data = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert data["agent_id"] == "kaixing"
    assert data["workspace_dir"] == str(workspace)
    assert data["status"] == "completed"
    assert not (runtime_config.locks_dir / "workspace-calculator-development.lock").exists()


@pytest.mark.anyio()
async def test_run_engine_marks_manifest_failed_when_handle_message_returns_none(
    tmp_path,
    monkeypatch,
):
    cfg = _make_cfg(bot_username="kaixing")
    workspace = tmp_path / "agent-runtime" / "workspaces" / "calculator-development"
    workspace.mkdir(parents=True)
    runtime_config = AgentRuntimeConfig(root=tmp_path / "agent-runtime", enabled=True)

    cfg.runtime.resolve_message.return_value = _make_resolved_message()
    cfg.runtime.resolve_engine.return_value = "codex"
    cfg.runtime.format_context_line.return_value = None
    cfg.runtime.resolve_run_cwd.return_value = workspace
    cfg.runtime.resolve_runner.return_value = _make_resolved_runner()
    cfg.runtime.is_resume_line = MagicMock()
    cfg.runtime.resolve_run_environment.side_effect = (
        lambda *, agent_id, workspace_dir: resolve_run_environment(
            runtime_config,
            agent_id=agent_id,
            workspace_dir=workspace_dir,
        )
    )

    async def fake_handle_message(*args, **kwargs):
        return None

    monkeypatch.setattr(loop, "handle_message", fake_handle_message)

    await _run_engine(
        _ResolvedPrompt(text="hello", file_context=""),
        _make_msg(text="hello"),
        cfg,
        {},
        AsyncMock(),
        None,
        AsyncMock(),
    )

    manifests = list((workspace / ".agent" / "run-manifests").glob("*.json"))
    assert len(manifests) == 1
    data = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert data["status"] == "failed"


@pytest.mark.anyio()
async def test_run_engine_swallows_agent_runtime_resolution_errors(monkeypatch):
    cfg = _make_cfg(bot_username="kaixing")
    cfg.runtime.resolve_message.return_value = _make_resolved_message()
    cfg.runtime.resolve_engine.return_value = "codex"
    cfg.runtime.format_context_line.return_value = None
    cfg.runtime.resolve_run_cwd.return_value = "/workspace"
    cfg.runtime.resolve_runner.return_value = _make_resolved_runner()
    cfg.runtime.is_resume_line = MagicMock()
    cfg.runtime.resolve_run_environment.side_effect = AgentRuntimeError("bad runtime")
    handle_message = AsyncMock(return_value="ok")
    monkeypatch.setattr(loop, "handle_message", handle_message)

    await _run_engine(
        _ResolvedPrompt(text="hello", file_context=""),
        _make_msg(text="hello"),
        cfg,
        {},
        AsyncMock(),
        None,
        AsyncMock(),
    )

    handle_message.assert_not_awaited()


# ---------------------------------------------------------------------------
# _handle_file_command tests
# ---------------------------------------------------------------------------


class TestHandleFileCommand:
    @pytest.mark.anyio()
    async def test_files_disabled_sends_message(self):
        cfg = _make_cfg(files_enabled=False)
        msg = _make_msg(text="!file put")

        await _handle_file_command("put", msg, cfg)

        cfg.exec_cfg.transport.send.assert_called()
        call_kwargs = cfg.exec_cfg.transport.send.call_args[1]
        assert "disabled" in call_kwargs["message"].text.lower()

    @pytest.mark.anyio()
    async def test_put_without_files_warns(self):
        cfg = _make_cfg(files_enabled=True)
        msg = _make_msg(text="!file put", file_ids=())

        await _handle_file_command("put", msg, cfg)

        call_kwargs = cfg.exec_cfg.transport.send.call_args[1]
        assert "attach" in call_kwargs["message"].text.lower()

    @pytest.mark.anyio()
    async def test_get_without_path_shows_usage(self):
        cfg = _make_cfg(files_enabled=True)
        msg = _make_msg(text="!file get")

        await _handle_file_command("get  ", msg, cfg)

        call_kwargs = cfg.exec_cfg.transport.send.call_args[1]
        assert "usage" in call_kwargs["message"].text.lower()

    @pytest.mark.anyio()
    async def test_unknown_subcmd_shows_usage(self):
        cfg = _make_cfg(files_enabled=True)
        msg = _make_msg(text="!file delete")

        await _handle_file_command("delete", msg, cfg)

        call_kwargs = cfg.exec_cfg.transport.send.call_args[1]
        assert "usage" in call_kwargs["message"].text.lower()

    @pytest.mark.anyio()
    async def test_put_with_files_calls_put_files(self):
        cfg = _make_cfg(files_enabled=True)
        msg = _make_msg(file_ids=("f1",))

        with patch("tunapi.mattermost.loop._put_files", new_callable=AsyncMock) as mock_put:
            mock_result = MagicMock()
            mock_result.message = "saved file.txt"
            mock_put.return_value = [mock_result]

            await _handle_file_command("put", msg, cfg)

        call_kwargs = cfg.exec_cfg.transport.send.call_args[1]
        assert "saved file.txt" in call_kwargs["message"].text

    @pytest.mark.anyio()
    async def test_put_no_results(self):
        cfg = _make_cfg(files_enabled=True)
        msg = _make_msg(file_ids=("f1",))

        with patch("tunapi.mattermost.loop._put_files", new_callable=AsyncMock) as mock_put:
            mock_put.return_value = []
            await _handle_file_command("put", msg, cfg)

        call_kwargs = cfg.exec_cfg.transport.send.call_args[1]
        assert "No files processed" in call_kwargs["message"].text


# ---------------------------------------------------------------------------
# _try_dispatch_command tests
# ---------------------------------------------------------------------------


class TestTryDispatchCommand:
    @pytest.fixture()
    def sessions(self):
        s = AsyncMock()
        s.clear = AsyncMock()
        s.has_any = AsyncMock(return_value=False)
        return s

    @pytest.fixture()
    def chat_prefs(self):
        cp = AsyncMock()
        cp.get_context = AsyncMock(return_value=None)
        cp.get_default_engine = AsyncMock(return_value=None)
        return cp

    @pytest.mark.anyio()
    async def test_not_a_command_returns_false(self, sessions, chat_prefs):
        msg = _make_msg(text="hello world")
        cfg = _make_cfg()
        send = AsyncMock()

        result = await _try_dispatch_command(
            msg, cfg, {}, sessions, chat_prefs, None, send
        )
        assert result is False

    @pytest.mark.anyio()
    async def test_new_clears_session(self, sessions, chat_prefs):
        msg = _make_msg(text="!new")
        cfg = _make_cfg()
        sent: list[RenderedMessage] = []
        send = AsyncMock(side_effect=lambda m: sent.append(m))

        result = await _try_dispatch_command(
            msg, cfg, {}, sessions, chat_prefs, None, send
        )

        assert result is True
        sessions.clear.assert_called_once_with("ch1")
        assert "새 대화" in sent[0].text

    @pytest.mark.anyio()
    async def test_new_clears_journal(self, sessions, chat_prefs):
        msg = _make_msg(text="!new")
        cfg = _make_cfg()
        journal = AsyncMock()
        send = AsyncMock()

        await _try_dispatch_command(
            msg, cfg, {}, sessions, chat_prefs, None, send, journal=journal
        )

        journal.mark_reset.assert_called_once_with("ch1")

    @pytest.mark.anyio()
    async def test_help_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!help")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch("tunapi.mattermost.loop.handle_help", new_callable=AsyncMock) as mock_help:
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_help.assert_called_once()

    @pytest.mark.anyio()
    async def test_unknown_command_returns_false(self, sessions, chat_prefs):
        msg = _make_msg(text="!nonexistent_xyz")
        cfg = _make_cfg()
        send = AsyncMock()

        result = await _try_dispatch_command(
            msg, cfg, {}, sessions, chat_prefs, None, send
        )

        assert result is False

    @pytest.mark.anyio()
    async def test_cancel_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!cancel")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch("tunapi.mattermost.loop.handle_cancel", new_callable=AsyncMock) as mock_cancel:
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_cancel.assert_called_once()

    @pytest.mark.anyio()
    async def test_status_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!status")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch("tunapi.mattermost.loop.handle_status", new_callable=AsyncMock) as mock_status:
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_status.assert_called_once()
        sessions.has_any.assert_called_once_with("ch1")

    @pytest.mark.anyio()
    async def test_file_command_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!file put")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch(
            "tunapi.mattermost.loop._handle_file_command", new_callable=AsyncMock
        ) as mock_file:
            mock_file.return_value = True
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_file.assert_called_once()

    @pytest.mark.anyio()
    async def test_model_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!model claude opus")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch("tunapi.mattermost.loop.handle_model", new_callable=AsyncMock) as mock_model:
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_model.assert_called_once()

    @pytest.mark.anyio()
    async def test_trigger_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!trigger mentions")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch("tunapi.mattermost.loop.handle_trigger", new_callable=AsyncMock) as mock_trigger:
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_trigger.assert_called_once()

    @pytest.mark.anyio()
    async def test_new_clears_project_session_when_bound(self, sessions, chat_prefs):
        """!new should clear unified project session if channel is bound."""
        ctx = MagicMock()
        ctx.project = "myproject"
        chat_prefs.get_context.return_value = ctx

        msg = _make_msg(text="!new")
        cfg = _make_cfg()
        send = AsyncMock()
        project_sessions = AsyncMock()

        result = await _try_dispatch_command(
            msg, cfg, {}, sessions, chat_prefs, None, send,
            project_sessions=project_sessions,
        )

        assert result is True
        project_sessions.clear.assert_called_once_with("myproject")

    @pytest.mark.anyio()
    async def test_persona_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!persona list")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch("tunapi.mattermost.loop.handle_persona", new_callable=AsyncMock) as mock_persona:
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_persona.assert_called_once()

    @pytest.mark.anyio()
    async def test_project_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!project list")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch("tunapi.mattermost.loop.handle_project", new_callable=AsyncMock) as mock_project:
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_project.assert_called_once()

    @pytest.mark.anyio()
    async def test_workspace_add_and_bind_commands(self, tmp_path: Path) -> None:
        cfg = _make_cfg(bot_username="codeview")
        cfg.runtime.set_agent_runtime(
            AgentRuntimeConfig(root=tmp_path / "agent-runtime", enabled=True)
        )
        workspace = tmp_path / "agent-runtime" / "workspaces" / "agent-mem"
        workspace.mkdir(parents=True)
        send = AsyncMock()

        added = await _try_dispatch_command(
            _make_msg(text=f"!workspace add agent-mem {workspace}"),
            cfg,
            {},
            MagicMock(),
            None,
            None,
            send,
        )
        bound = await _try_dispatch_command(
            _make_msg(text="!workspace bind codeview agent-mem"),
            cfg,
            {},
            MagicMock(),
            None,
            None,
            send,
        )

        assert added is True
        assert bound is True
        assert send.await_count == 2
        assert "Workspace `agent-mem` added" in send.await_args_list[0].args[0].text
        assert "Bound `codeview` to `agent-mem`" in send.await_args_list[1].args[0].text

    @pytest.mark.anyio()
    async def test_models_dispatches(self, sessions, chat_prefs):
        msg = _make_msg(text="!models")
        cfg = _make_cfg()
        send = AsyncMock()

        with patch("tunapi.mattermost.loop.handle_models", new_callable=AsyncMock) as mock_models:
            result = await _try_dispatch_command(
                msg, cfg, {}, sessions, chat_prefs, None, send
            )

        assert result is True
        mock_models.assert_called_once()


# ---------------------------------------------------------------------------
# _resolve_prompt tests
# ---------------------------------------------------------------------------


class TestResolvePrompt:
    @pytest.mark.anyio()
    async def test_auto_file_put_no_text(self):
        """Files with no text → auto upload, return None."""
        cfg = _make_cfg(files_enabled=True)
        msg = _make_msg(text="", file_ids=("f1",))
        send = AsyncMock()

        with patch("tunapi.mattermost.loop._put_files", new_callable=AsyncMock) as mock_put:
            mock_result = MagicMock()
            mock_result.message = "saved"
            mock_put.return_value = [mock_result]

            result = await _resolve_prompt(msg, cfg, None, send)

        assert result is None
        send.assert_called_once()

    @pytest.mark.anyio()
    async def test_returns_none_when_trigger_not_met(self):
        """Non-DM, mentions mode, no mention → None."""
        cfg = _make_cfg(bot_username="tunabot")
        msg = _make_msg(text="hello world", channel_type="O")
        send = AsyncMock()

        with patch(
            "tunapi.mattermost.loop.resolve_trigger_mode", new_callable=AsyncMock
        ) as mock_trigger_mode:
            mock_trigger_mode.return_value = "mentions"
            result = await _resolve_prompt(msg, cfg, None, send)

        assert result is None

    @pytest.mark.anyio()
    async def test_uses_configured_trigger_mode_as_default(self):
        """Transport config trigger_mode applies when no channel override exists."""
        cfg = _make_cfg(bot_username="tunabot")
        cfg.trigger_mode = "mentions"
        msg = _make_msg(text="hello world", channel_type="O")
        send = AsyncMock()

        result = await _resolve_prompt(msg, cfg, None, send)

        assert result is None

    @pytest.mark.anyio()
    async def test_returns_resolved_on_dm(self):
        """DMs always trigger."""
        cfg = _make_cfg(bot_username="tunabot")
        msg = _make_msg(text="hello", channel_type="D")
        send = AsyncMock()

        result = await _resolve_prompt(msg, cfg, None, send)

        assert result is not None
        assert result.text == "hello"

    @pytest.mark.anyio()
    async def test_trigger_all_returns_resolved(self):
        """In 'all' mode, every message triggers."""
        cfg = _make_cfg(bot_username="tunabot")
        msg = _make_msg(text="hello", channel_type="O")
        send = AsyncMock()

        with patch(
            "tunapi.mattermost.loop.resolve_trigger_mode", new_callable=AsyncMock
        ) as mock_trigger_mode:
            mock_trigger_mode.return_value = "all"
            result = await _resolve_prompt(msg, cfg, None, send)

        assert result is not None
        assert result.text == "hello"

    @pytest.mark.anyio()
    async def test_strips_mention_from_prompt(self):
        cfg = _make_cfg(bot_username="tunabot")
        msg = _make_msg(text="@tunabot what is this?", channel_type="D")
        send = AsyncMock()

        result = await _resolve_prompt(msg, cfg, None, send)

        assert result is not None
        assert "@tunabot" not in result.text
        assert "what is this?" in result.text

    @pytest.mark.anyio()
    async def test_empty_text_after_strip_returns_none(self):
        cfg = _make_cfg(bot_username="tunabot")
        msg = _make_msg(text="@tunabot", channel_type="D")
        send = AsyncMock()

        result = await _resolve_prompt(msg, cfg, None, send)

        assert result is None

    @pytest.mark.anyio()
    async def test_main_channel_bot_message_is_not_dispatched_to_engine(self):
        """Bot-authored main-channel posts must not trigger another bot."""
        cfg = _make_cfg(bot_username="tunabot")
        cfg.bot = MagicMock()
        cfg.bot.get_user = AsyncMock(return_value=MagicMock(is_bot=True))
        cfg.cross_roundtable_enabled = True
        msg = _make_msg(
            text="🎯 **圆桌会议已开启** @tunabot",
            channel_type="O",
            sender_id="other-bot",
            sender_username="kaixing",
        )

        with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
            await _dispatch_message(
                msg,
                cfg,
                {},
                MagicMock(),
                None,
                None,
                None,
                None,
                None,
                None,
            )

        run.assert_not_awaited()

    @pytest.mark.anyio()
    async def test_voice_transcription_used_when_available(self):
        cfg = _make_cfg(voice_enabled=True)
        msg = _make_msg(
            text="",
            channel_type="D",
            file_ids=("f1",),
        )
        send = AsyncMock()

        with patch(
            "tunapi.mattermost.loop._handle_voice", new_callable=AsyncMock
        ) as mock_voice:
            mock_voice.return_value = "transcribed text"
            result = await _resolve_prompt(msg, cfg, None, send)

        assert result is not None
        assert result.text == "transcribed text"

    @pytest.mark.anyio()
    async def test_file_context_appended_to_prompt(self):
        cfg = _make_cfg(files_enabled=True)
        msg = _make_msg(text="check this", channel_type="D", file_ids=("f1",))
        send = AsyncMock()

        with patch("tunapi.mattermost.loop._put_files", new_callable=AsyncMock) as mock_put:
            mock_result = MagicMock()
            mock_result.ok = True
            mock_result.path = "/tmp/file.py"
            mock_result.message = "saved"
            mock_put.return_value = [mock_result]

            result = await _resolve_prompt(msg, cfg, None, send)

        assert result is not None
        assert "/tmp/file.py" in result.text
        assert "check this" in result.text
        assert result.file_context != ""

    @pytest.mark.anyio()
    async def test_empty_prompt_returns_none(self):
        cfg = _make_cfg()
        msg = _make_msg(text="", channel_type="D")
        send = AsyncMock()

        with patch(
            "tunapi.mattermost.loop._handle_voice", new_callable=AsyncMock
        ) as mock_voice:
            mock_voice.return_value = None
            result = await _resolve_prompt(msg, cfg, None, send)

        assert result is None

    @pytest.mark.anyio()
    async def test_completed_roundtable_thread_human_mention_gets_thread_context(self):
        cfg = _make_cfg(bot_username="kaixing")
        cfg.cross_roundtable_enabled = True
        cfg.bot = MagicMock()
        cfg.bot._client = MagicMock()
        cfg.bot._client.get_thread = AsyncMock(
            return_value=PostList(
                order=["root1", "p1", "p2", "human1"],
                posts={
                    "root1": Post(
                        id="root1",
                        channel_id="ch1",
                        user_id="u-human",
                        message='<!-- tunapi:roundtable {"version":1,"topic":"讨论计算器实现","participants":["kaixing","codeview"],"max_rounds":1} -->',
                    ),
                    "p1": Post(
                        id="p1",
                        channel_id="ch1",
                        user_id="u-kaixing",
                        root_id="root1",
                        message="前端使用表单和结果区 @codeview",
                        create_at=1,
                    ),
                    "p2": Post(
                        id="p2",
                        channel_id="ch1",
                        user_id="u-codeview",
                        root_id="root1",
                        message="后端使用纯函数和单元测试",
                        create_at=2,
                    ),
                    "human1": Post(
                        id="human1",
                        channel_id="ch1",
                        user_id="u-human",
                        root_id="root1",
                        message="@kaixing 总结一下上述讨论",
                        create_at=3,
                    ),
                },
            )
        )
        cfg.bot.get_user = AsyncMock(
            side_effect=lambda user_id: User(
                id=user_id,
                username={
                    "u-human": "minusjiang",
                    "u-kaixing": "kaixing",
                    "u-codeview": "codeview",
                }[user_id],
                is_bot=user_id != "u-human",
            )
        )
        msg = _make_msg(
            text="@kaixing 总结一下上述讨论",
            root_id="root1",
            sender_id="u-human",
            sender_username="minusjiang",
        )

        with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
            await _dispatch_message(
                msg,
                cfg,
                {},
                MagicMock(),
                None,
            )

        run.assert_awaited_once()
        resolved = run.await_args.args[0]
        assert "Topic: 讨论计算器实现" in resolved.text
        assert "Participants: kaixing, codeview" in resolved.text
        assert "[kaixing]: 前端使用表单和结果区 @codeview" in resolved.text
        assert "[codeview]: 后端使用纯函数和单元测试" in resolved.text
        assert "[Current request]\n总结一下上述讨论" in resolved.text

    @pytest.mark.anyio()
    async def test_stalled_active_roundtable_human_mention_is_released_to_normal_chat(
        self,
    ):
        cfg = _make_cfg(bot_username="kaixing")
        cfg.cross_roundtable_enabled = True
        cfg.bot = MagicMock()
        cfg.bot._client = MagicMock()
        cfg.bot._client.get_thread = AsyncMock(
            return_value=PostList(
                order=["root1", "p1", "p2", "p3", "human1"],
                posts={
                    "root1": Post(
                        id="root1",
                        channel_id="ch1",
                        user_id="u-human",
                        message='<!-- tunapi:roundtable {"version":1,"topic":"agent runtime","participants":["kaixing","codeview"],"max_rounds":3} -->',
                    ),
                    "p1": Post(
                        id="p1",
                        channel_id="ch1",
                        user_id="u-kaixing",
                        root_id="root1",
                        message="建议拆成 agent env 和 task workspace @codeview",
                        create_at=1,
                    ),
                    "p2": Post(
                        id="p2",
                        channel_id="ch1",
                        user_id="u-codeview",
                        root_id="root1",
                        message="补充 runtime 与 sandbox 约束 @kaixing",
                        create_at=2,
                    ),
                    "p3": Post(
                        id="p3",
                        channel_id="ch1",
                        user_id="u-kaixing",
                        root_id="root1",
                        message="收敛成三层：agents、runtime、tasks",
                        create_at=3,
                    ),
                    "human1": Post(
                        id="human1",
                        channel_id="ch1",
                        user_id="u-human",
                        root_id="root1",
                        message="@kaixing 总结一下最终方案",
                        create_at=4,
                    ),
                },
            )
        )
        cfg.bot.get_user = AsyncMock(
            side_effect=lambda user_id: User(
                id=user_id,
                username={
                    "u-human": "minusjiang",
                    "u-kaixing": "kaixing",
                    "u-codeview": "codeview",
                }[user_id],
                is_bot=user_id != "u-human",
            )
        )
        msg = _make_msg(
            text="@kaixing 总结一下最终方案",
            root_id="root1",
            sender_id="u-human",
            sender_username="minusjiang",
        )

        with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
            await _dispatch_message(
                msg,
                cfg,
                {},
                MagicMock(),
                None,
            )

        run.assert_awaited_once()
        resolved = run.await_args.args[0]
        assert "Topic: agent runtime" in resolved.text
        assert "[Current request]\n总结一下最终方案" in resolved.text

    @pytest.mark.anyio()
    async def test_external_mention_gets_busy_message_while_roundtable_engine_running(
        self,
    ):
        cfg = _make_cfg(bot_username="kaixing")
        cfg.cross_roundtable_enabled = True
        cfg.bot = MagicMock()
        cfg.bot.get_user = AsyncMock(return_value=User(id="u-human", is_bot=False))
        msg = _make_msg(
            text="@kaixing 你现在能回答另一个问题吗",
            sender_id="u-human",
            sender_username="minusjiang",
        )
        send_calls: list[RenderedMessage] = []

        with patch("tunapi.mattermost.loop._is_roundtable_busy", return_value=True):
            with patch(
                "tunapi.mattermost.loop._send_to_channel",
                new_callable=AsyncMock,
            ) as send:
                send.side_effect = (
                    lambda _cfg, _channel_id, message: send_calls.append(message)
                )
                with patch(
                    "tunapi.mattermost.loop._run_engine",
                    new_callable=AsyncMock,
                ) as run:
                    await _dispatch_message(
                        msg,
                        cfg,
                        {},
                        MagicMock(),
                        None,
                    )

        run.assert_not_awaited()
        assert send_calls
        assert "正在参与圆桌讨论中" in send_calls[0].text

    @pytest.mark.anyio()
    async def test_roundtable_busy_state_clears_when_engine_raises(self):
        from tunapi.mattermost.loop import (
            _is_roundtable_busy,
            _run_cross_roundtable_engine,
        )

        cfg = _make_cfg(bot_username="kaixing")
        resolved = _ResolvedPrompt(text="roundtable prompt", file_context="")
        msg = _make_msg(
            text="前端视角 @kaixing",
            root_id="root1",
            sender_username="codeview",
        )

        with patch(
            "tunapi.mattermost.loop._run_engine",
            new_callable=AsyncMock,
            side_effect=RuntimeError("boom"),
        ):
            with pytest.raises(RuntimeError, match="boom"):
                await _run_cross_roundtable_engine(
                    resolved,
                    msg,
                    cfg,
                    {},
                    MagicMock(),
                    None,
                    AsyncMock(),
                )

        assert _is_roundtable_busy(cfg) is False


# ---------------------------------------------------------------------------
# _archive_roundtable tests
# ---------------------------------------------------------------------------


class TestArchiveRoundtable:
    @pytest.mark.anyio()
    async def test_archives_transcript_to_journal(self):
        session = RoundtableSession(
            thread_id="t1",
            channel_id="ch1",
            topic="design review",
            engines=["claude", "gemini"],
            total_rounds=2,
        )
        session.transcript.append(("claude", "Option A is better."))
        session.transcript.append(("gemini", "I prefer option B."))

        journal = AsyncMock()
        sent: list[RenderedMessage] = []
        send = AsyncMock(side_effect=lambda m: sent.append(m))

        await _archive_roundtable(session, journal, send)

        journal.append.assert_called_once()
        entry = journal.append.call_args[0][0]
        assert entry.event == "roundtable_closed"
        assert "design review" in entry.data["topic"]
        assert "Option A" in entry.data["transcript"]
        assert len(sent) == 1

    @pytest.mark.anyio()
    async def test_no_journal_still_sends_close_message(self):
        session = RoundtableSession(
            thread_id="t1",
            channel_id="ch1",
            topic="test",
            engines=["claude"],
            total_rounds=1,
        )
        sent: list[RenderedMessage] = []
        send = AsyncMock(side_effect=lambda m: sent.append(m))

        await _archive_roundtable(session, None, send)

        assert len(sent) == 1

    @pytest.mark.anyio()
    async def test_empty_transcript_skips_journal_write(self):
        session = RoundtableSession(
            thread_id="t1",
            channel_id="ch1",
            topic="test",
            engines=["claude"],
            total_rounds=1,
        )
        journal = AsyncMock()
        send = AsyncMock()

        await _archive_roundtable(session, journal, send)

        journal.append.assert_not_called()

    @pytest.mark.anyio()
    async def test_facade_called_when_project_provided(self):
        session = RoundtableSession(
            thread_id="t1",
            channel_id="ch1",
            topic="architecture",
            engines=["claude"],
            total_rounds=1,
        )
        session.transcript.append(("claude", "Some response"))

        journal = AsyncMock()
        facade = AsyncMock()
        send = AsyncMock()

        await _archive_roundtable(
            session, journal, send,
            facade=facade, project="myproj", branch="main",
        )

        facade.save_roundtable.assert_called_once()
        call_kwargs = facade.save_roundtable.call_args
        assert call_kwargs[0][1] == "myproj"

    @pytest.mark.anyio()
    async def test_facade_not_called_without_project(self):
        session = RoundtableSession(
            thread_id="t1",
            channel_id="ch1",
            topic="test",
            engines=["claude"],
            total_rounds=1,
        )
        session.transcript.append(("claude", "response"))

        facade = AsyncMock()
        send = AsyncMock()

        await _archive_roundtable(session, None, send, facade=facade)

        facade.save_roundtable.assert_not_called()

    @pytest.mark.anyio()
    async def test_transcript_truncated_to_500_chars(self):
        session = RoundtableSession(
            thread_id="t1",
            channel_id="ch1",
            topic="test",
            engines=["claude"],
            total_rounds=1,
        )
        long_answer = "x" * 1000
        session.transcript.append(("claude", long_answer))

        journal = AsyncMock()
        send = AsyncMock()

        await _archive_roundtable(session, journal, send)

        entry = journal.append.call_args[0][0]
        assert len(entry.data["transcript"]) < 600


# ---------------------------------------------------------------------------
# _send_startup tests
# ---------------------------------------------------------------------------


class TestSendStartup:
    @pytest.mark.anyio()
    async def test_sends_startup_message(self):
        cfg = _make_cfg(channel_id="ch1")

        await _send_startup(cfg)

        cfg.exec_cfg.transport.send.assert_called_once()
        call_kwargs = cfg.exec_cfg.transport.send.call_args[1]
        assert call_kwargs["channel_id"] == "ch1"
        assert call_kwargs["message"].text == "Bot started"


# ---------------------------------------------------------------------------
# MattermostIncomingMessage property tests (extra)
# ---------------------------------------------------------------------------


class TestMattermostIncomingMessageExtra:
    def test_is_direct_group(self):
        """G (group DM) is not considered direct."""
        msg = _make_msg(channel_type="G")
        assert msg.is_direct is False

    def test_is_thread_reply_with_root(self):
        msg = _make_msg(root_id="root123")
        assert msg.is_thread_reply is True

    def test_not_thread_reply_empty_root(self):
        msg = _make_msg(root_id="")
        assert msg.is_thread_reply is False

    def test_default_transport_value(self):
        msg = _make_msg()
        assert msg.transport == "mattermost"

    def test_file_ids_as_tuple(self):
        msg = _make_msg(file_ids=("f1", "f2"))
        assert len(msg.file_ids) == 2
