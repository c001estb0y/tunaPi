"""Mattermost loop tests for multi-agent roundtable routing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tunapi.mattermost.api_models import Post, PostList, User
from tunapi.mattermost.loop import _try_dispatch_command, _try_dispatch_cross_roundtable
from tunapi.mattermost.types import MattermostIncomingMessage
from tunapi.transport import MessageRef, RenderedMessage


def _make_msg(
    text: str,
    *,
    root_id: str = "",
    sender_id: str = "user-in",
    sender_username: str = "minusjiang",
) -> MattermostIncomingMessage:
    return MattermostIncomingMessage(
        channel_id="ch1",
        post_id="post-in",
        text=text,
        root_id=root_id,
        sender_id=sender_id,
        sender_username=sender_username,
        channel_type="O",
    )


def _make_cfg(*, bot_username: str = "kaixing") -> MagicMock:
    cfg = MagicMock()
    cfg.bot_username = bot_username
    cfg.bot_user_id = "bot-user"
    cfg.cross_roundtable_enabled = True
    cfg.cross_roundtable_max_rounds = 3
    cfg.cross_roundtable_timeout_minutes = 5
    cfg.session_mode = "chat"
    cfg.runtime = MagicMock()
    cfg.exec_cfg = MagicMock()
    cfg.exec_cfg.transport = MagicMock()
    cfg.exec_cfg.transport.send = AsyncMock(
        return_value=MessageRef(channel_id="ch1", message_id="root1")
    )
    cfg.bot = MagicMock()
    cfg.bot._client = MagicMock()
    return cfg


async def _send_capture(message: RenderedMessage) -> None:
    return None


@pytest.mark.anyio
async def test_rt_start_sends_root_and_thread_kickoff():
    cfg = _make_cfg()
    msg = _make_msg("!rt start @kaixing @agent2 分析架构")

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock):
        result = await _try_dispatch_command(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    assert cfg.exec_cfg.transport.send.await_count == 2
    root_call = cfg.exec_cfg.transport.send.await_args_list[0].kwargs
    kickoff_call = cfg.exec_cfg.transport.send.await_args_list[1].kwargs
    assert "tunapi:roundtable" in root_call["message"].text
    assert "分析架构" in root_call["message"].text
    assert "@kaixing" in kickoff_call["message"].text
    assert kickoff_call["options"].thread_id == "root1"


@pytest.mark.anyio
async def test_rt_start_is_ignored_by_non_owner_bot_instance():
    cfg = _make_cfg(bot_username="agent2")
    msg = _make_msg("!rt start @kaixing @agent2 分析架构")

    result = await _try_dispatch_command(
        msg,
        cfg,
        {},
        MagicMock(),
        None,
        None,
        AsyncMock(side_effect=_send_capture),
    )

    assert result is True
    cfg.exec_cfg.transport.send.assert_not_awaited()


@pytest.mark.anyio
async def test_rt_start_directly_runs_owner_first_participant():
    cfg = _make_cfg(bot_username="kaixing")
    msg = _make_msg("!rt start @kaixing @agent2 分析架构")

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
        result = await _try_dispatch_command(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    run.assert_awaited_once()
    resolved_prompt = run.await_args.args[0]
    run_cfg = run.await_args.args[2]
    assert "分析架构" in resolved_prompt.text
    assert "@agent2" in resolved_prompt.text
    assert run_cfg.session_mode == "stateless"


@pytest.mark.anyio
async def test_cross_roundtable_thread_mention_runs_expected_bot():
    cfg = _make_cfg(bot_username="agent2")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1", "p1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                ),
                "p1": Post(
                    id="p1",
                    channel_id="ch1",
                    user_id="u-kaixing",
                    root_id="root1",
                    message="前端视角 @agent2",
                    create_at=1,
                ),
            },
        )
    )
    cfg.bot.get_user = AsyncMock(
        side_effect=lambda user_id: User(
            id=user_id,
            username={"u-human": "minusjiang", "u-kaixing": "kaixing"}[user_id],
        )
    )
    msg = _make_msg("前端视角 @agent2", root_id="root1", sender_username="kaixing")

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
        result = await _try_dispatch_cross_roundtable(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    run.assert_awaited_once()
    resolved_prompt = run.await_args.args[0]
    run_cfg = run.await_args.args[2]
    assert "分析架构" in resolved_prompt.text
    assert "@kaixing" in resolved_prompt.text
    assert run_cfg.session_mode == "stateless"


@pytest.mark.anyio
async def test_cross_roundtable_mention_requires_username_boundary():
    cfg = _make_cfg(bot_username="agent2")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1", "p1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                ),
                "p1": Post(
                    id="p1",
                    channel_id="ch1",
                    user_id="u-kaixing",
                    root_id="root1",
                    message="前端视角 @agent20",
                    create_at=1,
                ),
            },
        )
    )
    cfg.bot.get_user = AsyncMock(
        side_effect=lambda user_id: User(
            id=user_id,
            username={"u-human": "minusjiang", "u-kaixing": "kaixing"}[user_id],
        )
    )
    msg = _make_msg("前端视角 @agent20", root_id="root1", sender_username="kaixing")

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
        result = await _try_dispatch_cross_roundtable(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    run.assert_not_awaited()


@pytest.mark.anyio
async def test_cross_roundtable_ignores_out_of_order_mention():
    cfg = _make_cfg(bot_username="agent2")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                )
            },
        )
    )
    cfg.bot.get_user = AsyncMock(return_value=User(id="u-human", username="minusjiang"))
    msg = _make_msg("乱序 @agent2", root_id="root1", sender_username="minusjiang")

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
        result = await _try_dispatch_cross_roundtable(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    run.assert_not_awaited()


@pytest.mark.anyio
async def test_cross_roundtable_human_mention_does_not_run_expected_bot():
    cfg = _make_cfg(bot_username="agent2")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1", "p1", "human1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                ),
                "p1": Post(
                    id="p1",
                    channel_id="ch1",
                    user_id="u-kaixing",
                    root_id="root1",
                    message="前端视角 @agent2",
                    create_at=1,
                ),
                "human1": Post(
                    id="human1",
                    channel_id="ch1",
                    user_id="u-human",
                    root_id="root1",
                    message="@agent2 补充一下鉴权风险",
                    create_at=2,
                ),
            },
        )
    )
    cfg.bot.get_user = AsyncMock(
        side_effect=lambda user_id: User(
            id=user_id,
            username={"u-human": "minusjiang", "u-kaixing": "kaixing"}[user_id],
        )
    )
    msg = _make_msg(
        "@agent2 补充一下鉴权风险",
        root_id="root1",
        sender_username="minusjiang",
    )

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
        result = await _try_dispatch_cross_roundtable(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    run.assert_not_awaited()


@pytest.mark.anyio
async def test_cross_roundtable_stop_writes_pause_control_marker():
    cfg = _make_cfg(bot_username="kaixing")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                )
            },
        )
    )
    cfg.bot.get_user = AsyncMock(return_value=User(id="u-human", username="minusjiang"))
    msg = _make_msg("!rt stop", root_id="root1", sender_username="minusjiang")

    result = await _try_dispatch_cross_roundtable(
        msg,
        cfg,
        {},
        MagicMock(),
        None,
        AsyncMock(side_effect=_send_capture),
    )

    assert result is True
    sent = cfg.exec_cfg.transport.send.await_args.kwargs
    assert "tunapi:roundtable-control" in sent["message"].text
    assert '"event":"pause"' in sent["message"].text
    assert sent["options"].thread_id == "root1"


@pytest.mark.anyio
async def test_cross_roundtable_control_is_ignored_by_non_owner_bot_instance():
    cfg = _make_cfg(bot_username="agent2")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                )
            },
        )
    )
    cfg.bot.get_user = AsyncMock(return_value=User(id="u-human", username="minusjiang"))
    msg = _make_msg("!rt stop", root_id="root1", sender_username="minusjiang")

    result = await _try_dispatch_cross_roundtable(
        msg,
        cfg,
        {},
        MagicMock(),
        None,
        AsyncMock(side_effect=_send_capture),
    )

    assert result is True
    cfg.exec_cfg.transport.send.assert_not_awaited()


@pytest.mark.anyio
async def test_cross_roundtable_resume_mentions_next_participant():
    cfg = _make_cfg(bot_username="kaixing")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1", "p1", "pause1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                ),
                "p1": Post(
                    id="p1",
                    channel_id="ch1",
                    user_id="u-kaixing",
                    root_id="root1",
                    message="前端视角 @agent2",
                    create_at=1,
                ),
                "pause1": Post(
                    id="pause1",
                    channel_id="ch1",
                    user_id="u-human",
                    root_id="root1",
                    message='<!-- tunapi:roundtable-control {"version":1,"event":"pause"} -->',
                    create_at=2,
                ),
            },
        )
    )
    cfg.bot.get_user = AsyncMock(
        side_effect=lambda user_id: User(
            id=user_id,
            username={"u-human": "minusjiang", "u-kaixing": "kaixing"}[user_id],
        )
    )
    msg = _make_msg("!rt resume", root_id="root1", sender_username="minusjiang")

    result = await _try_dispatch_cross_roundtable(
        msg,
        cfg,
        {},
        MagicMock(),
        None,
        AsyncMock(side_effect=_send_capture),
    )

    assert result is True
    sent = cfg.exec_cfg.transport.send.await_args.kwargs
    assert '"event":"resume"' in sent["message"].text
    assert "@agent2" in sent["message"].text
    assert sent["options"].thread_id == "root1"


@pytest.mark.anyio
async def test_cross_roundtable_resume_active_thread_does_not_dispatch():
    cfg = _make_cfg(bot_username="kaixing")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                )
            },
        )
    )
    cfg.bot.get_user = AsyncMock(return_value=User(id="u-human", username="minusjiang"))
    msg = _make_msg("!rt resume", root_id="root1", sender_username="minusjiang")

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
        result = await _try_dispatch_cross_roundtable(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    run.assert_not_awaited()
    sent = cfg.exec_cfg.transport.send.await_args.kwargs
    assert "not paused" in sent["message"].text
    assert "tunapi:roundtable-control" not in sent["message"].text


@pytest.mark.anyio
async def test_cross_roundtable_paused_thread_does_not_run_engine():
    cfg = _make_cfg(bot_username="agent2")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1", "pause1", "p1"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->',
                ),
                "pause1": Post(
                    id="pause1",
                    channel_id="ch1",
                    user_id="u-human",
                    root_id="root1",
                    message='<!-- tunapi:roundtable-control {"version":1,"event":"pause"} -->',
                    create_at=1,
                ),
                "p1": Post(
                    id="p1",
                    channel_id="ch1",
                    user_id="u-kaixing",
                    root_id="root1",
                    message="前端视角 @agent2",
                    create_at=2,
                ),
            },
        )
    )
    cfg.bot.get_user = AsyncMock(
        side_effect=lambda user_id: User(
            id=user_id,
            username={"u-human": "minusjiang", "u-kaixing": "kaixing"}[user_id],
        )
    )
    msg = _make_msg("前端视角 @agent2", root_id="root1", sender_username="kaixing")

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
        result = await _try_dispatch_cross_roundtable(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    run.assert_not_awaited()


@pytest.mark.anyio
async def test_closed_roundtable_human_mention_is_released_to_normal_chat():
    cfg = _make_cfg(bot_username="kaixing")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1", "p1", "p2"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","codeview"],"max_rounds":1} -->',
                ),
                "p1": Post(
                    id="p1",
                    channel_id="ch1",
                    user_id="u-kaixing",
                    root_id="root1",
                    message="前端视角 @codeview",
                    create_at=1,
                ),
                "p2": Post(
                    id="p2",
                    channel_id="ch1",
                    user_id="u-codeview",
                    root_id="root1",
                    message="后端视角",
                    create_at=2,
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
                "user-in": "minusjiang",
            }[user_id],
            is_bot=user_id in {"u-kaixing", "u-codeview"},
        )
    )
    msg = _make_msg(
        "@kaixing 总结一下上述讨论",
        root_id="root1",
        sender_id="user-in",
        sender_username="minusjiang",
    )

    result = await _try_dispatch_cross_roundtable(
        msg,
        cfg,
        {},
        MagicMock(),
        None,
        AsyncMock(side_effect=_send_capture),
    )

    assert result is False


@pytest.mark.anyio
async def test_closed_roundtable_bot_mention_is_ignored():
    cfg = _make_cfg(bot_username="kaixing")
    cfg.bot._client.get_thread = AsyncMock(
        return_value=PostList(
            order=["root1", "p1", "p2"],
            posts={
                "root1": Post(
                    id="root1",
                    channel_id="ch1",
                    user_id="u-human",
                    message='<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","codeview"],"max_rounds":1} -->',
                ),
                "p1": Post(
                    id="p1",
                    channel_id="ch1",
                    user_id="u-kaixing",
                    root_id="root1",
                    message="前端视角 @codeview",
                    create_at=1,
                ),
                "p2": Post(
                    id="p2",
                    channel_id="ch1",
                    user_id="u-codeview",
                    root_id="root1",
                    message="后端视角 @kaixing",
                    create_at=2,
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
                "user-in": "codeview",
            }[user_id],
            is_bot=user_id in {"u-kaixing", "u-codeview", "user-in"},
        )
    )
    msg = _make_msg(
        "后端视角 @kaixing",
        root_id="root1",
        sender_id="user-in",
        sender_username="codeview",
    )

    with patch("tunapi.mattermost.loop._run_engine", new_callable=AsyncMock) as run:
        result = await _try_dispatch_cross_roundtable(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            AsyncMock(side_effect=_send_capture),
        )

    assert result is True
    run.assert_not_awaited()


@pytest.mark.anyio
async def test_main_channel_cross_rt_status_does_not_start_old_roundtable():
    cfg = _make_cfg(bot_username="kaixing")
    msg = _make_msg("!rt status")
    send = AsyncMock(side_effect=_send_capture)

    with patch("tunapi.mattermost.loop.handle_rt", new_callable=AsyncMock) as handle_rt:
        result = await _try_dispatch_command(
            msg,
            cfg,
            {},
            MagicMock(),
            None,
            None,
            send,
        )

    assert result is True
    handle_rt.assert_not_awaited()
    send.assert_awaited_once()
