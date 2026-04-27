"""Pure-logic tests for cross-instance Mattermost roundtables."""

from __future__ import annotations

from tunapi.core.cross_roundtable import (
    CrossRTMetadata,
    CrossRTState,
    CrossRTStatus,
    ThreadPost,
    build_agent_prompt,
    build_thread_context_prompt,
    derive_state,
    format_control_marker,
    format_metadata_marker,
    meaningful_thread_posts,
    parse_metadata,
)


def _post(
    sender: str,
    message: str,
    *,
    created_at: int,
    root_id: str = "root1",
) -> ThreadPost:
    return ThreadPost(
        sender_username=sender,
        message=message,
        created_at=created_at,
        root_id=root_id,
    )


def test_parse_metadata_from_system_block():
    text = """🎯 圆桌会议已开启

<!-- tunapi:roundtable {"version":1,"topic":"分析架构","participants":["kaixing","agent2"],"max_rounds":3} -->
"""

    metadata = parse_metadata(text)

    assert metadata is not None
    assert metadata.topic == "分析架构"
    assert metadata.participants == ["kaixing", "agent2"]
    assert metadata.max_rounds == 3


def test_format_metadata_round_trips():
    marker = format_metadata_marker(
        topic="分析 codebuddy-mem",
        participants=["kaixing", "agent2"],
        max_rounds=2,
    )

    metadata = parse_metadata(marker)

    assert metadata is not None
    assert metadata.topic == "分析 codebuddy-mem"
    assert metadata.participants == ["kaixing", "agent2"]
    assert metadata.max_rounds == 2


def test_parse_metadata_rejects_single_or_duplicate_participants():
    single = parse_metadata(
        format_metadata_marker(
            topic="架构讨论",
            participants=["kaixing"],
            max_rounds=3,
        )
    )
    duplicate = parse_metadata(
        format_metadata_marker(
            topic="架构讨论",
            participants=["kaixing", "@kaixing"],
            max_rounds=3,
        )
    )

    assert single is None
    assert duplicate is None


def test_derive_state_defaults_to_active_after_start():
    metadata = parse_metadata(
        format_metadata_marker(
            topic="架构讨论",
            participants=["kaixing", "agent2"],
            max_rounds=3,
        )
    )
    assert metadata is not None

    state = derive_state(metadata, [])

    assert state.status == CrossRTStatus.ACTIVE
    assert state.current_round == 0
    assert state.next_participant == "kaixing"


def test_derive_state_uses_latest_pause_and_resume_control_event():
    metadata = parse_metadata(
        format_metadata_marker(
            topic="架构讨论",
            participants=["kaixing", "agent2"],
            max_rounds=3,
        )
    )
    assert metadata is not None
    paused_posts = [
        _post("minusjiang", format_control_marker("pause"), created_at=1),
    ]
    resumed_posts = [
        *paused_posts,
        _post("minusjiang", format_control_marker("resume"), created_at=2),
    ]

    assert derive_state(metadata, paused_posts).status == CrossRTStatus.PAUSED
    assert derive_state(metadata, resumed_posts).status == CrossRTStatus.ACTIVE


def test_derive_state_close_beats_later_resume():
    metadata = parse_metadata(
        format_metadata_marker(
            topic="架构讨论",
            participants=["kaixing", "agent2"],
            max_rounds=3,
        )
    )
    assert metadata is not None
    posts = [
        _post("minusjiang", format_control_marker("close"), created_at=1),
        _post("minusjiang", format_control_marker("resume"), created_at=2),
    ]

    state = derive_state(metadata, posts)

    assert state.status == CrossRTStatus.CLOSED


def test_derive_state_counts_completed_rounds_from_participants_only():
    metadata = parse_metadata(
        format_metadata_marker(
            topic="架构讨论",
            participants=["kaixing", "agent2"],
            max_rounds=3,
        )
    )
    assert metadata is not None
    posts = [
        _post("kaixing", "前端视角 @agent2", created_at=1),
        _post("minusjiang", "别忘了鉴权", created_at=2),
        _post("agent2", "后端视角 @kaixing", created_at=3),
        _post("kaixing", "补充一点 @agent2", created_at=4),
    ]

    state = derive_state(metadata, posts)

    assert state.status == CrossRTStatus.ACTIVE
    assert state.current_round == 1
    assert state.next_participant == "agent2"


def test_derive_state_ignores_system_marker_posts_from_participants():
    metadata = parse_metadata(
        format_metadata_marker(
            topic="架构讨论",
            participants=["kaixing", "agent2"],
            max_rounds=3,
        )
    )
    assert metadata is not None
    posts = [
        _post(
            "kaixing",
            format_metadata_marker(
                topic="架构讨论",
                participants=["kaixing", "agent2"],
                max_rounds=3,
            ),
            created_at=0,
        ),
        _post(
            "kaixing",
            '<!-- tunapi:roundtable-control {"version":1,"event":"kickoff"} --> @kaixing 请开始',
            created_at=1,
        ),
        _post("agent2", "真实发言 @kaixing", created_at=2),
    ]

    state = derive_state(metadata, posts)

    assert state.current_round == 0
    assert state.next_participant == "kaixing"


def test_derive_state_closes_when_max_rounds_reached():
    metadata = parse_metadata(
        format_metadata_marker(
            topic="架构讨论",
            participants=["kaixing", "agent2"],
            max_rounds=1,
        )
    )
    assert metadata is not None
    posts = [
        _post("kaixing", "前端视角 @agent2", created_at=1),
        _post("agent2", "后端视角", created_at=2),
    ]

    state = derive_state(metadata, posts)

    assert state.status == CrossRTStatus.CLOSED
    assert state.current_round == 1
    assert state.next_participant is None


def test_build_agent_prompt_includes_human_context_and_next_mention():
    metadata = parse_metadata(
        format_metadata_marker(
            topic="分析 codebuddy-mem",
            participants=["kaixing", "agent2"],
            max_rounds=3,
        )
    )
    assert metadata is not None
    posts = [
        _post("kaixing", "我先看前端入口 @agent2", created_at=1),
        _post("minusjiang", "别忘了考虑安全性", created_at=2),
    ]
    state = derive_state(metadata, posts)

    prompt = build_agent_prompt(
        metadata=metadata,
        state=state,
        posts=posts,
        current_bot="agent2",
    )

    assert "分析 codebuddy-mem" in prompt
    assert "kaixing" in prompt
    assert "agent2" in prompt
    assert "别忘了考虑安全性" in prompt
    assert "@kaixing" in prompt
    assert "不要 @" in prompt
    assert "不要伪造控制命令" in prompt


def test_build_agent_prompt_uses_expected_speaker_for_out_of_order_mention():
    metadata = parse_metadata(
        format_metadata_marker(
            topic="三方讨论",
            participants=["a", "b", "c"],
            max_rounds=3,
        )
    )
    assert metadata is not None
    posts = [
        _post("a", "a 先说", created_at=1),
    ]
    state = derive_state(metadata, posts)
    assert state.next_participant == "b"

    prompt = build_agent_prompt(
        metadata=metadata,
        state=state,
        posts=posts,
        current_bot="c",
    )

    assert "@b" in prompt


def test_meaningful_thread_posts_filters_roundtable_system_noise():
    posts = [
        ThreadPost(
            sender_username="minusjiang",
            message='<!-- tunapi:roundtable {"version":1,"topic":"计算器","participants":["kaixing","codeview"],"max_rounds":1} -->',
            created_at=1,
            root_id="root1",
        ),
        ThreadPost(
            sender_username="kaixing",
            message="working · codex",
            created_at=2,
            root_id="root1",
        ),
        ThreadPost(
            sender_username="kaixing",
            message="可以先实现加减乘除 @codeview",
            created_at=3,
            root_id="root1",
        ),
        ThreadPost(
            sender_username="codeview",
            message="后端建议补充输入校验",
            created_at=4,
            root_id="root1",
        ),
    ]

    result = meaningful_thread_posts(posts)

    assert [post.message for post in result] == [
        "可以先实现加减乘除 @codeview",
        "后端建议补充输入校验",
    ]


def test_build_thread_context_prompt_includes_roundtable_summary_fields():
    metadata = CrossRTMetadata(
        topic="讨论计算器实现",
        participants=["kaixing", "codeview"],
        max_rounds=1,
    )
    state = CrossRTState(
        metadata=metadata,
        status=CrossRTStatus.CLOSED,
        current_round=1,
        next_participant=None,
    )
    posts = [
        ThreadPost(
            sender_username="minusjiang",
            message="讨论计算器实现",
            created_at=1,
            root_id="root1",
        ),
        ThreadPost(
            sender_username="kaixing",
            message="前端用一个表单和结果区 @codeview",
            created_at=2,
            root_id="root1",
        ),
        ThreadPost(
            sender_username="codeview",
            message="后端只需要纯函数和单元测试",
            created_at=3,
            root_id="root1",
        ),
    ]

    prompt = build_thread_context_prompt(
        posts=posts,
        current_request="总结一下上述讨论",
        metadata=metadata,
        state=state,
        max_posts=20,
        max_chars=12_000,
    )

    assert "Topic: 讨论计算器实现" in prompt
    assert "Participants: kaixing, codeview" in prompt
    assert "Status: closed" in prompt
    assert "[kaixing]: 前端用一个表单和结果区 @codeview" in prompt
    assert "[codeview]: 后端只需要纯函数和单元测试" in prompt
    assert "[Current request]\n总结一下上述讨论" in prompt
