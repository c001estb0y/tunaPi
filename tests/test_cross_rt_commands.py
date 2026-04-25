"""Command parsing tests for multi-agent roundtable commands."""

from __future__ import annotations

from tunapi.mattermost.commands import parse_cross_rt_start
from tunapi.mattermost.roundtable import parse_rt_args
from tunapi.transport_runtime import RoundtableConfig


def test_parse_start_two_bots_with_chinese_topic():
    participants, topic, error = parse_cross_rt_start(
        "@kaixing @agent2 分析 codebuddy-mem 架构"
    )

    assert error is None
    assert participants == ["kaixing", "agent2"]
    assert topic == "分析 codebuddy-mem 架构"


def test_parse_start_three_bots():
    participants, topic, error = parse_cross_rt_start("@a @b @c discuss architecture")

    assert error is None
    assert participants == ["a", "b", "c"]
    assert topic == "discuss architecture"


def test_parse_start_requires_two_bots():
    participants, topic, error = parse_cross_rt_start("@kaixing analyze code")

    assert participants == []
    assert topic == ""
    assert error is not None
    assert "at least 2" in error


def test_parse_start_requires_topic():
    participants, topic, error = parse_cross_rt_start("@kaixing @agent2")

    assert participants == ["kaixing", "agent2"]
    assert topic == ""
    assert error is not None
    assert "Topic is required" in error


def test_parse_start_rejects_duplicate_participants():
    participants, topic, error = parse_cross_rt_start("@kaixing @kaixing analyze")

    assert participants == []
    assert topic == ""
    assert error is not None
    assert "Duplicate" in error


def test_parse_start_stops_collecting_mentions_after_topic_starts():
    participants, topic, error = parse_cross_rt_start("@kaixing @agent2 compare @ops")

    assert error is None
    assert participants == ["kaixing", "agent2"]
    assert topic == "compare @ops"


def test_existing_single_instance_rt_parser_still_accepts_topic():
    topic, rounds, error = parse_rt_args(
        '"single instance topic" --rounds 2',
        RoundtableConfig(engines=(), rounds=1, max_rounds=3),
    )

    assert error is None
    assert topic == "single instance topic"
    assert rounds == 2
