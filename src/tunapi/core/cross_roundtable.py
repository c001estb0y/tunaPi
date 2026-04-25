"""Cross-instance Mattermost roundtable state helpers.

The authoritative state for this feature is the Mattermost Thread history.
This module stays transport-neutral so the Mattermost loop can fetch posts and
then derive status, rounds, and prompts from plain dataclasses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

_METADATA_RE = re.compile(r"<!--\s*tunapi:roundtable\s+(\{.*?\})\s*-->", re.DOTALL)
_CONTROL_RE = re.compile(
    r"<!--\s*tunapi:roundtable-control\s+(\{.*?\})\s*-->",
    re.DOTALL,
)

ControlEvent = Literal["pause", "resume", "close"]


class CrossRTStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class CrossRTMetadata:
    topic: str
    participants: list[str]
    max_rounds: int
    version: int = 1


@dataclass(frozen=True, slots=True)
class ThreadPost:
    sender_username: str
    message: str
    created_at: int
    root_id: str = ""


@dataclass(frozen=True, slots=True)
class CrossRTState:
    metadata: CrossRTMetadata
    status: CrossRTStatus
    current_round: int
    next_participant: str | None


def _load_marker_json(pattern: re.Pattern[str], text: str) -> dict[str, Any] | None:
    match = pattern.search(text)
    if match is None:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def format_metadata_marker(
    *,
    topic: str,
    participants: list[str],
    max_rounds: int,
) -> str:
    payload = {
        "version": 1,
        "topic": topic,
        "participants": participants,
        "max_rounds": max_rounds,
    }
    return f"<!-- tunapi:roundtable {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))} -->"


def parse_metadata(text: str) -> CrossRTMetadata | None:
    payload = _load_marker_json(_METADATA_RE, text)
    if payload is None:
        return None

    topic = payload.get("topic")
    participants = payload.get("participants")
    max_rounds = payload.get("max_rounds")
    version = payload.get("version", 1)

    if not isinstance(topic, str) or not topic.strip():
        return None
    if (
        not isinstance(participants, list)
        or not participants
        or not all(isinstance(p, str) and p.strip() for p in participants)
    ):
        return None
    if not isinstance(max_rounds, int) or max_rounds < 1:
        return None
    if not isinstance(version, int):
        return None

    normalized_participants = [p.lstrip("@") for p in participants]
    if len(normalized_participants) < 2:
        return None
    if len(set(normalized_participants)) != len(normalized_participants):
        return None

    return CrossRTMetadata(
        topic=topic,
        participants=normalized_participants,
        max_rounds=max_rounds,
        version=version,
    )


def format_control_marker(event: str) -> str:
    payload = {"version": 1, "event": event}
    return f"<!-- tunapi:roundtable-control {json.dumps(payload, ensure_ascii=False, separators=(',', ':'))} -->"


def parse_control_event(text: str) -> ControlEvent | None:
    payload = _load_marker_json(_CONTROL_RE, text)
    if payload is None:
        return None
    event = payload.get("event")
    if event in {"pause", "resume", "close"}:
        return event
    return None


def latest_control_event(posts: list[ThreadPost]) -> ControlEvent | None:
    latest: ControlEvent | None = None
    for post in sorted(posts, key=lambda p: p.created_at):
        event = parse_control_event(post.message)
        if event == "close":
            return "close"
        if event is not None:
            latest = event
    return latest


def is_system_marker_post(text: str) -> bool:
    return parse_metadata(text) is not None or _CONTROL_RE.search(text) is not None


def _participant_turns(posts: list[ThreadPost], participants: list[str]) -> list[str]:
    participant_set = set(participants)
    turns: list[str] = []
    for post in sorted(posts, key=lambda p: p.created_at):
        if is_system_marker_post(post.message):
            continue
        sender = post.sender_username.lstrip("@")
        if sender in participant_set:
            turns.append(sender)
    return turns


def count_completed_rounds(posts: list[ThreadPost], participants: list[str]) -> int:
    if not participants:
        return 0

    completed = 0
    seen: set[str] = set()
    for sender in _participant_turns(posts, participants):
        if sender in seen:
            # A participant speaking twice starts a new partial round.
            seen = {sender}
            continue
        seen.add(sender)
        if len(seen) == len(participants):
            completed += 1
            seen.clear()
    return completed


def resolve_next_participant(
    posts: list[ThreadPost],
    participants: list[str],
) -> str | None:
    if not participants:
        return None

    seen: set[str] = set()
    for sender in _participant_turns(posts, participants):
        if sender in seen:
            seen = {sender}
        else:
            seen.add(sender)
        if len(seen) == len(participants):
            seen.clear()

    for participant in participants:
        if participant not in seen:
            return participant
    return participants[0]


def derive_state(metadata: CrossRTMetadata, posts: list[ThreadPost]) -> CrossRTState:
    control = latest_control_event(posts)
    completed_rounds = count_completed_rounds(posts, metadata.participants)

    if control == "close" or completed_rounds >= metadata.max_rounds:
        status = CrossRTStatus.CLOSED
        next_participant = None
    elif control == "pause":
        status = CrossRTStatus.PAUSED
        next_participant = resolve_next_participant(posts, metadata.participants)
    else:
        status = CrossRTStatus.ACTIVE
        next_participant = resolve_next_participant(posts, metadata.participants)

    return CrossRTState(
        metadata=metadata,
        status=status,
        current_round=completed_rounds,
        next_participant=next_participant,
    )


def build_agent_prompt(
    *,
    metadata: CrossRTMetadata,
    state: CrossRTState,
    posts: list[ThreadPost],
    current_bot: str,
) -> str:
    participants = ", ".join(metadata.participants)
    transcript_lines: list[str] = []
    for post in sorted(posts, key=lambda p: p.created_at):
        sender = post.sender_username.lstrip("@") or "unknown"
        event = parse_control_event(post.message)
        if event is not None:
            transcript_lines.append(f"[system]: {event}")
            continue
        transcript_lines.append(f"[{sender}]: {post.message}")

    current_name = current_bot.lstrip("@")
    if state.next_participant and state.next_participant != current_name:
        next_after_current = state.next_participant
    else:
        try:
            current_index = metadata.participants.index(current_name)
        except ValueError:
            next_after_current = state.next_participant
        else:
            next_after_current = metadata.participants[
                (current_index + 1) % len(metadata.participants)
            ]

    next_instruction = (
        f"如果需要继续讨论，请在结尾明确 @{next_after_current}；"
        "如果讨论已收敛或无新信息，不要 @ 下一位，直接总结结论。"
        if next_after_current and state.status == CrossRTStatus.ACTIVE
        else "如果讨论已收敛或无新信息，不要 @ 下一位，直接总结结论。"
    )

    transcript = "\n".join(transcript_lines) if transcript_lines else "(暂无历史发言)"
    return (
        "你正在参与一场 Mattermost multi-agent roundtable。\n\n"
        f"主题: {metadata.topic}\n"
        f"参与者: {participants}\n"
        f"当前 bot: {current_bot}\n"
        f"当前轮次: {state.current_round}/{metadata.max_rounds}\n\n"
        "Thread 历史:\n"
        f"{transcript}\n\n"
        "规则:\n"
        "1. 阅读 Thread 历史，包括其他 agent 和人类的发言。\n"
        "2. 发言保持简洁，默认不超过 300 字。\n"
        f"3. {next_instruction}\n"
        "4. 不要伪造控制命令；暂停、恢复、结束只由 !rt 控制。"
    )
