"""Trigger mode: control when the bot responds in channels."""

from __future__ import annotations

import re

from ..core.trigger import resolve_trigger_mode as _core_resolve
from .chat_prefs import ChatPrefsStore
from .types import MattermostIncomingMessage


async def resolve_trigger_mode(
    channel_id: str,
    chat_prefs: ChatPrefsStore | None,
    *,
    default: str = "all",
) -> str:
    """Return 'all' or 'mentions' for a channel."""
    return await _core_resolve(channel_id, chat_prefs, default=default)


def should_trigger(
    msg: MattermostIncomingMessage,
    *,
    bot_username: str,
    trigger_mode: str,
) -> bool:
    """Decide whether to process this message based on trigger mode.

    - DMs always trigger.
    - In 'all' mode, every message triggers.
    - In 'mentions' mode, only messages containing @bot_username trigger.
    """
    if msg.is_direct:
        return True
    if trigger_mode == "all":
        return True
    if trigger_mode == "mentions":
        if not bot_username:
            return True
        mention = f"@{bot_username}"
        return mention.lower() in msg.text.lower()
    return True


def strip_mention(text: str, bot_username: str) -> str:
    """Remove @bot_username from message text."""
    if not bot_username:
        return text
    mention = f"@{bot_username}"
    return re.sub(re.escape(mention), "", text, flags=re.IGNORECASE).strip()
