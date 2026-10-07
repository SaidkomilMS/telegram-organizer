"""telegram/links.py: permalinks for public, private and basic-group chats."""

from tg_curator.telegram.links import permalink


def test_public_channel_uses_username() -> None:
    assert permalink(-1001234567890, "kunuz", 42) == "https://t.me/kunuz/42"


def test_username_with_at_sign_is_stripped() -> None:
    assert permalink(-1001234567890, "@kunuz", 42) == "https://t.me/kunuz/42"


def test_private_channel_uses_internal_id() -> None:
    assert permalink(-1001234567890, None, 7) == "https://t.me/c/1234567890/7"


def test_private_supergroup_uses_internal_id() -> None:
    assert permalink(-1009876543210, "", 15) == "https://t.me/c/9876543210/15"


def test_basic_group_has_no_link() -> None:
    assert permalink(-123456789, None, 3) is None


def test_bare_positive_channel_id_is_accepted() -> None:
    # defensive: a bare (unmarked) id still produces the private-channel form
    assert permalink(1234567890, None, 1) == "https://t.me/c/1234567890/1"
