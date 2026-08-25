from core.audit import AuditState, _handle_rollback_conversation


def test_rollback_index_zero_clears_combined_messages_without_system_message():
    state = AuditState(
        target_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        combined_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )

    _handle_rollback_conversation(state, "rollback-1", {"index": 0})

    assert state.target_messages == []
    assert state.combined_messages == []


def test_rollback_index_zero_preserves_combined_system_message():
    system_message = {"role": "system", "content": "Be helpful"}
    state = AuditState(
        target_system="Be helpful",
        target_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
        combined_messages=[
            system_message,
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ],
    )

    _handle_rollback_conversation(state, "rollback-2", {"index": 0})

    assert state.target_messages == []
    assert state.combined_messages == [system_message]
