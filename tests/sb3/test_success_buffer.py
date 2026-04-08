from rl_pipeline.sb3.utils.eval import SuccessBuffer


def test_success_buffer_computes_goal_wise_success_rate() -> None:
    success_buffer = SuccessBuffer()

    # Goal A: 1/2 success
    success_buffer._log_success_callback(
        {"info": {"is_success": True, "goal_name": "goal_a"}, "done": True},
        {},
    )
    success_buffer._log_success_callback(
        {"info": {"is_success": False, "goal_name": "goal_a"}, "done": True},
        {},
    )

    # Goal B: 2/2 success
    success_buffer._log_success_callback(
        {"info": {"is_success": True, "goal_name": "goal_b"}, "done": True},
        {},
    )
    success_buffer._log_success_callback(
        {"info": {"is_success": True, "goal_name": "goal_b"}, "done": True},
        {},
    )

    # No goal_name should not affect goal-wise stats
    success_buffer._log_success_callback(
        {"info": {"is_success": True}, "done": True},
        {},
    )

    result = success_buffer.post_eval()

    assert result.success_rate == 0.8
    assert result.goal_success_rates == {"goal_a": 0.5, "goal_b": 1.0}


def test_success_buffer_clears_goal_wise_stats_after_post_eval() -> None:
    success_buffer = SuccessBuffer()

    success_buffer._log_success_callback(
        {"info": {"is_success": True, "goal_name": "goal_a"}, "done": True},
        {},
    )

    first_result = success_buffer.post_eval()
    second_result = success_buffer.post_eval()

    assert first_result.goal_success_rates == {"goal_a": 1.0}
    assert second_result.goal_success_rates == {}
