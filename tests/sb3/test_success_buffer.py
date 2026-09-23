from __future__ import annotations

from absl.testing import absltest

from rl_pipeline.sb3.utils.eval import SuccessBuffer


class SuccessBufferTest(absltest.TestCase):
    def test_computes_goal_wise_success_rate(self) -> None:
        success_buffer = SuccessBuffer()

        # Goal A: 1/2 success
        success_buffer._log_success_callback(
            {"info": {"is_success": True, "goal_name": "goal_a"}, "done": True},
            {},
        )
        success_buffer._log_success_callback(
            {
                "info": {"is_success": False, "goal_name": "goal_a"},
                "done": True,
            },
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

        self.assertEqual(result.success_rate, 0.8)
        self.assertEqual(
            result.goal_success_rates, {"goal_a": 0.5, "goal_b": 1.0}
        )

    def test_clears_goal_wise_stats_after_post_eval(self) -> None:
        success_buffer = SuccessBuffer()

        success_buffer._log_success_callback(
            {"info": {"is_success": True, "goal_name": "goal_a"}, "done": True},
            {},
        )

        first_result = success_buffer.post_eval()
        second_result = success_buffer.post_eval()

        self.assertEqual(first_result.goal_success_rates, {"goal_a": 1.0})
        self.assertEqual(second_result.goal_success_rates, {})


if __name__ == "__main__":
    absltest.main()
