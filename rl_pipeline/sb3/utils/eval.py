from typing import Any

import numpy as np
from pydantic import BaseModel


class SuccessBufferEval(BaseModel):
    episode_successes: list[bool] = []
    success_rate: float | None = None
    episode_failures: list[bool] = []
    failure_rate: float | None = None


class SuccessBuffer:
    """
    Buffer for success/failure metrics from SB3 ``evaluate_policy``.

    Usage
    -----
    This class is used to track the success and failure rates of episodes during evaluation.
    It can be used to compute the success and failure rates after evaluation is complete.

    Example
    -------
    .. code-block:: python

        success_buffer = SuccessBuffer()

        episode_rewards, episode_lengths = evaluate_policy(
            model,
            return_episode_rewards=True,
            callback=success_buffer._log_success_callback,
        )
        success_buffer_result: SuccessBufferEval = success_buffer.post_eval()
    """

    def __init__(self):
        self._is_success_buffer: list[bool] = []
        self._is_failure_buffer: list[bool] = []

    def _log_success_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.

        Parameters
        ----------
        locals_: dict[str, Any]
            Local variables of the callback function.
        globals_: dict[str, Any]
            Global variables of the callback function.
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success: bool | None = info.get("is_success")
            maybe_is_failure: bool | None = info.get("is_failure")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)
            if maybe_is_failure is not None:
                self._is_failure_buffer.append(maybe_is_failure)

    def post_eval(self) -> SuccessBufferEval:
        """
        Return evaluation success/failure metrics and reset the buffer.

        Returns
        -------
        result : SuccessBufferEval
            Evaluation results with:

            - ``episode_successes``: Per-episode success flags.
            - ``success_rate``: Mean success rate, or ``None`` when unavailable.
            - ``episode_failures``: Per-episode failure flags.
            - ``failure_rate``: Mean failure rate, or ``None`` when unavailable.
        """
        episode_successes: list[bool] = []
        episode_failures: list[bool] = []
        success_rate: float | None = None
        failure_rate: float | None = None

        if self._is_success_buffer:
            episode_successes: list[bool] = np.array(
                self._is_success_buffer, dtype=bool
            ).tolist()
            success_rate = float(np.mean(episode_successes))

        if self._is_failure_buffer:
            episode_failures: list[bool] = np.array(
                self._is_failure_buffer, dtype=bool
            ).tolist()
            failure_rate = float(np.mean(episode_failures))

        # Reset the buffer
        self._is_success_buffer.clear()
        self._is_failure_buffer.clear()

        result = SuccessBufferEval(
            episode_successes=episode_successes,
            success_rate=success_rate,
            episode_failures=episode_failures,
            failure_rate=failure_rate,
        )

        return result
