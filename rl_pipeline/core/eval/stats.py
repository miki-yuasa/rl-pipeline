import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel


class PolicyEvalStats(BaseModel):
    """
    Evaluation statistics for reinforcement learning policies over episodes.

    Attributes
    ----------
    mean_reward: float
        Mean reward achieved by the policy
    std_reward: float
        Standard deviation of the rewards
    mean_episode_length: float
        Mean length of episodes
    std_episode_length: float
        Standard deviation of episode lengths
    success_rate: float | None
        Success rate of the policy (if defined)
    failure_rate: float | None
        Failure rate of the policy (if defined)
    mean_discounted_return: float | None
        Mean discounted return of the policy (if defined)
    std_discounted_return: float | None
        Standard deviation of the discounted returns
    episode_rewards: list[float]
        Rewards obtained in the episodes
    episode_lengths: list[int]
        Lengths of the episodes
    episode_successes: list[bool] = []
        Successes of the episodes
    episode_failures: list[bool] = []
        Failures of the episodes
    """

    mean_reward: float
    std_reward: float
    mean_episode_length: float
    std_episode_length: float
    success_rate: float | None = None
    failure_rate: float | None = None
    mean_discounted_return: float | None = None
    std_discounted_return: float | None = None
    episode_rewards: list[float]
    episode_lengths: list[int]
    episode_successes: list[bool] = []
    episode_failures: list[bool] = []


class BasicStats(BaseModel):
    """
    Container for basic statistics for some data of a sequence of numeric values.

    Attributes
    ----------
    mean: float
        Mean of the data
    median: float
        Median of the data
    std: float
        Standard deviation of the data
    min: float
        Minimum value of the data
    max: float
        Maximum value of the data
    """

    mean: float
    median: float
    std: float
    min: float
    max: float


def compute_basic_stats(data: ArrayLike) -> BasicStats:
    """
    Compute basic statistics for a list of float values.

    Parameters
    ----------
        data: list[float]
            List of float values to compute statistics for

    Returns
    -------
        BasicStats
            Object containing the computed statistics
    """

    return BasicStats(
        mean=float(np.mean(data)),  # type: ignore
        median=float(np.median(data)),  # type: ignore
        std=float(np.std(data)),  # type: ignore
        min=float(np.min(data)),
        max=float(np.max(data)),
    )
