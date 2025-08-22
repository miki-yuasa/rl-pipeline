from typing import Any

from gymnasium import Wrapper
from pydantic import BaseModel, ConfigDict

from rl_pipeline.core.config import ConfigReader
from rl_pipeline.core.utils.io import get_class


class MakeEnvConfig(BaseModel):
    id: str = "multigrid-rooms-v0"
    max_episode_steps: int | None
    disable_env_checker: bool | None = None
    render_mode: str = "rgb_array"
    env_kwargs: dict[str, Any] = {}

    # allow arbitrary kwargs
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def make_env_args(self) -> dict[str, Any]:
        """Create the environment arguments."""
        return {
            "id": self.id,
            "max_episode_steps": self.max_episode_steps,
            "disable_env_checker": self.disable_env_checker,
            "render_mode": self.render_mode,
            **self.env_kwargs,
        }


class WrapperConfig(BaseModel):
    wrapper_class: type[Wrapper]
    wrapper_kwargs: dict[str, Any] = {}


class WrapperConfigReader(BaseModel, ConfigReader[WrapperConfig]):
    wrapper_class: str
    wrapper_kwargs: dict[str, Any] = {}

    def to_config(self) -> WrapperConfig:
        wrapper_class = get_class(self.wrapper_class)
        assert wrapper_class is not None, (
            f"Could not find wrapper class for {self.wrapper_class}"
        )
        return WrapperConfig(
            wrapper_class=wrapper_class,
            wrapper_kwargs=self.wrapper_kwargs,
        )


class GymEnvConfig(BaseModel):
    make_env_config: MakeEnvConfig
    wrapper_config: WrapperConfig | None = None
