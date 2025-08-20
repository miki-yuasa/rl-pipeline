from typing import Any

from gymnasium import Wrapper
from pydantic import BaseModel, ConfigDict, SerializationInfo, model_serializer


class MakeEnvConfig(BaseModel):
    id: str = "multigrid-rooms-v0"
    max_episode_steps: int | None
    disable_env_checker: bool | None = None
    env_kwargs: dict[str, Any] = {}

    # allow arbitrary kwargs
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def make_env_args(self) -> dict[str, Any]:
        """Create the environment arguments."""
        return {
            "id": self.id,
            "max_episode_steps": self.max_episode_steps,
            "disable_env_checker": self.disable_env_checker,
            **self.env_kwargs,
        }

    @model_serializer
    def serialize(self, info: SerializationInfo) -> dict[str, Any]:
        """Serialize the model to a dictionary."""
        context = info.context
        if context:
            if context.get("flatten", False):
                return {
                    "id": self.id,
                    "max_episode_steps": self.max_episode_steps,
                    "disable_env_checker": self.disable_env_checker,
                    **self.env_kwargs,
                }

        return {
            "id": self.id,
            "max_episode_steps": self.max_episode_steps,
            "disable_env_checker": self.disable_env_checker,
            "env_kwargs": self.env_kwargs,
        }


class WrapperConfig(BaseModel):
    wrapper_class: type[Wrapper]
    wrapper_kwargs: dict[str, Any] = {}


class GymEnvConfig(BaseModel):
    make_env_config: MakeEnvConfig
    wrapper_config: WrapperConfig | None = None
