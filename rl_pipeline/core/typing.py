from typing import TypeVar

from pydantic import BaseModel

ConfigType = TypeVar("ConfigType", bound=BaseModel)

PipelineConfigType = TypeVar("PipelineConfigType", bound=BaseModel)
