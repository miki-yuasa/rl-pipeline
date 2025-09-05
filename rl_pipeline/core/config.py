import os
from typing import Generic, Self

import yaml
from pydantic import BaseModel, Field, computed_field

from rl_pipeline.core.typing import ConfigType
from rl_pipeline.core.utils.io import replace_extension


class SaveConfig(BaseModel):
    """Paths for saving models."""

    model_save_path: str
    best_model_save_path: str
    monitor_save_dir: str
    tb_save_dir: str
    eval_save_dir: str
    eval_metrics_save_path: str
    animation_save_path: str
    tb_save_dir: str

    @computed_field
    @property
    def model_save_dir(self) -> str:
        return os.path.dirname(self.model_save_path)

    @computed_field
    @property
    def model_full_name(self) -> str:
        # model_save_path - model_save_dir
        return os.path.basename(self.model_save_path)


class ConfigReader(Generic[ConfigType]):
    def to_config(self) -> ConfigType:
        raise NotImplementedError


class YAMLReaderMixin:
    @classmethod
    def from_yaml(cls, filepath: str) -> Self:
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class SaveConfigReader(BaseModel, YAMLReaderMixin):
    """Configuration for saving models."""

    models_dir: str = "out/model/"
    model_name: str = "my_model"
    model_filename: str = "final_model.zip"
    best_model_filename: str = "best_model.zip"
    monitor_dir: str = "monitor"
    tb_dir: str = "tb"
    eval_dir: str = "eval"
    eval_metrics_filename: str = "eval_metrics.yaml"
    animation_dir: str = "animations"
    animation_ext: str = "gif"
    include_suffix_in_filename: bool = True

    def to_config(
        self,
        experiment_id: str = "",
        model_name_suffix: str = "",
        replicate_signature: str = "",
    ) -> SaveConfig:
        model_full_name: str = self.model_name + "_" + model_name_suffix
        model_save_dir: str = os.path.join(
            self.models_dir, experiment_id, model_full_name, replicate_signature
        )
        model_filename: str = self._model_filename(
            "_".join(
                [
                    s
                    for s in [experiment_id, model_name_suffix, replicate_signature]
                    if s
                ]
            )
        )
        model_save_path: str = os.path.join(model_save_dir, model_filename)
        best_model_save_path: str = os.path.join(
            model_save_dir, self.best_model_filename
        )

        animation_save_dir: str = os.path.join(model_save_dir, self.animation_dir)
        animation_save_path: str = replace_extension(
            os.path.join(animation_save_dir, model_filename), self.animation_ext
        )

        monitor_save_dir: str = os.path.join(model_save_dir, self.monitor_dir)
        tb_save_dir: str = os.path.join(model_save_dir, self.tb_dir)
        eval_save_dir: str = os.path.join(model_save_dir, self.eval_dir)
        eval_metrics_save_path: str = os.path.join(
            eval_save_dir, self.eval_metrics_filename
        )

        save_config = SaveConfig(
            model_save_path=model_save_path,
            best_model_save_path=best_model_save_path,
            monitor_save_dir=monitor_save_dir,
            tb_save_dir=tb_save_dir,
            eval_save_dir=eval_save_dir,
            eval_metrics_save_path=eval_metrics_save_path,
            animation_save_path=animation_save_path,
        )

        return save_config

    def _model_filename(self, suffix: str = "") -> str:
        if self.include_suffix_in_filename and suffix:
            # Get base and extension from model_save_path
            base, ext = os.path.splitext(self.model_filename)
            return f"{base}_{suffix}{ext}"
        else:
            return self.model_filename


class ReplicateConfig(BaseModel):
    num_replicates: int = Field(ge=1, default=1)
    replicate_start_id: int = Field(ge=0, default=0)
    replicate_signature: str = "{rep_id}"
