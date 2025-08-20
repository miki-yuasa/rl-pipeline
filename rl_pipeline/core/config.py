import os
from typing import Generic, TypeVar

from pydantic import BaseModel, computed_field

from rl_pipeline.utils.io import replace_extension


class SaveConfig(BaseModel):
    """Paths for saving models."""

    model_save_path: str
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


ConfigType = TypeVar("ConfigType", bound=BaseModel)


class ConfigReader(Generic[ConfigType]):
    def to_config(self) -> ConfigType:
        raise NotImplementedError


class SaveConfigReader(BaseModel):
    """Configuration for saving models."""

    models_dir: str = "out/model/"
    model_name: str = "my_model"
    model_filename: str = "final_model.zip"
    monitor_dir: str = "monitor"
    tb_dir: str = "tb"
    eval_dir: str = "eval"
    eval_metrics_filename: str = "eval_metrics.yaml"
    animation_dir: str = "animations"
    animation_ext: str = "gif"
    include_model_name_suffix: bool = True

    def to_config(
        self, experiment_id: str = "", model_name_suffix: str = ""
    ) -> SaveConfig:
        model_full_name: str = self.model_name + model_name_suffix
        model_save_dir: str = os.path.join(
            self.models_dir, experiment_id, model_full_name
        )
        model_filename: str = self._model_filename(experiment_id + model_name_suffix)
        model_save_path: str = os.path.join(model_save_dir, model_filename)

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
            monitor_save_dir=monitor_save_dir,
            tb_save_dir=tb_save_dir,
            eval_save_dir=eval_save_dir,
            eval_metrics_save_path=eval_metrics_save_path,
            animation_save_path=animation_save_path,
        )

        return save_config

    def _model_filename(self, suffix: str = "") -> str:
        if suffix:
            # Get base and extension from model_save_path
            base, ext = os.path.splitext(self.model_filename)
            return f"{base}_{suffix}{ext}"
        else:
            return self.model_filename
