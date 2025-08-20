import os

from pydantic import BaseModel, computed_field


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
