from __future__ import annotations

"""Core configuration models and serialization utilities."""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rl_pipeline.core.utils.io import replace_extension


class YAMLReaderMixin:
    """A mixin providing YAML and mapping deserialization methods."""

    @classmethod
    def from_yaml(cls, filepath: str | Path) -> Self:
        """Load configuration from a YAML file.

        Args:
            filepath: Path to the YAML file.

        Returns:
            Instantiated configuration model.
        """
        path = Path(filepath)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Create configuration from a dictionary.

        Args:
            data: Mapping containing configuration keys.

        Returns:
            Instantiated configuration model.
        """
        return cls(**dict(data))


def _compute_model_filename(
    base_filename: str,
    suffix: str,
    include_suffix: bool,
) -> str:
    """Compute filename with optional suffix.

    Args:
        base_filename: Base model filename.
        suffix: Suffix to append before extension.
        include_suffix: Whether suffix should be included.

    Returns:
        Formatted filename string.
    """
    if include_suffix and suffix:
        base, ext = os.path.splitext(base_filename)
        return f"{base}_{suffix}{ext}"
    return base_filename


class SaveConfig(BaseModel, YAMLReaderMixin):
    """Specification and resolved file paths for experiment artifacts.

    Attributes:
        models_dir: Directory where model artifacts are organized.
        model_name: Base name for the model.
        model_filename: File name for the final model archive.
        best_model_filename: File name for the best checkpoint archive.
        monitor_dir: Subdirectory name for monitor logs.
        tb_dir: Subdirectory name for TensorBoard event files.
        eval_dir: Subdirectory name for evaluation logs.
        eval_metrics_filename: Filename for evaluation summary metrics.
        animation_dir: Subdirectory name for rendering animations.
        animation_ext: File extension for generated animations.
        include_suffix_in_filename: Whether to append suffix to model filename.
        model_save_path: Fully resolved path to final model archive.
        best_model_save_path: Fully resolved path to best model archive.
        monitor_save_dir: Fully resolved directory for monitor logs.
        tb_save_dir: Fully resolved directory for TensorBoard logs.
        eval_save_dir: Fully resolved directory for evaluation metrics.
        eval_metrics_save_path: Fully resolved path to evaluation metrics YAML.
        animation_save_path: Fully resolved path to rollout animations.
    """

    models_dir: str = "out"
    model_name: str = "model"
    model_filename: str = "final_model.zip"
    best_model_filename: str = "best_model.zip"
    monitor_dir: str = "monitor"
    tb_dir: str = "tb"
    eval_dir: str = "eval"
    eval_metrics_filename: str = "eval_metrics.yaml"
    animation_dir: str = "animations"
    animation_ext: str = "gif"
    include_suffix_in_filename: bool = True

    model_save_path: str = ""
    best_model_save_path: str = ""
    monitor_save_dir: str = ""
    tb_save_dir: str = ""
    eval_save_dir: str = ""
    eval_metrics_save_path: str = ""
    animation_save_path: str = ""

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    @property
    def model_save_dir(self) -> str:
        """Directory containing the saved model files."""
        return os.path.dirname(self.model_save_path)

    @property
    def model_full_name(self) -> str:
        """Filename of the final saved model."""
        return os.path.basename(self.model_save_path)

    def resolve(
        self,
        experiment_id: str = "",
        model_name_suffix: str = "",
        replicate_signature: str = "",
    ) -> Self:
        """Resolve all output paths from template settings.

        Args:
            experiment_id: Identifier for the experiment.
            model_name_suffix: Suffix derived from training duration or specification.
            replicate_signature: Replica-specific tag for disambiguation.

        Returns:
            Self with populated absolute/relative paths.
        """
        suffix_parts = [
            s
            for s in [experiment_id, model_name_suffix, replicate_signature]
            if s
        ]
        full_suffix = "_".join(suffix_parts)

        dir_parts = [self.models_dir]
        if experiment_id:
            dir_parts.append(experiment_id)
        model_full_name = (
            f"{self.model_name}_{model_name_suffix}"
            if model_name_suffix
            else self.model_name
        )
        dir_parts.append(model_full_name)
        if replicate_signature:
            dir_parts.append(replicate_signature)

        target_model_save_dir = os.path.join(*dir_parts)
        filename = _compute_model_filename(
            self.model_filename,
            full_suffix,
            self.include_suffix_in_filename,
        )

        self.model_save_path = os.path.join(target_model_save_dir, filename)
        self.best_model_save_path = os.path.join(
            target_model_save_dir, self.best_model_filename
        )
        self.monitor_save_dir = os.path.join(
            target_model_save_dir, self.monitor_dir
        )
        self.tb_save_dir = os.path.join(target_model_save_dir, self.tb_dir)
        self.eval_save_dir = os.path.join(target_model_save_dir, self.eval_dir)
        self.eval_metrics_save_path = os.path.join(
            self.eval_save_dir, self.eval_metrics_filename
        )
        anim_dir = os.path.join(target_model_save_dir, self.animation_dir)
        self.animation_save_path = replace_extension(
            os.path.join(anim_dir, filename), self.animation_ext
        )
        return self


class ReplicateConfig(BaseModel, YAMLReaderMixin):
    """Configuration settings for experiment replication.

    Attributes:
        num_replicates: Total number of replicas to train.
        replicate_start_id: Starting replica index.
        replicate_signature: Template for replica directory name.
    """

    num_replicates: int = Field(ge=1, default=1)
    replicate_start_id: int = Field(ge=0, default=0)
    replicate_signature: str = "{rep_id}"

    model_config = ConfigDict(extra="allow")
