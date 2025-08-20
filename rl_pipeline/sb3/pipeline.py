import os
from typing import Any

from stable_baselines3.common.base_class import BaseAlgorithm

from rl_pipeline.core.pipeline import BasePipeline
from rl_pipeline.utils.io import add_number_to_existing_filepath

from .config import SB3PipelineConfig
from .loader import SB3EnvLoader, SB3ModelLoader


class SB3Pipeline(BasePipeline[SB3PipelineConfig, SB3EnvLoader, SB3ModelLoader]):
    def __init__(self, config: SB3PipelineConfig, verbose: bool = True):
        super().__init__(config, verbose=verbose)
        self.env_loader = SB3EnvLoader(
            config.env_config, config.wrapper_config, config.vec_config
        )
        self.model_loader = SB3ModelLoader(
            config.algo_config, config.save_config.tb_save_dir
        )

    def train(self) -> BaseAlgorithm:
        """
        Train the model using the provided training configuration.
        """
        train_env = self.env_loader.vec_env()
        model = self.model_loader.model(train_env, device=self.config.device)
        model.learn(**self.config.learn_config.model_dump())
        train_env.close()
        os.makedirs(self.config.save_config.model_save_dir, exist_ok=True)
        # Save the model
        # if there is already an existing model, add a number suffix e.g. "_1"
        save_path: str = add_number_to_existing_filepath(
            self.config.save_config.model_save_path
        )
        model.save(save_path)

        return model

    def train_on_unsaved_model(self) -> BaseAlgorithm:
        """
        Train the model on an unsaved model.
        This method is a placeholder and should be implemented if needed.
        """
        demo_env = self.env_loader.env()
        if (
            not os.path.exists(self.config.save_config.model_save_path)
            or self.config.retrain_model
        ):
            model = self.train()
        else:
            print(
                f"SB3Pipeline: Model {self.config.save_config.model_save_path} already exists, loading..."
            )
            model = self.model_loader.load_model(
                self.config.save_config.model_save_path, demo_env, self.config.device
            )
        return model
    
    def
