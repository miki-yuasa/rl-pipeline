from typing import Any, Literal, TypeVar


class BaseEnvLoader:
    env_config: Any
    wrapper_config: Any
    vec_config: (
        Any  # Type should be defined based on the actual configuration structure
    )

    def env(self):
        """Return the environment make configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def eval_env(self):
        """Return the evaluation environment make configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def vec_env(self):
        """Return the vectorized environment make configuration."""
        raise NotImplementedError("Subclasses must implement this method.")

    def eval_vec_env(self):
        """Return the evaluation vectorized environment make configuration."""
        raise NotImplementedError("Subclasses must implement this method.")


class BaseModelLoader:
    algo_config: Any

    # TODO: Come up with type annotation
    def model(self, env, device):
        """Return the model."""
        raise NotImplementedError("Subclasses must implement this method.")

    def load_model(self, filepath: str, env, device):
        """Load the model from the saved path."""
        raise NotImplementedError("Subclasses must implement this method.")

    def load_checkpoint(
        self,
        ckpt_dir: str,
        ckpt_name_prefix: str,
        timestep: int | Literal["latest"],
        env,
        device: str,
        file_ext: str = ".zip",
    ):
        """Load the model from the checkpoint."""
        raise NotImplementedError("Subclasses must implement this method.")


EnvLoaderType = TypeVar("EnvLoaderType", bound=BaseEnvLoader)
ModelLoaderType = TypeVar("ModelLoaderType", bound=BaseModelLoader)
