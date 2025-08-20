from pydantic import BaseModel


class BaseEnvLoader:
    def __init__(self, config: BaseModel) -> None:
        self.config = config

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
    def __init__(self, config: BaseModel) -> None:
        self.config = config

    def model(self):
        """Return the model."""
        raise NotImplementedError("Subclasses must implement this method.")

    def load_model(self, env=None):
        """Load the model from the saved path."""
        raise NotImplementedError("Subclasses must implement this method.")
