import uuid
from datetime import datetime


def unique_id() -> str:
    """Generate a unique identifier for the training run."""
    return str(uuid.uuid4())[:4]


def exp_time() -> str:
    """Get the current time formatted for the experiment."""
    return datetime.now().strftime("%m-%d_%H-%M-%S.%f")
