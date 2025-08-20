import importlib
import os
from typing import Literal


def format_large_number(timesteps: int) -> str:
    if timesteps >= 1_000_000:
        return f"{timesteps / 1_000_000}M"
    elif timesteps >= 1_000:
        return f"{timesteps / 1_000}K"
    else:
        return str(timesteps)


def get_class(cls_name: str | None) -> type | None:
    """Get a class by its name from the module."""
    if cls_name and "." in cls_name:
        module_name, class_name = cls_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    elif cls_name is None:
        return None
    else:
        raise ValueError(f"Class name '{cls_name}' must include the module prefix.")


def add_number_to_existing_filepath(file_path: str) -> str:
    """
    Add a number to the end of the file name if it already exists.

    Parameters
    ----------
    file_path: str
        The original file path to modify.

    Returns
    -------
    file_path: str
        The modified file path with a number added if it already existed.
    """
    base, ext = os.path.splitext(file_path)
    i = 1
    while os.path.exists(file_path):
        file_path = f"{base}_{i}{ext}"
        i += 1
    return file_path


def get_file_with_largest_number(file_path: str) -> str:
    """
    Get the file path with the largest number suffix.

    Parameters
    ----------
    file_path: str
        The original file path to modify.

    Returns
    -------
    file_path: str
        The modified file path with the largest number suffix.
    """
    base, ext = os.path.splitext(file_path)
    i = 1
    while True:
        new_file_path = f"{base}_{i}{ext}"
        if not os.path.exists(new_file_path):
            return file_path
        else:
            file_path = new_file_path
            i += 1


def get_ckpt_file(
    ckpt_dir: str,
    ckpt_name_prefix: str,
    timestep: int | Literal["latest"],
    file_ext: str = ".zip",
) -> str:
    """
    Retrieve the filepath of the checkpoint file ending with the specified timestep and file extension.

    Parameters
    ----------
    ckpt_dir: str
        Directory containing the checkpoint files.
    ckpt_name_prefix: str
        Prefix of the checkpoint file names.
    timestep: int | Literal["latest"]
        Timestep of the checkpoint file to retrieve.
    file_ext: str
        File extension of the checkpoint files.

    Returns
    -------
    ckpt_filepath: str
        The filepath of the checkpoint file.

    """
    ckpt_files: list[str] = [f for f in ckpt_dir if f.startswith(ckpt_name_prefix)]

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint files found in {ckpt_dir} directory")
    if timestep == "latest":
        # Find the latest checkpoint
        ckpt_files.sort()
        ckpt_file = ckpt_files[-1]
    else:
        # Find the checkpoint with the specified timestep
        ckpt_file = None
        for f in ckpt_files:
            if f.endswith(f"{timestep}{file_ext}"):
                ckpt_file = f
                break
        if ckpt_file is None:
            raise FileNotFoundError(
                f"No checkpoint file found for timestep {timestep} in {ckpt_dir}",
                f"Available files: {ckpt_files}",
            )

    return ckpt_file
