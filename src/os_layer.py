# Authors: Sebastian Szyller
# Copyright 2022 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pickle
from pathlib import Path
from typing import Any, Optional, Union
import logging


def create_dir_if_doesnt_exist(path_to_dir: Union[str, Path], log: logging.Logger) -> Optional[Path]:
    """Create directory using the provided path.
    No explicit checks after creation because pathlib verifies it itself.
    Check pathlib source if errors happen.
    Args:
        path_to_dir (Union[str, Path]): Directory to be created
        log (logging.Logger): Logging facility
    Returns:
        Optional[Path]: Maybe Path to the created directory
    """

    path = Path(path_to_dir) if isinstance(path_to_dir, str) else path_to_dir
    resolved_path: Path = path.resolve()

    if not resolved_path.exists():
        log.info(f"{resolved_path} does not exist. Creating...")

        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            log.exception(e)
            return None

        log.info("{} created.".format(resolved_path))
    else:
        log.info("{} already exists.".format(resolved_path))

    return resolved_path


def create_dir_dirty(path_to_dir: Union[str, Path]) -> Path:
    """This is like create_dir_if_doesnt_exist but with no safeguards.
    Used only for stuff that is critical to the running of the system.
    Args:
        path_to_dir (Union[str, Path]): Create this directory.
    Returns:
        Path: Created/existing directory.
    """

    path = Path(path_to_dir) if isinstance(path_to_dir, str) else path_to_dir
    resolved_path: Path = path.resolve()

    if not resolved_path.exists():
        resolved_path.mkdir(parents=True, exist_ok=True)

    return resolved_path


def load_from_pickle_file(load_file: Union[str, Path], log: logging.Logger) -> Optional[Any]:
    """Load any file from a pickle binary.
    Args:
        load_file (Union[str, Path]): Absolute path (with file name)
        log (logging.Logger): Logging facility
    Returns:
        Optional[Any]: Loaded file if successful, None otherwise
    """

    path = Path(load_file) if isinstance(load_file, str) else load_file
    resolved_path: Path = path.resolve()

    if not resolved_path.exists():
        log.error("Provided file doesn't exist: {}".format(resolved_path))
        return None

    try:
        with resolved_path.open(mode="rb") as f:
            loaded_object: Any = pickle.load(f)

    except Exception as e:
        log.exception(e)
        return None

    return loaded_object


def save_to_pickle_file(obj: Any, save_file: Path, log: logging.Logger) -> Optional[Path]:
    """Save any file to a pickle binary.
    This does not check if required folder structure exists and will fail.
    Perform path checks yourself before calling.
    Args:
        obj (Any): Python object to be saved
        save_file (Path): Absolute path (with file name)
        log (logging.Logger): Logging facility
    Returns:
        Optional[Path]: Path where saved if successful, None otherwise
    """

    resolved_path: Path = save_file.resolve()

    try:
        with resolved_path.open(mode="wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        log.exception(e)
        return None

    if not resolved_path.exists():
        log.error("Saving went fine but file doesn't exist anyway {}.".format(resolved_path))
        return None

    return resolved_path
