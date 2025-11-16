"""Persistent settings container."""
import os
import shelve
from pathlib import Path
from typing import Any

from roboeval.const import CACHE_PATH


class SettingsShelf:
    """Persistent settings container."""

    _SUFFIX = ".shelf"
    _SETTINGS_DIRECTORY = "settings_shelf"

    def __init__(self, file_name: str):
        """Init."""
        custom_directory = CACHE_PATH / self._SETTINGS_DIRECTORY
        if not os.path.exists(custom_directory):
            os.makedirs(custom_directory)
        file_name = Path(file_name)
        if file_name.suffix != self._SUFFIX:
            file_name.with_suffix(self._SUFFIX)
        self._shelf_file = str(custom_directory / file_name)

    def get(self, key: str, default: Any) -> Any:
        """Get value."""
        with shelve.open(self._shelf_file) as shelf:
            return shelf.get(key, default)

    def set(self, key: str, value: Any):
        """Set value."""
        with shelve.open(self._shelf_file) as shelf:
            shelf[key] = value
