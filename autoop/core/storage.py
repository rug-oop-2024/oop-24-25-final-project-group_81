from abc import ABC, abstractmethod
import os
from typing import List
from glob import glob


class NotFoundError(Exception):
    """
    A custom Not Found Error
    """

    def __init__(self, path: str) -> None:
        """
        A way to instantiate an exception.
        """
        super().__init__(f"Path not found: {path}")


class Storage(ABC):
    """
    An abstract class for Storage
    """

    @abstractmethod
    def save(self, data: bytes, path: str) -> None:
        """
        Save data to a given path
        Args:
            data (bytes): Data to save
            path (str): Path to save data
        """
        pass

    @abstractmethod
    def load(self, path: str) -> bytes:
        """
        Load data from a given path
        Args:
            path (str): Path to load data
        Returns:
            bytes: Loaded data
        """
        pass

    @abstractmethod
    def delete(self, path: str) -> None:
        """
        Delete data at a given path
        Args:
            path (str): Path to delete data
        """
        pass

    @abstractmethod
    def list(self, path: str) -> list:
        """
        List all paths under a given path
        Args:
            path (str): Path to list
        Returns:
            list: List of paths
        """
        pass


class LocalStorage(Storage):
    """
    A concrete Local Storage class, responsible for
    saving artifacts to a database.
    """

    def __init__(self, base_path: str = "./assets") -> None:
        """
        Initializes the LocalStorage instance.

        This constructor creates the base directory
        if it doesn't already exist.

        :param base_path: The base directory for storing artifacts.
        Defaults to './assets'.
        :type base_path: str
        """
        self._base_path = base_path
        if not os.path.exists(self._base_path):
            os.makedirs(self._base_path)

    def save(self, data: bytes, key: str) -> None:
        """
        Saves the provided data to a file at the specified key path.

        This method ensures that any necessary directories are created
        before saving the data to the local filesystem.

        :param data: The data to save to disk, in bytes.
        :type data: bytes
        :param key: The path (relative to base_path) where
        the data should be saved.
        :type key: str
        """
        path = self._join_path(key)
        if not os.path.exists(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def load(self, key: str) -> bytes:
        """
        Loads the data stored at the specified key path.

        This method reads and returns the data from the
        file associated with the key. If the file does not exist,
        it raises a NotFoundError.

        :param key: The path (relative to base_path) where the data is
        stored.
        :type key: str
        :return: The data stored at the specified path.
        :rtype: bytes
        :raises NotFoundError: If the file does not exist at the given path.
        """
        path = self._join_path(key)
        self._assert_path_exists(path)
        with open(path, "rb") as f:
            return f.read()

    def delete(self, key: str = "/") -> None:
        """
        Deletes the file at the specified key path.

        This method removes the file associated with the
        given key. If the file does not exist, it raises a NotFoundError.

        :param key: The path (relative to base_path) of the file to delete.
        Defaults to "/".
        :type key: str
        :raises NotFoundError: If the file does not exist at the given path.
        """
        self._assert_path_exists(self._join_path(key))
        path = self._join_path(key)
        os.remove(path)

    def list(self, prefix: str) -> List[str]:
        """
        Lists all files stored under a specific prefix.

        This method recursively lists all files within the
        directory corresponding to the given prefix and
        returns their relative paths.

        :param prefix: The prefix (directory path) under which to list files.
        :type prefix: str
        :return: A list of relative file paths stored under the given prefix.
        :rtype: List[str]
        :raises NotFoundError: If the directory does not exist.
        """
        path = self._join_path(prefix)
        self._assert_path_exists(path)
        keys = glob(path + "/**/*", recursive=True)
        # Extract relative paths and replace backslashes with forward slashes
        relative_keys = [
            os.path.relpath(key, self._base_path).replace("\\", "/")
            for key in keys
            if os.path.isfile(key)
        ]
        return relative_keys

    def _assert_path_exists(self, path: str) -> None:
        """
        Asserts that the specified path exists.

        This method raises a NotFoundError if the path does not exist.

        :param path: The path to check for existence.
        :type path: str
        :raises NotFoundError: If the path does not exist.
        """
        if not os.path.exists(path):
            raise NotFoundError(path)

    def _join_path(self, path: str) -> str:
        """
        Joins the base path with the provided path.

        This helper method combines the base directory path
        with a given relative path to generate the full path
        for storage operations.

        :param path: The relative path to join with the base path.
        :type path: str
        :return: The full path created by combining the base path
        and the relative path.
        :rtype: str
        """
        return os.path.join(self._base_path, path)
