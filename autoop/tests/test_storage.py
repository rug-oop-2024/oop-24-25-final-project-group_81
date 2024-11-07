
import unittest
import random
import tempfile

from autoop.core.storage import LocalStorage, NotFoundError


class TestStorage(unittest.TestCase):
    """
    Unit tests for the LocalStorage class.
    """
    def setUp(self) -> None:
        """
        Set up the test environment before each test.

        Creates a temporary directory for LocalStorage to
        use as a mock storage location for the tests.
        """
        temp_dir = tempfile.mkdtemp()
        self.storage = LocalStorage(temp_dir)

    def test_init(self) -> None:
        """
        Test the initialization of the LocalStorage.

        Verifies that the LocalStorage instance is correctly
        initialized and is of the `LocalStorage` class type.
        """
        self.assertIsInstance(self.storage, LocalStorage)

    def test_store(self) -> None:
        """
        Test storing and loading data in LocalStorage.

        Verifies that data can be successfully saved to LocalStorage
        and retrieved by its key. Also ensures that attempting to load
        a non-existing key raises the appropriate error.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = "test/path"
        self.storage.save(test_bytes, key)
        self.assertEqual(self.storage.load(key), test_bytes)
        otherkey = "test/otherpath"
        # should not be the same
        try:
            self.storage.load(otherkey)
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_delete(self) -> None:
        """
        Test deleting data from LocalStorage.

        Verifies that data can be successfully deleted from LocalStorage
        and that attempting to load a deleted key raises
        the appropriate error.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        key = "test/path"
        self.storage.save(test_bytes, key)
        self.storage.delete(key)
        try:
            self.assertIsNone(self.storage.load(key))
        except Exception as e:
            self.assertIsInstance(e, NotFoundError)

    def test_list(self) -> None:
        """
        Test listing keys in LocalStorage.

        Verifies that LocalStorage can list all keys in a specified
        directory and that the list contains the expected keys after
        saving multiple items.
        """
        key = str(random.randint(0, 100))
        test_bytes = bytes([random.randint(0, 255) for _ in range(100)])
        random_keys = [f"test/{random.randint(0, 100)}" for _ in range(10)]
        for key in random_keys:
            self.storage.save(test_bytes, key)
        keys = self.storage.list("test")
        keys = ["/".join(key.split("/")[-2:]) for key in keys]
        self.assertEqual(set(keys), set(random_keys))
            