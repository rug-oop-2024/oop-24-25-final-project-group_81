import unittest
import random
import tempfile

from autoop.core.database import Database
from autoop.core.storage import LocalStorage


class TestDatabase(unittest.TestCase):
    """
    Unit tests for the Database class, which tests the core functionalities
    of interacting with the database, including setting, getting, deleting,
    and persisting data, as well as refreshing the database.
    """

    def setUp(self) -> None:
        """
        Set up the test environment before each test.

        Creates a temporary storage location and initializes a
        Database instance with it. This ensures that each test
        starts with a fresh environment.
        """
        self.storage = LocalStorage(tempfile.mkdtemp())
        self.db = Database(self.storage)

    def test_init(self) -> None:
        """
        Test the initialization of the Database.

        Verifies that the Database instance is created successfully
        and is an instance of the Database class.
        """
        self.assertIsInstance(self.db, Database)

    def test_set(self) -> None:
        """
        Test the 'set' method of the Database.

        Verifies that an entry can be added to the database by
        setting a key-value pair in a specific collection.
        Ensures that the value can be retrieved
        correctly afterward.
        """
        id = str(random.randint(0, 100))
        entry = {"key": random.randint(0, 100)}
        self.db.set("collection", id, entry)
        self.assertEqual(self.db.get("collection", id)["key"], entry["key"])

    def test_delete(self) -> None:
        """
        Test the 'delete' method of the Database.

        Verifies that an entry can be deleted from the
        database by specifying its collection and ID.
        Ensures that the deleted entry is no longer retrievable,
        even after calling the 'refresh' method.
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        self.db.delete("collection", id)
        self.assertIsNone(self.db.get("collection", id))
        self.db.refresh()
        self.assertIsNone(self.db.get("collection", id))

    def test_persistance(self) -> None:
        """
        Test persistence of data in the Database.

        Verifies that data added to the database persists
        across different instances of the Database.
        A second instance of the Database is created
        using the same storage, and it is verified that
        the data set in the first instance is still accessible.
        """
        id = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", id, value)
        other_db = Database(self.storage)
        self.assertEqual(other_db.get("collection", id)["key"], value["key"])

    def test_refresh(self) -> None:
        """
        Test the 'refresh' method of the Database.

        Verifies that the 'refresh' method reloads
        the most recent data from
        storage into the database instance.
        After setting a value in one instance,
        it ensures that another instance, after
        calling 'refresh', has the same
        value for the same key.
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        other_db = Database(self.storage)
        self.db.set("collection", key, value)
        other_db.refresh()
        self.assertEqual(other_db.get("collection", key)["key"], value["key"])

    def test_list(self):
        """
        Test the 'list' method of the Database.

        Verifies that the 'list' method
        correctly lists all entries in a
        collection. After adding a key-value
        pair to a collection, the key-value
        pair should appear in the list of the collection.
        """
        key = str(random.randint(0, 100))
        value = {"key": random.randint(0, 100)}
        self.db.set("collection", key, value)
        # collection should now contain the key
        self.assertIn((key, value), self.db.list("collection"))
