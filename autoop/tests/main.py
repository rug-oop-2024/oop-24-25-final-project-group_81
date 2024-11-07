import unittest
from autoop.tests.test_database import TestDatabase
from autoop.tests.test_storage import TestStorage
from autoop.tests.test_features import TestFeatures
from autoop.tests.test_pipeline import TestPipeline


_ = TestDatabase()
_ = TestStorage()
_ = TestFeatures()
_ = TestPipeline()

if __name__ == '__main__':
    unittest.main()
    