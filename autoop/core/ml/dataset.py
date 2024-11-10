import pandas as pd
import io

from autoop.core.ml.artifact import Artifact


class Dataset(Artifact):
    """
    A class for encompasing a dataset
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Instantiates a dataset by creating an artifact of type
        `dataset`.
        """
        super().__init__(*args, **kwargs)

    @staticmethod
    def from_dataframe(
        data: pd.DataFrame, name: str, asset_path: str, version: str = "1.0.0"
    ) -> "Dataset":
        """
        This static method is used to create a dataset object
        from data frame.


        :param data: the data.
        :type data: pd.DataFrame
        :param name: the name.
        :type name: str
        :param asset_path: the path to the dataset.
        :type asset_path: str
        :param version: the version of the dataset, defaults to "1.0.0"
        :type version: str, optional
        :return: a Dataset object.
        :rtype: Dataset
        """
        return Dataset(
            name=name,
            asset_path=asset_path,
            data=data.to_csv(index=False).encode(),
            version=version,
            type="dataset",
        )

    def read(self) -> pd.DataFrame:
        """
        This method reads the data of the Dataset.

        :return: the data of the Dataset.
        :rtype: pd.DataFrame
        """
        bytes = super().read()
        csv = bytes.decode()
        return pd.read_csv(io.StringIO(csv))

    def save(self, data: pd.DataFrame) -> bytes:
        """
        This method is used to save the data as bytes
        in an artifact.

        :param data: the data that needs to be saved.
        :type data: pd.DataFrame
        :return: the encoded data as bytes.
        :rtype: bytes
        """
        bytes = data.to_csv(index=False).encode()
        return super().save(bytes)
