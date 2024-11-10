import base64


class Artifact:
    """
    A class defining Artifact that can be saved in a Database
    """

    def __init__(self, *args, **kwargs) -> None:
        """
        The way of instantiating an artifact.

        :param type_: the type of the artifact
        :type type_: str
        """
        self.asset_path = "assets\\objects"
        self.name = "Unknown"
        self.version = "beta"
        self.tags = []
        self.metadata = {}

        # Unpacking the kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Generating the id
        self.id = self._generate_id()

    def __repr__(self) -> str:
        """
        Defines the representation of the artifact.
        """
        return f"{self.name} (v{self.version})"

    def _generate_id(self) -> str:
        """
        This method generate an id for an artifact.

        :param asset_path: the path to the artifact.
        :type asset_path: str
        :param version: the version of the artifact.
        :type version: str
        :return: the id of the artifact with the template:
        id={base64(asset_path)}:{version}
        :rtype: str
        """
        # Encode the asset_path
        encoded_asset_path = base64.b64encode(
            self.asset_path.encode("utf-8")
        ).decode("utf-8")

        # Combine the encoded asset_path and version into the id
        asset_id = f"{encoded_asset_path}_{self.version}"

        return asset_id

    def read(self) -> bytes:
        """
        A way to retrieve an artifact data.

        :return: the artifact's data.
        :rtype: bytes
        """
        if not self.data:
            raise ValueError("No data available in the artifact.")
        return self.data

    def save(self, bytes: bytes) -> None:
        """
        A way of saving an artifact data.

        :param bytes: the data in bytes.
        :type bytes: bytes
        """
        self.data = bytes
