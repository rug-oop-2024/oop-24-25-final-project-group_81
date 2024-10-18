import base64


class Artifact:
    def __init__(self, type_: str, *args, **kwargs):
        """
        The way of instantiating an artifact.

        :param type_: the type of the artifact
        :type type_: str
        """
        self._type = type_
        self._args = args
        self._kwargs = kwargs
        self._artifact_id = self._generate_id(
            self._kwargs["asset_path"], self._kwargs["version"]
            )
    
    @property
    def artifact_id(self) -> str:
        """
        Getter for the artrifact ID

        :return: the id of the artifact.
        :rtype: str
        """
        return self._artifact_id

    def _generate_id(self, asset_path: str, version: str) -> str:
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
            asset_path.encode('utf-8')
            ).decode('utf-8')
        
        # Combine the encoded asset_path and version into the id
        asset_id = f"{encoded_asset_path}:{version}"

        return asset_id

    def read(self) -> bytes:
        """
        A way to retrieve an artifact data.

        :return: the artifact's data.
        :rtype: bytes
        """
        return self._kwargs["data"]

    def save(self, bytes: bytes) -> None:
        """
        A way of saving an artifact data.

        :param bytes: the data in bytes.
        :type bytes: bytes
        """
        self._kwargs["data"] = bytes
