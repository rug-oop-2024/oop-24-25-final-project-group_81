from autoop.core.storage import LocalStorage
from autoop.core.database import Database
from autoop.core.ml.artifact import Artifact
from autoop.core.storage import Storage
from typing import List


class ArtifactRegistry:
    """
    Manages the registration, retrieval, listing,
    and deletion of artifacts.
    Artifacts are stored in a storage system,
    and metadata is saved in a database.
    """

    def __init__(self, database: Database, storage: Storage) -> None:
        """
        Initializes the ArtifactRegistry with a given database
        and storage system.

        :param database: The database instance to store
        artifact metadata.
        :type database: Database
        :param storage: The storage instance to save and
        load artifact data.
        :type storage: Storage
        """

        self._database = database
        self._storage = storage

    def register(self, artifact: Artifact) -> None:
        """
        Registers an artifact by saving its data to
        storage and its metadata to the database.

        :param artifact: The artifact to be registered.
        :type artifact: Artifact
        :raises TypeError: If the artifact's data is
        invalid or missing.
        """
        try:
            # save the artifact in the storage
            self._storage.save(artifact.data, artifact.asset_path)
        except TypeError:
            print(
                f"{artifact.name} has no data to be saved!"
                "Procceded with saving the metadata..."
            )
        # save the metadata in the database
        entry = {
            "name": artifact.name,
            "version": artifact.version,
            "asset_path": artifact.asset_path,
            "tags": artifact.tags,
            "metadata": artifact.metadata,
            "type": artifact.type,
        }
        self._database.set("artifacts", artifact.id, entry)

    def list(self, type: str = None) -> List[Artifact]:
        """
        Lists all registered artifacts, optionally filtered by type.

        :param type: The type of artifacts to filter by (optional).
        :type type: str, optional
        :return: A list of artifacts matching the specified type
        (if provided).
        :rtype: List[Artifact]
        """
        entries = self._database.list("artifacts")
        artifacts = []
        for id, data in entries:
            if type is not None and data["type"] != type:
                continue
            artifact = Artifact(
                name=data["name"],
                version=data["version"],
                asset_path=data["asset_path"],
                tags=data["tags"],
                metadata=data["metadata"],
                data=self._storage.load(data["asset_path"]),
                type=data["type"],
            )
            artifacts.append(artifact)
        return artifacts

    def get(self, artifact_id: str) -> Artifact:
        """
        Retrieves an artifact by its ID.

        :param artifact_id: The ID of the artifact to retrieve.
        :type artifact_id: str
        :return: The artifact with the specified ID.
        :rtype: Artifact
        """
        data = self._database.get("artifacts", artifact_id)
        return Artifact(
            name=data["name"],
            version=data["version"],
            asset_path=data["asset_path"],
            tags=data["tags"],
            metadata=data["metadata"],
            data=self._storage.load(data["asset_path"]),
            type=data["type"],
        )

    def delete(self, artifact_id: str) -> None:
        """
        Deletes an artifact by its ID, including its
        data in storage and metadata in the database.

        :param artifact_id: The ID of the artifact to delete.
        :type artifact_id: str
        """
        data = self._database.get("artifacts", artifact_id)
        self._storage.delete(data["asset_path"])
        self._database.delete("artifacts", artifact_id)


class AutoMLSystem:
    """
    A singleton class representing an AutoML system
    that manages artifacts and their metadata.

    The AutoMLSystem is responsible for interacting
    with a storage system and a database
    to manage machine learning artifacts
    and related metadata.
    """

    _instance = None

    def __init__(self, storage: LocalStorage, database: Database) -> None:
        """
        Initializes the AutoMLSystem with the specified
        storage and database systems.

        This constructor is private to enforce the singleton
        pattern, and should not be called directly.
        Use `get_instance()` to get the single instance of AutoMLSystem.

        :param storage: The local storage instance used for
        saving and loading artifact data.
        :type storage: LocalStorage
        :param database: The database instance used for
        storing artifact metadata.
        :type database: Database
        """
        self._storage = storage
        self._database = database
        self._registry = ArtifactRegistry(database, storage)

    @staticmethod
    def get_instance() -> "AutoMLSystem":
        """
        Retrieves the single instance of the AutoMLSystem.
        If the instance does not exist, it is created and initialized.

        :return: The singleton instance of the AutoMLSystem.
        :rtype: AutoMLSystem
        """
        if AutoMLSystem._instance is None:
            AutoMLSystem._instance = AutoMLSystem(
                LocalStorage("./assets/objects"), Database(LocalStorage("./assets/dbo"))
            )
        AutoMLSystem._instance._database.refresh()
        return AutoMLSystem._instance

    @property
    def registry(self) -> ArtifactRegistry:
        """
        Returns the artifact registry for the AutoMLSystem.

        The registry is used to manage and interact
        with artifacts and their metadata.

        :return: The artifact registry instance.
        :rtype: ArtifactRegistry
        """
        return self._registry
