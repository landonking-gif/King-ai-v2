"""
MinIO/S3 Storage Adapter for Cold Storage.

Module: memory-service/service/storage/s3.py
"""

from typing import Any, Dict, Optional
import json
from io import BytesIO

import anyio
from minio import Minio
from minio.error import S3Error

from ..config import Settings


class S3Adapter:
    """MinIO/S3 adapter for cold storage of large artifacts."""

    def __init__(self, settings: Settings) -> None:
        """
        Initialize S3 adapter.

        Args:
            settings: Service configuration settings
        """
        self.settings = settings
        self.client: Optional[Minio] = None
        self.bucket_name = settings.minio_bucket

    async def connect(self) -> None:
        """Establish connection to MinIO/S3."""

        def _connect() -> Minio:
            return Minio(
                self.settings.minio_endpoint,
                access_key=self.settings.minio_access_key,
                secret_key=self.settings.minio_secret_key,
                secure=self.settings.minio_secure,
            )

        self.client = await anyio.to_thread.run_sync(_connect)

        # Ensure bucket exists
        await self._ensure_bucket_exists()

    async def disconnect(self) -> None:
        """Close S3 connection."""
        # MinIO client doesn't require explicit disconnect
        self.client = None

    async def _ensure_bucket_exists(self) -> None:
        """Create bucket if it doesn't exist."""
        if not self.client:
            raise RuntimeError("S3 client not connected")

        def _check_and_create() -> None:
            if not self.client.bucket_exists(self.bucket_name):
                self.client.make_bucket(self.bucket_name)

        await anyio.to_thread.run_sync(_check_and_create)

    async def store_artifact(
        self, artifact_id: str, content: Dict[str, Any], content_type: str = "application/json"
    ) -> str:
        """
        Store artifact in cold storage.

        Args:
            artifact_id: Unique artifact identifier
            content: Artifact content to store
            content_type: MIME type of content

        Returns:
            Storage reference (object key)
        """
        if not self.client:
            raise RuntimeError("S3 client not connected")

        object_key = f"artifacts/{artifact_id}.json"

        def _put_object() -> None:
            # Serialize content to JSON
            serialized = json.dumps(content, indent=2, default=str)
            data = serialized.encode("utf-8")
            data_stream = BytesIO(data)

            self.client.put_object(
                bucket_name=self.bucket_name,
                object_name=object_key,
                data=data_stream,
                length=len(data),
                content_type=content_type,
            )

        await anyio.to_thread.run_sync(_put_object)
        return object_key

    async def retrieve_artifact(self, storage_ref: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve artifact from cold storage.

        Args:
            storage_ref: Storage reference (object key)

        Returns:
            Artifact content if found, None otherwise
        """
        if not self.client:
            raise RuntimeError("S3 client not connected")

        def _get_object() -> Optional[bytes]:
            try:
                response = self.client.get_object(
                    bucket_name=self.bucket_name, object_name=storage_ref
                )
                data = response.read()
                response.close()
                response.release_conn()
                return data
            except S3Error as e:
                if e.code == "NoSuchKey":
                    return None
                raise

        data = await anyio.to_thread.run_sync(_get_object)
        if data:
            return json.loads(data.decode("utf-8"))
        return None

    async def delete_artifact(self, storage_ref: str) -> bool:
        """
        Delete artifact from cold storage.

        Args:
            storage_ref: Storage reference (object key)

        Returns:
            True if deleted, False if not found
        """
        if not self.client:
            raise RuntimeError("S3 client not connected")

        def _remove_object() -> bool:
            try:
                self.client.remove_object(bucket_name=self.bucket_name, object_name=storage_ref)
                return True
            except S3Error as e:
                if e.code == "NoSuchKey":
                    return False
                raise

        return await anyio.to_thread.run_sync(_remove_object)

    async def list_artifacts(self, prefix: str = "artifacts/") -> list[str]:
        """
        List all artifact references with given prefix.

        Args:
            prefix: Object key prefix

        Returns:
            List of storage references
        """
        if not self.client:
            raise RuntimeError("S3 client not connected")

        def _list_objects() -> list[str]:
            objects = self.client.list_objects(
                bucket_name=self.bucket_name, prefix=prefix, recursive=True
            )
            return [obj.object_name for obj in objects]

        return await anyio.to_thread.run_sync(_list_objects)

    async def get_artifact_metadata(self, storage_ref: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for an artifact without downloading content.

        Args:
            storage_ref: Storage reference (object key)

        Returns:
            Metadata dict if found, None otherwise
        """
        if not self.client:
            raise RuntimeError("S3 client not connected")

        def _stat_object() -> Optional[Dict[str, Any]]:
            try:
                stat = self.client.stat_object(
                    bucket_name=self.bucket_name, object_name=storage_ref
                )
                return {
                    "size": stat.size,
                    "etag": stat.etag,
                    "content_type": stat.content_type,
                    "last_modified": stat.last_modified,
                    "metadata": stat.metadata,
                }
            except S3Error as e:
                if e.code == "NoSuchKey":
                    return None
                raise

        return await anyio.to_thread.run_sync(_stat_object)
