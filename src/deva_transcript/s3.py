
from typing import Annotated
from faststream import Depends
from minio import Minio

from config import settings

async def get_s3_client() -> Minio:
    return Minio(
        endpoint=settings.minio_ip + ":" + str(settings.minio_port),
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure
    )

S3_client = Annotated[Minio, Depends(get_s3_client)]