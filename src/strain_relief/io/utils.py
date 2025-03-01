from contextlib import contextmanager

from boto3 import client
from upath import UPath


@contextmanager
def s3_client():
    """Context manager for creating and closing an S3 client."""
    s3 = client("s3")
    try:
        yield s3
    finally:
        s3.close()


def copy_from_s3(s3_path: str, local_path: str) -> None:
    """Copy a file from an S3 bucket to a local path.

    Parameters:
    s3_path (str): The S3 path of the file to copy.
    local_path (str): The local path where the file will be copied to.
    """
    p = UPath(s3_path)
    with s3_client() as Client:
        Client.download_file(p.anchor[:-1], p.path.removeprefix(p.anchor), local_path)


def upload_to_s3(local_path: str, s3_bucket: str, s3_path: str) -> str:
    """Upload a file to an S3 bucket.

    Parameters:
    local_path (str): The local path of the file to upload.
    s3_bucket (str): The name of the S3 bucket.
    s3_path (str): The S3 path where the file will be uploaded.

    Returns:
    str: The S3 URI of the uploaded file.
    """
    with s3_client() as Client:
        Client.upload_file(local_path, s3_bucket, s3_path)
    s3_uri = f"s3://{s3_bucket}/{s3_path}"
    return s3_uri
