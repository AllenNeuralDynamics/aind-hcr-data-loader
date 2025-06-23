"""S3 utilities for HCR data loading."""

import s3fs


def check_connect_to_bucket(bucket_name: str) -> bool:
    """
    Check if we can connect to a given S3 bucket using s3fs.

    Args:
        bucket_name (str): Name of the S3 bucket to check

    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        # Create s3fs filesystem object
        fs = s3fs.S3FileSystem()

        # Try to check if bucket exists and is accessible
        if fs.exists(bucket_name):
            return True
        else:
            print(f"Bucket '{bucket_name}' does not exist or is not accessible")
            return False

    except PermissionError:
        print(f"Error: Permission denied to access bucket '{bucket_name}'")
        return False
    except FileNotFoundError:
        print(f"Error: Bucket '{bucket_name}' not found")
        return False
    except Exception as e:
        print(f"Error connecting to bucket '{bucket_name}': {e}")
        return False


def check_prefix_exists(bucket_name: str, prefix: str) -> bool:
    """
    Check if a prefix exists in a given S3 bucket using s3fs.

    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): The prefix/path to check in the bucket

    Returns:
        bool: True if prefix exists, False otherwise
    """
    try:
        # Create s3fs filesystem object
        fs = s3fs.S3FileSystem()

        # Construct the full S3 path
        s3_path = f"{bucket_name}/{prefix}"

        # Check if the prefix exists as a directory or file
        if fs.exists(s3_path):
            return True

        # Also check if there are any objects with this prefix
        try:
            files = fs.ls(s3_path, detail=False)
            return len(files) > 0
        except FileNotFoundError:
            return False

    except PermissionError:
        print(f"Error: Permission denied to access prefix '{prefix}' in bucket '{bucket_name}'")
        return False
    except Exception as e:
        print(f"Error checking prefix '{prefix}' in bucket '{bucket_name}': {e}")
        return False


def list_prefix_contents(bucket_name: str, prefix: str, detail: bool = False) -> list:
    """
    List all objects under a given prefix in an S3 bucket using s3fs.

    Args:
        bucket_name (str): Name of the S3 bucket
        prefix (str): The prefix/path to list contents from
        detail (bool): If True, return detailed information about each object

    Returns:
        list: List of object paths (or detailed info if detail=True)
    """
    try:
        # Create s3fs filesystem object
        fs = s3fs.S3FileSystem()

        # Construct the full S3 path
        s3_path = f"{bucket_name}/{prefix}"

        # List contents under the prefix
        contents = fs.ls(s3_path, detail=detail)

        if detail:
            return contents
        else:
            # Remove bucket name from paths to return relative paths
            return [path.replace(f"{bucket_name}/", "") for path in contents]

    except PermissionError:
        print(f"Error: Permission denied to access prefix '{prefix}' in bucket '{bucket_name}'")
        return []
    except FileNotFoundError:
        print(f"Error: Prefix '{prefix}' not found in bucket '{bucket_name}'")
        return []
    except Exception as e:
        print(f"Error listing contents of prefix '{prefix}' in bucket '{bucket_name}': {e}")
        return []
