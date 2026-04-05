"""
Azure Data Lake Storage (ADLS) Fetcher Module
Handles reading JSON documents from ADLS Gen2.
"""
import json
import logging
from typing import List, Dict, Optional, Generator
from pathlib import Path
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import AzureError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ADLSFetcher:
    """Fetches JSON documents from Azure Data Lake Storage Gen2."""
    
    def __init__(
        self,
        account_name: str,
        account_key: str,
        container_name: str
    ):
        """
        Initialize ADLS fetcher.
        
        Args:
            account_name: Azure storage account name
            account_key: Azure storage account key
            container_name: Container name in ADLS
        """
        self.account_name = account_name
        self.container_name = container_name
        
        logger.info(f"Initializing ADLS connection: account={account_name}, container={container_name}")
        
        try:
            account_url = f"https://{account_name}.dfs.core.windows.net"
            self.service_client = DataLakeServiceClient(
                account_url=account_url,
                credential=account_key
            )
            self.file_system_client = self.service_client.get_file_system_client(container_name)
            logger.info("ADLS connection established successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ADLS client: {e}")
            raise
    
    def list_files(
        self,
        path: str = "",
        pattern: str = "*.json",
        recursive: bool = True
    ) -> List[str]:
        """
        List all files matching pattern in ADLS path.
        
        Args:
            path: Base path in container
            pattern: File pattern (e.g., '*.json')
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        logger.info(f"Listing files in path: {path}, pattern: {pattern}, recursive: {recursive}")
        
        try:
            paths = self.file_system_client.get_paths(path=path, recursive=recursive)
            
            # Filter by pattern
            extension = pattern.replace("*", "")
            file_paths = [
                p.name for p in paths 
                if not p.is_directory and p.name.endswith(extension)
            ]
            
            logger.info(f"Found {len(file_paths)} files matching pattern")
            return file_paths
            
        except AzureError as e:
            logger.error(f"Error listing files: {e}")
            raise
    
    def read_json_file(self, file_path: str) -> Dict:
        """
        Read a single JSON file from ADLS.
        
        Args:
            file_path: Path to file in container
            
        Returns:
            Parsed JSON as dictionary
        """
        try:
            file_client = self.file_system_client.get_file_client(file_path)
            download = file_client.download_file()
            content = download.readall()
            
            # Parse JSON
            data = json.loads(content.decode('utf-8'))
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise
        except AzureError as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise
    
    def read_multiple_files(
        self,
        file_paths: List[str],
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Read multiple JSON files from ADLS.
        
        Args:
            file_paths: List of file paths
            show_progress: Whether to show progress bar
            
        Returns:
            List of parsed JSON documents
        """
        logger.info(f"Reading {len(file_paths)} files from ADLS")
        
        documents = []
        iterator = tqdm(file_paths, desc="Reading files") if show_progress else file_paths
        
        for file_path in iterator:
            try:
                doc = self.read_json_file(file_path)
                
                # Add source metadata
                if isinstance(doc, dict):
                    doc["_source_file"] = file_path
                
                documents.append(doc)
                
            except Exception as e:
                logger.warning(f"Skipping file {file_path} due to error: {e}")
                continue
        
        logger.info(f"Successfully read {len(documents)} documents")
        return documents
    
    def fetch_all(
        self,
        path: str = "",
        pattern: str = "*.json",
        recursive: bool = True,
        max_files: Optional[int] = None,
        show_progress: bool = True
    ) -> List[Dict]:
        """
        Fetch all JSON documents from ADLS path.
        
        Args:
            path: Base path in container
            pattern: File pattern
            recursive: Whether to search recursively
            max_files: Maximum number of files to read (None = all)
            show_progress: Whether to show progress
            
        Returns:
            List of documents
        """
        # List files
        file_paths = self.list_files(path, pattern, recursive)
        
        # Limit if specified
        if max_files and max_files < len(file_paths):
            logger.info(f"Limiting to {max_files} files (out of {len(file_paths)})")
            file_paths = file_paths[:max_files]
        
        # Read files
        return self.read_multiple_files(file_paths, show_progress)
    
    def fetch_generator(
        self,
        path: str = "",
        pattern: str = "*.json",
        recursive: bool = True,
        max_files: Optional[int] = None,
        show_progress: bool = True
    ) -> Generator[Dict, None, None]:
        """
        Fetch documents one at a time (memory efficient).
        
        Args:
            path: Base path in container
            pattern: File pattern
            recursive: Whether to search recursively
            max_files: Maximum number of files to read
            show_progress: Whether to show progress
            
        Yields:
            Individual documents
        """
        # List files
        file_paths = self.list_files(path, pattern, recursive)
        
        # Limit if specified
        if max_files and max_files < len(file_paths):
            file_paths = file_paths[:max_files]
        
        logger.info(f"Fetching {len(file_paths)} files as generator")
        
        iterator = tqdm(file_paths, desc="Fetching files") if show_progress else file_paths
        
        for file_path in iterator:
            try:
                doc = self.read_json_file(file_path)
                
                # Add source metadata
                if isinstance(doc, dict):
                    doc["_source_file"] = file_path
                
                yield doc
                
            except Exception as e:
                logger.warning(f"Skipping file {file_path} due to error: {e}")
                continue
    
    def download_to_local(
        self,
        adls_path: str,
        local_path: str,
        pattern: str = "*.json",
        recursive: bool = True
    ) -> int:
        """
        Download files from ADLS to local filesystem.
        
        Args:
            adls_path: Path in ADLS
            local_path: Local directory path
            pattern: File pattern
            recursive: Whether to search recursively
            
        Returns:
            Number of files downloaded
        """
        logger.info(f"Downloading files from {adls_path} to {local_path}")
        
        # Create local directory
        Path(local_path).mkdir(parents=True, exist_ok=True)
        
        # List files
        file_paths = self.list_files(adls_path, pattern, recursive)
        
        downloaded = 0
        for file_path in tqdm(file_paths, desc="Downloading"):
            try:
                # Read from ADLS
                file_client = self.file_system_client.get_file_client(file_path)
                download = file_client.download_file()
                content = download.readall()
                
                # Write to local
                local_file = Path(local_path) / Path(file_path).name
                local_file.write_bytes(content)
                
                downloaded += 1
                
            except Exception as e:
                logger.warning(f"Failed to download {file_path}: {e}")
                continue
        
        logger.info(f"Downloaded {downloaded} files to {local_path}")
        return downloaded


def fetch_from_adls(
    account_name: str,
    account_key: str,
    container_name: str,
    path: str = "",
    pattern: str = "*.json",
    recursive: bool = True,
    max_files: Optional[int] = None
) -> List[Dict]:
    """
    Convenience function to fetch documents from ADLS.
    
    Args:
        account_name: Azure storage account name
        account_key: Azure storage account key
        container_name: Container name
        path: Base path in container
        pattern: File pattern
        recursive: Whether to search recursively
        max_files: Maximum files to read
        
    Returns:
        List of documents
    """
    fetcher = ADLSFetcher(account_name, account_key, container_name)
    return fetcher.fetch_all(path, pattern, recursive, max_files)


if __name__ == "__main__":
    # Test with environment variables
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test connection
    try:
        account_name = os.getenv("ADLS_ACCOUNT_NAME")
        account_key = os.getenv("ADLS_ACCOUNT_KEY")
        container_name = os.getenv("ADLS_CONTAINER_NAME")
        if account_name is None or account_key is None or container_name is None:
            raise EnvironmentError(
                "Environment variables ADLS_ACCOUNT_NAME, ADLS_ACCOUNT_KEY and ADLS_CONTAINER_NAME must be set"
            )

        fetcher = ADLSFetcher(
            account_name=account_name,
            account_key=account_key,
            container_name=container_name
        )
        
        # List files
        path = os.getenv("ADLS_INPUT_PATH", "")
        files = fetcher.list_files(path, recursive=True)
        print(f"\nFound {len(files)} JSON files")
        
        if files:
            print(f"\nFirst 5 files:")
            for f in files[:5]:
                print(f"  - {f}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
