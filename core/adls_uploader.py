"""
Azure Data Lake Storage (ADLS) Uploader Module
Handles writing JSON documents and chunks back to ADLS Gen2.
"""
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path
from azure.storage.filedatalake import DataLakeServiceClient
from azure.core.exceptions import AzureError
from tqdm import tqdm

logger = logging.getLogger(__name__)


class ADLSUploader:
    """Uploads JSON documents and chunks to Azure Data Lake Storage Gen2."""
    
    def __init__(
        self,
        account_name: str,
        account_key: str,
        container_name: str
    ):
        """
        Initialize ADLS uploader.
        
        Args:
            account_name: Azure storage account name
            account_key: Azure storage account key
            container_name: Container name in ADLS
        """
        self.account_name = account_name
        self.container_name = container_name
        
        logger.info(f"Initializing ADLS uploader: account={account_name}, container={container_name}")
        
        try:
            account_url = f"https://{account_name}.dfs.core.windows.net"
            self.service_client = DataLakeServiceClient(
                account_url=account_url,
                credential=account_key
            )
            self.file_system_client = self.service_client.get_file_system_client(container_name)
            logger.info("ADLS uploader connection established successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ADLS uploader client: {e}")
            raise
    
    def upload_json_file(
        self,
        data: Dict | List[Dict],
        adls_path: str,
        overwrite: bool = True
    ) -> bool:
        """
        Upload JSON data to ADLS as a file.
        
        Args:
            data: Dictionary or list to save as JSON
            adls_path: Path in ADLS container (e.g., 'output/chunks/all_chunks.json')
            overwrite: Whether to overwrite existing file
            
        Returns:
            True if successful
        """
        try:
            logger.info(f"Uploading JSON to ADLS path: {adls_path}")
            
            # Convert to JSON string
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            json_bytes = json_str.encode('utf-8')
            
            # Get file client
            file_client = self.file_system_client.get_file_client(adls_path)
            
            # Upload (create or overwrite)
            file_client.upload_data(
                data=json_bytes,
                overwrite=overwrite
            )
            
            logger.info(f"Successfully uploaded {len(json_bytes)} bytes to {adls_path}")
            return True
            
        except AzureError as e:
            logger.error(f"Failed to upload file to {adls_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading to {adls_path}: {e}")
            return False
    
    def upload_chunks(
        self,
        chunks: List[Dict],
        base_path: str = "output/chunks",
        filename: str = "all_chunks.json",
        show_progress: bool = False
    ) -> bool:
        """
        Upload chunks as a single JSON file to ADLS.
        
        Args:
            chunks: List of chunk dictionaries
            base_path: Base directory path in ADLS
            filename: Name of the output file
            show_progress: Whether to show progress (for large files)
            
        Returns:
            True if successful
        """
        adls_path = f"{base_path}/{filename}"
        
        logger.info(f"Uploading {len(chunks)} chunks to ADLS: {adls_path}")
        
        return self.upload_json_file(chunks, adls_path, overwrite=True)
    
    def upload_chunks_individually(
        self,
        chunks: List[Dict],
        base_path: str = "output/chunks/individual",
        show_progress: bool = True
    ) -> Dict[str, int]:
        """
        Upload each chunk as a separate JSON file (useful for very large datasets).
        
        Args:
            chunks: List of chunk dictionaries
            base_path: Base directory path in ADLS
            show_progress: Whether to show progress bar
            
        Returns:
            Statistics dictionary
        """
        logger.info(f"Uploading {len(chunks)} chunks individually to {base_path}")
        
        uploaded = 0
        failed = 0
        
        iterator = tqdm(chunks, desc="Uploading chunks") if show_progress else chunks
        
        for chunk in iterator:
            chunk_id = chunk.get("chunk_id", chunk.get("id", "unknown"))
            chunk_path = f"{base_path}/{chunk_id}.json"
            
            if self.upload_json_file(chunk, chunk_path, overwrite=True):
                uploaded += 1
            else:
                failed += 1
        
        stats = {
            "total": len(chunks),
            "uploaded": uploaded,
            "failed": failed,
            "success_rate": (uploaded / len(chunks) * 100) if chunks else 0
        }
        
        logger.info(
            f"Individual upload complete: {uploaded}/{len(chunks)} uploaded "
            f"({stats['success_rate']:.1f}% success rate)"
        )
        
        return stats
    
    def create_directory(self, directory_path: str) -> bool:
        """
        Create a directory in ADLS (if it doesn't exist).
        
        Args:
            directory_path: Path to directory
            
        Returns:
            True if created or already exists
        """
        try:
            directory_client = self.file_system_client.get_directory_client(directory_path)
            
            # Check if exists
            try:
                directory_client.get_directory_properties()
                logger.debug(f"Directory already exists: {directory_path}")
                return True
            except:
                # Doesn't exist, create it
                directory_client.create_directory()
                logger.info(f"Created directory: {directory_path}")
                return True
                
        except AzureError as e:
            logger.error(f"Failed to create directory {directory_path}: {e}")
            return False
    
    def upload_pipeline_outputs(
        self,
        all_chunks: List[Dict],
        top_k_chunks: Optional[List[Dict]] = None,
        stats: Optional[Dict] = None,
        base_path: str = "output/pipeline",
        timestamp: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        Upload all pipeline outputs to ADLS in organized structure.
        
        Args:
            all_chunks: All processed chunks
            top_k_chunks: Top-k selected chunks (optional)
            stats: Pipeline statistics (optional)
            base_path: Base output directory in ADLS
            timestamp: Optional timestamp for versioning
            
        Returns:
            Dictionary of upload results
        """
        logger.info(f"Uploading pipeline outputs to ADLS: {base_path}")
        
        # Create timestamped path if provided
        if timestamp:
            output_path = f"{base_path}/{timestamp}"
        else:
            output_path = base_path
        
        # Create base directory
        self.create_directory(output_path)
        
        results = {}
        
        # Upload all chunks
        all_chunks_path = f"{output_path}/all_chunks.json"
        results["all_chunks"] = self.upload_json_file(all_chunks, all_chunks_path)
        
        # Upload top-k chunks if provided
        if top_k_chunks:
            top_k_path = f"{output_path}/top_k_chunks.json"
            results["top_k_chunks"] = self.upload_json_file(top_k_chunks, top_k_path)
        
        # Upload statistics if provided
        if stats:
            stats_path = f"{output_path}/processing_stats.json"
            results["stats"] = self.upload_json_file(stats, stats_path)
        
        # Log summary
        successful = sum(1 for v in results.values() if v)
        logger.info(
            f"Pipeline outputs upload complete: {successful}/{len(results)} files uploaded successfully"
        )
        
        return results
    
    def file_exists(self, adls_path: str) -> bool:
        """
        Check if a file exists in ADLS.
        
        Args:
            adls_path: Path to file in container
            
        Returns:
            True if file exists
        """
        try:
            file_client = self.file_system_client.get_file_client(adls_path)
            file_client.get_file_properties()
            return True
        except:
            return False
    
    def delete_file(self, adls_path: str) -> bool:
        """
        Delete a file from ADLS.
        
        Args:
            adls_path: Path to file in container
            
        Returns:
            True if deleted successfully
        """
        try:
            file_client = self.file_system_client.get_file_client(adls_path)
            file_client.delete_file()
            logger.info(f"Deleted file: {adls_path}")
            return True
        except AzureError as e:
            logger.error(f"Failed to delete file {adls_path}: {e}")
            return False


def upload_to_adls(
    data: Dict | List[Dict],
    adls_path: str,
    account_name: str,
    account_key: str,
    container_name: str,
    overwrite: bool = True
) -> bool:
    """
    Convenience function to upload data to ADLS.
    
    Args:
        data: Dictionary or list to upload
        adls_path: Path in ADLS container
        account_name: Azure storage account name
        account_key: Azure storage account key
        container_name: Container name
        overwrite: Whether to overwrite existing file
        
    Returns:
        True if successful
    """
    uploader = ADLSUploader(account_name, account_key, container_name)
    return uploader.upload_json_file(data, adls_path, overwrite)


if __name__ == "__main__":
    # Test with environment variables
    import os
    from dotenv import load_dotenv
    from datetime import datetime
    
    load_dotenv()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test upload
    try:
        account_name = os.getenv("ADLS_ACCOUNT_NAME")
        account_key = os.getenv("ADLS_ACCOUNT_KEY")
        container_name = os.getenv("ADLS_CONTAINER_NAME")
        
        if account_name is None or account_key is None or container_name is None:
            raise EnvironmentError(
                "Environment variables ADLS_ACCOUNT_NAME, ADLS_ACCOUNT_KEY and ADLS_CONTAINER_NAME must be set"
            )

        uploader = ADLSUploader(
            account_name=account_name,
            account_key=account_key,
            container_name=container_name
        )
        
        # Test upload
        test_data = {
            "test": "data",
            "timestamp": datetime.now().isoformat()
        }
        
        test_path = "test/upload_test.json"
        success = uploader.upload_json_file(test_data, test_path)
        
        if success:
            print(f"\n✓ Test upload successful to {test_path}")
            
            # Check if file exists
            if uploader.file_exists(test_path):
                print(f"✓ File verified in ADLS")
                
                # Clean up
                uploader.delete_file(test_path)
                print(f"✓ Test file deleted")
        else:
            print(f"\n✗ Test upload failed")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")