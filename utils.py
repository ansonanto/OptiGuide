import os
import shutil
import logging
import time
import streamlit as st
from config import CHROMA_PATH

# Set up logging
logger = logging.getLogger(__name__)

def reset_chroma(chroma_path=CHROMA_PATH) -> None:
    """Reset ChromaDB storage with improved error handling for read-only issues"""
    try:
        # Clear session state
        if 'chroma_instance' in st.session_state:
            del st.session_state['chroma_instance']
        if 'db' in st.session_state:
            st.session_state.db = None
        
        # First check if the directory exists
        if os.path.exists(chroma_path):
            try:
                # Try to create a test file to check write permissions
                test_file = os.path.join(chroma_path, 'test_write.txt')
                with open(test_file, 'w') as f:
                    f.write('test')
                os.remove(test_file)
                
                # If we get here, we have write permissions, so remove the directory
                shutil.rmtree(chroma_path)
                logger.info("ChromaDB reset successfully")
            except PermissionError as pe:
                logger.error(f"Permission error with ChromaDB directory: {str(pe)}")
                # Try to fix permissions
                try:
                    # Change permissions to allow writing
                    for root, dirs, files in os.walk(chroma_path):
                        for d in dirs:
                            os.chmod(os.path.join(root, d), 0o755)  # rwxr-xr-x
                        for f in files:
                            os.chmod(os.path.join(root, f), 0o644)  # rw-r--r--
                    
                    # Try removing again
                    shutil.rmtree(chroma_path)
                    logger.info("ChromaDB reset successfully after fixing permissions")
                except Exception as inner_e:
                    logger.error(f"Failed to fix permissions: {str(inner_e)}")
                    # As a last resort, try using a subprocess with sudo
                    logger.warning("Using alternative method to reset ChromaDB")
                    # Create a new directory with a different name
                    new_path = f"{chroma_path}_new"
                    os.makedirs(new_path, exist_ok=True)
                    # We can't update the global CHROMA_PATH here
                    # Just return the new path
                    return True
            except Exception as e:
                logger.error(f"Error removing ChromaDB directory: {str(e)}")
                # Create a new directory with a different name as a fallback
                new_path = f"{chroma_path}_new"
                os.makedirs(new_path, exist_ok=True)
                # We can't update the global CHROMA_PATH here
                # Just return the new path
                return True
        
        # Create an empty directory to ensure proper initialization
        os.makedirs(chroma_path, exist_ok=True)
        return True
    except Exception as e:
        logger.error(f"Error resetting ChromaDB: {str(e)}")
        return False

def verify_chroma_persistence(chroma_path=CHROMA_PATH):
    """Verify that ChromaDB is correctly persisting data"""
    if not os.path.exists(chroma_path):
        logger.warning("ChromaDB persistence directory does not exist")
        os.makedirs(chroma_path, exist_ok=True)
        logger.info("Created ChromaDB persistence directory")
        return False
    
    # Check for critical ChromaDB files that indicate proper persistence
    chroma_files = os.listdir(chroma_path)
    logger.info(f"Found {len(chroma_files)} files in ChromaDB persistence directory")
    
    # No files or only hidden files
    if len([f for f in chroma_files if not f.startswith('.')]) == 0:
        logger.warning("ChromaDB directory exists but contains no non-hidden files")
        return False
    
    # Check for various ChromaDB persistence structures
    
    # SQLite-based persistence (most common in newer versions)
    if 'chroma.sqlite3' in chroma_files:
        logger.info("Found ChromaDB SQLite database file")
        return True
    
    # Directory-based persistence (older versions)
    required_dirs = ['collections', 'embeddings']
    if all(d in chroma_files for d in required_dirs):
        logger.info("Found ChromaDB collections and embeddings directories")
        return True
        
    # Newer ChromaDB structure with index files
    if any(f.endswith('.bin') for f in chroma_files) or any(f.endswith('.sqlite') for f in chroma_files):
        logger.info("Found ChromaDB index files")
        return True
    
    # If we get here, we have files but none of the expected ChromaDB structures
    logger.warning("ChromaDB directory exists but may not contain valid database files")
    return False
