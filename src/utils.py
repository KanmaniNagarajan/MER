import os
import zipfile
import logging
from config import ZIPPED_DATA_PATH, EXTRACTED_DATA_DIR, RESULTS_DIR

def extract_dataset():
    """
    Extract the zipped dataset to the specified directory.
    """
    if not os.path.exists(EXTRACTED_DATA_DIR):
        print(f"Extracting dataset to {EXTRACTED_DATA_DIR}")
        try:
            with zipfile.ZipFile(ZIPPED_DATA_PATH, 'r') as zip_ref:
                zip_ref.extractall(EXTRACTED_DATA_DIR)
            print("Dataset extraction complete")
        except Exception as e:
            print(f"Error extracting dataset: {str(e)}")
    else:
        print(f"Dataset already extracted to {EXTRACTED_DATA_DIR}")

def setup_logger(name, log_file, level=logging.INFO):
    """
    Set up a logger that writes to both console and file.
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)

    return logger

# Set up the main logger
main_logger = setup_logger('mer', os.path.join(RESULTS_DIR, 'mer.log'))
