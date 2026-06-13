"""
LNO-IonTransport Pipeline: Stream & File Logging Utility
"""
import logging
import os

def setup_logger(log_dir="logs", log_file="training.log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("LNO-Transport")
    logger.info("[+] Logger initialized. Systems operational.")
    return logger
