import os
import logging
from datetime import datetime

class Logger:
    def __init__(self, file_name):
        self.file_name = file_name
        self.log_dir = os.path.join('data', 'logs')
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, f'{file_name}.log')

        # Clear the log file if it exists
        if os.path.exists(self.log_file):
            open(self.log_file, 'w').close()

        # Set up logging
        self.logger = logging.getLogger(file_name)
        self.logger.setLevel(logging.DEBUG)

        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.DEBUG)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
        self.logger.info(f"Logger initialized for {file_name}. Previous logs cleared.")
        print(f"Logger initialized for {file_name}. Previous logs cleared.")

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)