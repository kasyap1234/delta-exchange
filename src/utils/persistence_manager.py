"""
Persistence Manager Module.
Handles saving and loading of the bot's state to/from a JSON file.
"""

import json
import os
import fcntl
from typing import Dict, Any, Optional
from utils.logger import log

class PersistenceManager:
    """
    Manages the persistent storage of the trading bot's state.
    Uses atomic writes to prevent data corruption.
    """
    
    def __init__(self, filepath: str = "data/bot_state.json"):
        self.filepath = filepath
        self._ensure_directory()
        
    def _ensure_directory(self) -> None:
        """Create the data directory if it doesn't exist."""
        directory = os.path.dirname(self.filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            log.info(f"Created directory: {directory}")
            
    def save_state(self, state: Dict[str, Any]) -> bool:
        """
        Save the state to the JSON file using atomic write and file locking.
        
        Args:
            state: Dictionary containing the bot's state
            
        Returns:
            True if successful
            
        Note:
            Uses fcntl.flock for cross-process locking and atomic os.replace for 
            durability. Non-serializable values are converted to strings with a warning.
        """
        temp_filepath = f"{self.filepath}.tmp"
        lock_filepath = f"{self.filepath}.lock"
        
        # Acquire lock before any file operations
        lock_file = open(lock_filepath, 'w')
        try:
            # Exclusive lock, blocking
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            
            # Serialize state
            try:
                # Try standard serialization first
                json_data = json.dumps(state, indent=4)
            except TypeError:
                # Fallback: find problematic keys and log warning
                log.warning("State contains non-serializable values. Converting to strings.")
                def custom_serializer(obj):
                    return str(obj)
                json_data = json.dumps(state, indent=4, default=custom_serializer)

            # Write to temp file
            with open(temp_filepath, 'w') as f:
                f.write(json_data)
            
            # Atomic rename (still under lock)
            os.replace(temp_filepath, self.filepath)
            log.debug(f"Bot state saved to {self.filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to save bot state: {e}")
            if os.path.exists(temp_filepath):
                try:
                    os.remove(temp_filepath)
                except Exception as cleanup_error:
                    log.error(f"Failed to cleanup temp file {temp_filepath}: {cleanup_error}")
            return False
        finally:
            # Release lock and close
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            
    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load the state from the JSON file.
        
        Returns:
            Dictionary containing the loaded state or None if file doesn't exist
        """
        if not os.path.exists(self.filepath):
            log.info(f"No persistence file found at {self.filepath}")
            return None
            
        try:
            with open(self.filepath, 'r') as f:
                state = json.load(f)
            log.info(f"Bot state loaded from {self.filepath}")
            return state
        except Exception as e:
            log.error(f"Failed to load bot state: {e}")
            return None

    def clear_state(self) -> bool:
        """Delete the persistence file."""
        try:
            if os.path.exists(self.filepath):
                os.remove(self.filepath)
                log.info(f"Cleared bot state at {self.filepath}")
            return True
        except Exception as e:
            log.error(f"Failed to clear bot state: {e}")
            return False
