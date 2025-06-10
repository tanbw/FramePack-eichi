"""
ConfigQueueManager CLASS DOCUMENTATION

PURPOSE:
The ConfigQueueManager is the core component that handles all config file operations
and queue processing logic. It manages the lifecycle of configuration files from
creation through queue processing to completion.

ARCHITECTURE:
- FILE-BASED QUEUE SYSTEM: Uses filesystem directories to manage queue state
- THREAD-SAFE PROCESSING: Handles concurrent access to queue operations
- IMAGE MANAGEMENT: Automatic image storage with deduplication
- ERROR RECOVERY: Robust error handling and state recovery mechanisms

DIRECTORY STRUCTURE:
configs/          - Saved configuration files (.json)
config_images/    - Permanent image storage with deduplication
queue/           - Items waiting to be processed
processing/      - Currently processing item
completed/       - Successfully processed items
error/           - Failed processing items with error details

WORKFLOW:
1. Save config → configs/
2. Queue config → copy to queue/
3. Start processing → move to processing/
4. Complete/Error → move to completed/ or error/

THREAD SAFETY:
- Queue processing runs in separate thread
- State flags prevent concurrent processing
- File operations use atomic moves where possible
"""

class ConfigQueueManager:
    """
    CORE QUEUE MANAGER: Handles all config file and queue operations
    
    INITIALIZATION PARAMETERS:
    base_path (str): Root directory for all queue-related folders
    
    INSTANCE VARIABLES:
    - Directory paths: configs_dir, queue_dir, processing_dir, completed_dir, error_dir
    - Processing state: is_processing, current_config, stop_processing
    - Threading: queue_thread for background processing
    
    THREAD SAFETY:
    Uses threading.Thread for queue processing with proper state management.
    File operations are atomic where possible to prevent corruption.
    """

    def __init__(self, base_path: str):
        """
        CONSTRUCTOR: Initialize queue manager with directory structure
        
        DIRECTORY CREATION:
        Automatically creates all necessary directories if they don't exist.
        This ensures the queue system can operate immediately after initialization.
        
        PARAMETERS:
        base_path: Root directory (typically the main application directory)
        """

    def is_gradio_temp_file(self, file_path):
        """
        UTILITY: Detect if file is in Gradio's temporary directory
        
        PURPOSE:
        Gradio stores uploaded files in temporary directories that are cleaned up
        automatically. This function detects such files so they can be copied
        to permanent storage before the temporary files are deleted.
        
        DETECTION PATTERNS:
        Checks for common Gradio temp directory patterns across platforms:
        - Windows: appdata\\local\\temp\\gradio
        - Linux/Mac: /tmp/gradio
        - Generic: any path containing 'gradio'
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        """

    def get_queue_status(self) -> Dict:
        """
        STATUS REPORTING: Get comprehensive queue status with completion order fix
        
        COMPREHENSIVE STATUS:
        Returns complete picture of queue state including:
        - Processing flags and current item
        - Queued items (waiting to process)
        - Completed items (newest first - FIXED)
        - Queue counts and metadata
        - Configs remaining count for accurate display
        
        COMPLETION ORDER FIX:
        Previously showed oldest completed items first. Now correctly shows
        newest completed items first for better user feedback.
        
        ERROR HANDLING:
        Returns error status dict if filesystem operations fail.
        
        STATUS STRUCTURE:
        {
            "is_processing": bool,
            "queued": [config_names...],
            "processing": config_name or None,
            "completed": [config_names...],  # newest first
            "queue_count": int,
            "current_config": config_name or None
        }
        
        RETURNS:
        Dict: Complete status information
        """

    def clear_queue(self) -> Tuple[bool, str]:
        """
        QUEUE CLEARING: Remove all items from queue (safety check included)
        
        SAFETY MECHANISM:
        Prevents clearing queue while processing is active to avoid data loss.
        
        ATOMIC OPERATION:
        Removes all .json files from queue directory in single operation.
        
        RETURNS:
        Tuple[bool, str]: (success, message_with_count)
        """

    def find_image_by_original_name(self, original_filename):
        """
        IMAGE RECOVERY: Find existing image by original filename pattern
        
        PURPOSE:
        When config loading fails due to missing image, this function attempts
        to recover by finding images with similar original names but different hashes.
        
        RECOVERY STRATEGY:
        - Extract clean name from original filename
        - Search for files starting with that name
        - Return first match found
        
        PARAMETERS:
        original_filename: Original filename to search for
        
        RETURNS:
        str or None: Path to found image, or None if not found
        """

    def copy_image_to_permanent_storage(self, source_path, config_name):
        """
        ADVANCED IMAGE STORAGE: Copy image with deduplication and conflict resolution
        
        ARCHITECTURE:
        This is a critical function for the config system. It ensures that all
        images referenced by configs are stored permanently and won't be lost
        when temporary files are cleaned up.
        
        FEATURES:
        1. DEDUPLICATION: Same content → same filename (hash-based)
        2. CONFLICT RESOLUTION: Different content with same name → versioning
        3. METADATA PRESERVATION: Maintains original filename information
        4. ERROR RECOVERY: Graceful handling of file system errors
        
        FILENAME STRATEGY:
        Original: "my_image.jpg"
        Stored as: "my_image_a1b2c3d4.jpg" (name + hash + extension)
        
        DEDUPLICATION LOGIC:
        - Calculate content hash
        - Check if file with same hash exists
        - If exists, reuse existing file (save storage space)
        - If not, copy with hash suffix
        
        PARAMETERS:
        source_path: Original file path (may be temporary)
        config_name: Config name for context (used in logging)
        
        RETURNS:
        Tuple[Optional[str], str]: (permanent_path or None, status_message)
        """

    def _get_relative_path(self, absolute_path: str) -> str:
        """
        UTILITY: Convert absolute path to relative path from base_path
        
        PURPOSE:
        Makes config files portable by storing paths relative to the project root.
        
        PARAMETERS:
        absolute_path: Full absolute path to convert
        
        RETURNS:
        str: Relative path from base_path, or absolute path if conversion fails
        """

    def _resolve_path(self, stored_path: str, search_dirs: List[str] = None) -> Optional[str]:
        """
        UTILITY: Resolve a stored path (relative or absolute) to actual file
        
        RESOLUTION STRATEGY:
        1. Try as relative path from base_path
        2. Try as absolute path
        3. Search by filename in provided directories
        
        PARAMETERS:
        stored_path: Path stored in config (relative or absolute)
        search_dirs: List of directories to search for filename
        
        RETURNS:
        str or None: Resolved absolute path, or None if not found
        """
        def start_queue_processing(self, process_function) -> Tuple[bool, str]:
        """
        PROCESSING STARTER: Initialize queue processing with background thread
        
        THREAD ARCHITECTURE:
        Creates background thread for queue processing to avoid blocking UI.
        Uses daemon thread to ensure clean shutdown with main application.
        
        SAFETY CHECKS:
        - Prevents multiple concurrent processing sessions
        - Validates queue has items before starting
        - Resets processing state cleanly
        
        PARAMETERS:
        process_function: Callable that processes individual config items
                         Must accept config_data dict and return bool (success)
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        
        THREAD LIFECYCLE:
        1. Validate preconditions
        2. Set processing flags
        3. Create and start daemon thread
        4. Return immediately (non-blocking)
        """

        def stop_queue_processing(self) -> Tuple[bool, str]:
        """
        PROCESSING STOPPER: Signal graceful shutdown of queue processing
        
        GRACEFUL SHUTDOWN:
        Sets stop flag to allow current item to complete before stopping.
        Does not forcefully terminate thread to avoid data corruption.
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        """

    def _has_queued_items(self) -> bool:
        """
        UTILITY: Check if queue contains items
        
        SAFE CHECKING:
        Handles directory existence and permission errors gracefully.
        
        RETURNS:
        bool: True if queue has .json files
        """

    def _get_next_queue_item(self) -> Optional[str]:
        """
        QUEUE ORDERING: Get next item from queue (FIFO - oldest first)
        
        FIFO PROCESSING:
        Uses file modification time to ensure oldest queued items process first.
        This provides predictable processing order for users.
        
        RETURNS:
        str or None: Config name of next item, or None if queue empty
        """
    def _move_to_processing(self, config_name: str) -> bool:
        """
        STATE TRANSITION: Move config from queue to processing
        
        ATOMIC OPERATION:
        Uses shutil.move for atomic file operation to prevent race conditions.
        
        PARAMETERS:
        config_name: Name of config to move
        
        RETURNS:
        bool: True if move successful
        """
    def _move_to_completed(self, config_name: str) -> bool:
        """
        COMPLETION TRANSITION: Move config from processing to completed
        
        SUCCESS HANDLING:
        Marks successful completion of config processing.
        Completed items are shown in status for user feedback.
        
        PARAMETERS:
        config_name: Name of config to mark complete
        
        RETURNS:
        bool: True if move successful
        """

    def _move_to_error(self, config_name: str, error_msg: str) -> bool:
        """
        ERROR HANDLING: Move config to error directory with error information
        
        ERROR METADATA:
        Enhances config file with error information for debugging:
        - Error message
        - Timestamp of failure
        - Original config data preserved
        
        DEBUGGING AID:
        Error configs can be inspected to understand failure causes.
        
        PARAMETERS:
        config_name: Name of failed config
        error_msg: Description of error that occurred
        
        RETURNS:
        bool: True if error handling successful
        """

    def _process_queue_worker_simple(self, process_function):
        """
        CORE PROCESSING WORKER: Main queue processing loop (simplified, robust version)
        
        ARCHITECTURE DECISION:
        This is a simplified version that focuses on reliability over complex features.
        It processes each queue item sequentially without trying to intercept
        individual generation steps, which proved problematic in earlier versions.
        
        PROCESSING LOOP:
        1. Get next queue item (FIFO order)
        2. Move to processing directory
        3. Load config data
        4. Call process function
        5. Handle success/failure appropriately
        6. Repeat until queue empty or stop signal
        
        ERROR HANDLING:
        - Individual config errors don't stop entire queue
        - Failed configs moved to error directory with details
        - Processing continues with remaining items
        - Comprehensive logging for debugging
        
        STATE MANAGEMENT:
        - Updates current_config for status display
        - Maintains processing flags correctly
        - Ensures clean shutdown on completion or error
        
        THREAD SAFETY:
        - Runs in separate thread to avoid blocking UI
        - Uses file system operations for state persistence
        - Graceful handling of stop signals
        
        PARAMETERS:
        process_function: Function that processes individual configs
                         Must accept config_data dict and return bool
        
        PERFORMANCE:
        - No complex stream interception (previous versions had issues)
        - Direct function calls for maximum compatibility
        - Minimal overhead between configs
        """

    def load_config_from_processing(self, config_name: str) -> Tuple[bool, Dict, str]:
        """
        PROCESSING LOADER: Load config file from processing directory
        
        PROCESSING CONTEXT:
        This loader is specifically for configs that are currently being processed.
        It loads from the processing directory rather than the main configs directory.
        
        PARAMETERS:
        config_name: Name of config currently in processing
        
        RETURNS:
        Tuple[bool, Dict, str]: (success, config_data, message)
        """
    # ==============================================================================
    # FUTURE ENHANCEMENT AREAS
    # ==============================================================================
    """
    MODIFICATION GUIDELINES:

    1. MAINTAINING COMPATIBILITY:
    - Keep existing public API unchanged
    - Add new features as optional parameters
    - Ensure backward compatibility with existing configs

    2. THREAD SAFETY:
    - Any new features must consider thread safety
    - File operations should remain atomic
    - State management should be consistent

    3. ERROR HANDLING:
    - All new features should have comprehensive error handling
    - Failed operations should not corrupt queue state
    - Error messages should be user-friendly

    4. TESTING:
    - New features should be thoroughly tested
    - Edge cases should be considered (empty queues, missing files, etc.)
    - Performance impact should be measured
    """

    def calculate_file_hash(self, file_path):
        """
        UTILITY: Calculate MD5 hash for file deduplication
        
        PURPOSE:
        Generates a unique hash for file content to enable deduplication.
        This prevents storing multiple copies of the same image file.
        
        ALGORITHM:
        Uses MD5 hash with chunked reading for memory efficiency.
        Returns first 12 characters for reasonable uniqueness with shorter filenames.
        
        PARAMETERS:
        file_path: Path to file to hash
        
        RETURNS:
        str: 12-character hash string, or None on error
        """

    def _init_directories(self):
        """
        INITIALIZATION: Create necessary directories if they don't exist
        
        DIRECTORY STRUCTURE:
        Creates the complete directory structure needed for queue operations.
        This ensures the system can operate immediately after initialization.
        """

    def generate_config_name(self, custom_name: str = "") -> str:
        """
        NAME GENERATION: Create unique config name with timestamp
        
        NAMING STRATEGY:
        - With custom name: "custom_name_20241225_143052"
        - Without custom name: "config_20241225_143052"
        
        SAFETY:
        Removes invalid filename characters to ensure filesystem compatibility.
        
        PARAMETERS:
        custom_name: User-provided name (optional)
        
        RETURNS:
        str: Safe, unique config name
        """

    def save_config_with_timestamp_option(self, config_name: str, image_path: str, prompt: str, 
                                        lora_settings: Dict, add_timestamp: bool = True, 
                                        other_params: Dict = None) -> Tuple[bool, str]:
        """
        PRIMARY SAVE FUNCTION: Save config with advanced image handling and timestamp control

        PORTABILITY UPDATE:
        Now saves paths as relative paths to make configs portable across systems.
        Exception: original_image_path remains absolute as it's typically a temporary file.
        
        CRITICAL FUNCTION for the entire config system. This function:
        1. Validates all inputs thoroughly
        2. Handles temporary vs permanent image storage
        3. Manages config name generation and collision detection
        4. Stores complete config data with metadata
        
        ADVANCED FEATURES:
        - Automatic image storage for Gradio temp files
        - Deduplication of identical images
        - LoRA file validation
        - Config name collision handling
        - Comprehensive metadata storage

        
        TIMESTAMP LOGIC:
        - add_timestamp=True: Always creates unique name (safe)
        - add_timestamp=False: Uses exact name (may overwrite)
        
        CONFIG DATA STRUCTURE:
        {
            "config_name": "actual_name_used",
            "timestamp": "2024-12-25T14:30:52.123456",
            "image_path": "/path/to/permanent/image.jpg",
            "original_image_path": "/path/to/original/temp/image.jpg",
            "original_filename": "user_uploaded_name.jpg",
            "prompt": "The girl dances gracefully...",
            "lora_settings": { ... },
            "other_params": { ... }
        }
        
        PARAMETERS:
        config_name: Desired config name (empty for auto-generation)
        image_path: Path to image file
        prompt: Text prompt for generation
        lora_settings: Complete LoRA configuration
        add_timestamp: Whether to add timestamp to name
        other_params: Additional parameters (reserved for future use)
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        
        CRITICAL NOTE:
        The returned message contains ONLY the clean config name for UI parsing.
        System messages are logged separately to avoid UI contamination.
        """

    def _remove_file_case_insensitive(self, directory: str, filename: str) -> bool:
        """
        UTILITY: Remove file handling case-insensitive file systems
        
        PURPOSE:
        Generic method to remove files with different casing before creating
        new ones with the desired casing. Works for any directory.
        
        PARAMETERS:
        directory: Directory path to search in
        filename: Filename to remove (without .json extension)
        
        RETURNS:
        bool: True if any file was removed
        """

    def load_config_for_editing(self, config_name: str) -> Tuple[bool, Dict, str]:
        """
        CONFIG LOADING: Load config for UI editing with image recovery
        
        EDITING MODE vs GENERATION MODE:
        This function loads configs for editing purposes, which means:
        - Missing images are allowed (can be replaced before generation)
        - LoRA files must exist (cannot be easily replaced)
        - Recovery attempts are made for missing images
        
        IMAGE RECOVERY STRATEGIES:
        1. Try original image path from config
        2. Try alternative paths stored in config metadata
        3. Search by original filename in config_images directory
        4. Allow loading with missing image (for editing/replacement)
        PORTABILITY UPDATE:
        - Now handles both relative and absolute paths for backward compatibility.
        
        RECOVERY STATUS INDICATORS:
        - "available": Image found at expected location
        - "recovered": Image found at alternative location
        - "missing": Image not found but config can still be edited
        
        PARAMETERS:
        config_name: Name of config to load
        
        RETURNS:
        Tuple[bool, Dict, str]: (success, config_data, message)
        
        CONFIG DATA ENHANCEMENT:
        Adds '_image_status' field to config data for UI feedback.
        This internal field is removed before generation.
        """

    def load_config_for_generation(self, config_name: str) -> Tuple[bool, Dict, str]:
        """
        GENERATION LOADING: Load config for video generation with strict validation
        
        GENERATION MODE REQUIREMENTS:
        - Image MUST be available (no missing images allowed)
        - All LoRA files MUST exist
        - All parameters must be valid
        
        This function is used by the queue processor and requires all assets
        to be available for successful video generation.
        
        PARAMETERS:
        config_name: Name of config to load
        
        RETURNS:
        Tuple[bool, Dict, str]: (success, config_data, message)
        """

    def config_exists(self, config_name: str) -> bool:
        """
        UTILITY: Check if a config file exists
        
        PARAMETERS:
        config_name: Config name to check
        
        RETURNS:
        bool: True if config file exists
        """

    def delete_config(self, config_name: str) -> Tuple[bool, str]:
        """
        CONFIG DELETION: Delete a config file with validation
        
        SAFETY:
        Validates file existence before deletion to provide appropriate feedback.
        
        PARAMETERS:
        config_name: Name of config to delete
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        """

    def get_available_configs(self) -> List[str]:
        """
        CONFIG DISCOVERY: Get list of available config files with validation
        
        ROBUST SCANNING:
        - Always performs fresh filesystem scan
        - Validates JSON format of each file
        - Skips corrupted files with warning
        - Returns sorted list for consistent ordering
        - Handles case-sensitive file systems properly
        
        RETURNS:
        List[str]: List of valid config names (without .json extension)
        """
        
    def queue_config(self, config_name: str) -> Tuple[bool, str]:
        """
        QUEUE OPERATION: Add config to processing queue
        
        QUEUE LOGIC:
        Copies config from configs/ directory to queue/ directory.
        Using copy (not move) preserves original config for future use.
        
        COLLISION DETECTION:
        Prevents duplicate queue entries for same config name.
        
        PARAMETERS:
        config_name: Name of config to queue
        
        RETURNS:
        Tuple[bool, str]: (success, message)
        """