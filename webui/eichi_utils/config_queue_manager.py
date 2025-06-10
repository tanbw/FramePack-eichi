# ==============================================================================
# CONFIG QUEUE MANAGER - CORE QUEUE PROCESSING SYSTEM
# ==============================================================================

import os
import json
import time
import shutil
import threading
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import traceback

# Load translations from JSON files
from locales.i18n_extended import (set_lang, translate)

class ConfigQueueManager:

    def __init__(self, base_path: str):

        self.base_path = base_path
        self.configs_dir = os.path.join(base_path, 'configs')
        self.queue_dir = os.path.join(base_path, 'queue')
        self.processing_dir = os.path.join(base_path, 'processing')
        self.completed_dir = os.path.join(base_path, 'completed')
        self.error_dir = os.path.join(base_path, 'error')
        
        # Queue processing state
        self.is_processing = False
        self.current_config = None
        self.queue_thread = None
        self.stop_processing = False
        
        # Initialize directories
        self._init_directories()
    
    # ==============================================================================
    # IMAGE MANAGEMENT SYSTEM
    # ==============================================================================
    def is_gradio_temp_file(self, file_path):

        if not file_path:
            return False
        
        normalized_path = os.path.normpath(file_path).lower()
        temp_patterns = [
            'appdata\\local\\temp\\gradio',
            '/tmp/gradio',
            'temp/gradio',
            '\\gradio\\',
            '/gradio/'
        ]
        
        return any(pattern in normalized_path for pattern in temp_patterns)

    def get_queue_status(self) -> Dict:

        try:
            # Get queued configs
            queued_configs = []
            if os.path.exists(self.queue_dir):
                for file in sorted(os.listdir(self.queue_dir)):
                    if file.endswith('.json'):
                        config_name = file[:-5]
                        queued_configs.append(config_name)
                        
            # Get processing config
            processing_config = None
            if os.path.exists(self.processing_dir):
                for file in os.listdir(self.processing_dir):
                    if file.endswith('.json'):
                        processing_config = file[:-5]
                        break
                        
            # Get completed configs - FIXED: Sort by completion time (newest first)
            completed_configs = []
            if os.path.exists(self.completed_dir):
                json_files = [f for f in os.listdir(self.completed_dir) if f.endswith('.json')]
                # Sort by modification time (newest first)
                json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.completed_dir, x)), reverse=True)
                
                for file in json_files:
                    config_name = file[:-5]  # Remove .json extension
                    completed_configs.append(config_name)
            
            # FIX: Calculate configs_remaining for accurate display
            configs_remaining = len(queued_configs)
                    
            return {
                "is_processing": self.is_processing,
                "queued": queued_configs,
                "processing": processing_config,
                "completed": completed_configs[:10],  # Show first 10 (newest first)
                "queue_count": len(queued_configs),
                "current_config": self.current_config,
                "configs_remaining": configs_remaining
            }
            
        except Exception as e:
            return {
                "error": translate("Error getting queue status: {0}").format(str(e)),
                "is_processing": False,
                "queued": [],
                "processing": None,
                "completed": [],
                "queue_count": 0,
                "current_config": None,
                "configs_remaining": 0
            }
        
    def clear_queue(self) -> Tuple[bool, str]:

        try:
            if self.is_processing:
                return False, translate("Cannot clear queue while processing")
                
            cleared_count = 0
            if os.path.exists(self.queue_dir):
                for file in os.listdir(self.queue_dir):
                    if file.endswith('.json'):
                        os.remove(os.path.join(self.queue_dir, file))
                        cleared_count += 1
                        
            return True, translate("Cleared {0} items from queue").format(cleared_count)
            
        except Exception as e:
            return False, translate("Error clearing queue: {0}").format(str(e))
       
    def find_image_by_original_name(self, original_filename):

        try:
            config_images_dir = os.path.join(self.base_path, 'config_images')
            
            if not os.path.exists(config_images_dir):
                return None
            
            # Extract name without extension from original
            original_name = os.path.splitext(os.path.basename(original_filename))[0]
            safe_original_name = "".join(c for c in original_name if c.isalnum() or c in ('_', '-', ' ')).strip().replace(' ', '_')
            
            # Look for files that start with the same name
            for existing_file in os.listdir(config_images_dir):
                if existing_file.startswith(f"{safe_original_name}_"):
                    return os.path.join(config_images_dir, existing_file)
            
            return None
        except Exception as e:
            print(translate("❌ Error searching for existing image: {0}").format(e))
            return None

    def copy_image_to_permanent_storage(self, source_path, config_name):

        try:
            if not source_path or not os.path.exists(source_path):
                return None, translate("Source image not found: {0}").format(source_path)
            
            # Create config images directory
            config_images_dir = os.path.join(self.base_path, 'config_images')
            os.makedirs(config_images_dir, exist_ok=True)
            
            # Get original filename and extension
            original_filename = os.path.basename(source_path)
            name_without_ext, file_extension = os.path.splitext(original_filename)
            
            # Clean the original filename
            safe_name = "".join(c for c in name_without_ext if c.isalnum() or c in ('_', '-', ' ')).strip()
            if not safe_name:
                safe_name = "image"
            safe_name = safe_name.replace(' ', '_')
            
            if not file_extension:
                file_extension = '.jpg'
            
            # Calculate content hash for deduplication
            content_hash = self.calculate_file_hash(source_path)
            if not content_hash:
                import time
                content_hash = str(int(time.time()))[-8:]
            
            # Create permanent filename: {original_name}_{hash}{extension}
            permanent_filename = f"{safe_name}_{content_hash}{file_extension}"
            permanent_path = os.path.join(config_images_dir, permanent_filename)
            
            # Check if file with same hash already exists
            existing_files = []
            for existing_file in os.listdir(config_images_dir):
                if existing_file.endswith(f"_{content_hash}{file_extension}"):
                    existing_files.append(existing_file)
            
            if existing_files:
                existing_path = os.path.join(config_images_dir, existing_files[0])
                print(translate("🔗 Reusing existing image: {0}").format(existing_files[0]))
                return existing_path, translate("Reused existing image: {0}").format(existing_files[0])
            
            # Copy file to permanent storage
            shutil.copy2(source_path, permanent_path)
            #print(translate("📁 Copied image to permanent storage: {0}").format(permanent_filename))
            
            return permanent_path, translate("Image copied to permanent storage: {0}").format(permanent_filename)
            
        except Exception as e:
            return None, translate("Error copying image: {0}").format(str(e))

    def _get_relative_path(self, absolute_path: str) -> str:

        try:
            return os.path.relpath(absolute_path, self.base_path)
        except (ValueError, TypeError):
            # Can't create relative path (e.g., different drives on Windows)
            return absolute_path

    def _resolve_path(self, stored_path: str, search_dirs: List[str] = None) -> Optional[str]:

        if not stored_path:
            return None
        
        # Strategy 1: Try as relative path from base_path
        try:
            relative_resolved = os.path.join(self.base_path, stored_path)
            if os.path.exists(relative_resolved):
                return os.path.abspath(relative_resolved)
        except:
            pass
        
        # Strategy 2: Try as absolute path
        if os.path.isabs(stored_path) and os.path.exists(stored_path):
            return stored_path
        
        # Strategy 3: ENHANCED - Search by filename in directories
        filename = os.path.basename(stored_path)
        print(translate("🔍 Searching for filename: {0}").format(filename))

        # Add default search directories if not provided
        if search_dirs is None:
            search_dirs = []

        # Always include these default directories
        default_dirs = [
            os.path.join(self.base_path, 'config_images'),
            os.path.join(self.base_path, 'lora'),
            # self.base_path,
            # os.path.join(self.base_path, 'inputs'),
            # os.path.join(self.base_path, 'outputs')
        ]

        all_dirs = search_dirs + default_dirs

        for search_dir in all_dirs:
            if os.path.exists(search_dir):
                potential_path = os.path.join(search_dir, filename)
                if os.path.exists(potential_path):
                    print(translate("✅ Found by filename: {0}").format(potential_path))
                    return os.path.abspath(potential_path)

        print(translate("❌ File not found by filename search: {0}").format(filename))
        
        return None

    # ==============================================================================
    # QUEUE PROCESSING ENGINE
    # ==============================================================================

    def start_queue_processing(self, process_function) -> Tuple[bool, str]:

        if self.is_processing:
            return False, translate("Queue processing is already running")
            
        if not self._has_queued_items():
            return False, translate("No items in queue")
            
        self.is_processing = True
        self.stop_processing = False
        
        # Start processing thread
        self.queue_thread = threading.Thread(
            target=self._process_queue_worker_simple, 
            args=(process_function,),
            daemon=True
        )
        self.queue_thread.start()
        
        return True, translate("Queue processing started")
        
    def stop_queue_processing(self) -> Tuple[bool, str]:

        if not self.is_processing:
            return False, translate("Queue processing is not running")
            
        self.stop_processing = True
        return True, translate("Queue processing will stop after current item")

    def _has_queued_items(self) -> bool:

        try:
            if not os.path.exists(self.queue_dir):
                return False
            return any(f.endswith('.json') for f in os.listdir(self.queue_dir))
        except:
            return False

    def _get_next_queue_item(self) -> Optional[str]:

        try:
            if not os.path.exists(self.queue_dir):
                return None
                
            json_files = [f for f in os.listdir(self.queue_dir) if f.endswith('.json')]
            if not json_files:
                return None
                
            # Sort by modification time (oldest first)
            json_files.sort(key=lambda x: os.path.getmtime(os.path.join(self.queue_dir, x)))
            return json_files[0][:-5]  # Remove .json extension
            
        except Exception as e:
            print(translate("Error getting next queue item: {0}").format(e))
            return None
        
    def _move_to_processing(self, config_name: str) -> bool:

        try:
            source = os.path.join(self.queue_dir, f"{config_name}.json")
            dest = os.path.join(self.processing_dir, f"{config_name}.json")
            
            if os.path.exists(source):
                shutil.move(source, dest)
                return True
            return False
        except Exception as e:
            print(translate("Error moving to processing: {0}").format(e))
            return False       

    def _move_to_completed(self, config_name: str) -> bool:

        try:
            source = os.path.join(self.processing_dir, f"{config_name}.json")
            dest = os.path.join(self.completed_dir, f"{config_name}.json")
            
            if os.path.exists(source):
                shutil.move(source, dest)
                return True
            return False
        except Exception as e:
            print(translate("Error moving to completed: {0}").format(e))
            return False

    def _move_to_error(self, config_name: str, error_msg: str) -> bool:

        try:
            source = os.path.join(self.processing_dir, f"{config_name}.json")
            dest = os.path.join(self.error_dir, f"{config_name}.json")
            
            if os.path.exists(source):
                # Load config and add error info
                with open(source, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    
                config_data['error'] = {
                    'message': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Save with error info
                with open(dest, 'w', encoding='utf-8') as f:
                    json.dump(config_data, f, indent=2, ensure_ascii=False)
                    
                # Remove from processing
                os.remove(source)
                return True
            return False
        except Exception as e:
            print(translate("Error moving to error: {0}").format(e))
            return False

    def _process_queue_worker_simple(self, process_function):

        try:
            print(translate("🔄 Queue worker thread started"))
            
            total_processed = 0
            total_errors = 0
            
            while not self.stop_processing:
                # Get next item
                config_name = self._get_next_queue_item()
                if not config_name:
                    print(translate("📭 No more items in queue"))
                    break
                    
                print(translate("🎬 Processing config: {0}").format(config_name))
                self.current_config = config_name
                
                # Move to processing
                if not self._move_to_processing(config_name):
                    print(translate("❌ Failed to move {0} to processing").format(config_name))
                    continue
                    
                try:
                    # Load config
                    success, config_data, message = self.load_config_from_processing(config_name)
                    if not success:
                        print(translate("❌ Failed to load config {0}: {1}").format(config_name, message))
                        self._move_to_error(config_name, message)
                        total_errors += 1
                        continue
                        
                    # Process the config
                    print(translate("🎯 Starting generation for: {0}").format(config_name))
                    result = process_function(config_data)
                    
                    if result:
                        # Success - move to completed
                        self._move_to_completed(config_name)
                        total_processed += 1
                        print(translate("✅ Completed processing: {0}").format(config_name))
                    else:
                        # Failed - move to error
                        self._move_to_error(config_name, translate("Processing failed"))
                        total_errors += 1
                        print(translate("❌ Failed processing: {0}").format(config_name))
                        
                except Exception as e:
                    # Error during processing
                    error_msg = translate("Processing error: {0}").format(str(e))
                    self._move_to_error(config_name, error_msg)
                    total_errors += 1
                    print(translate("❌ Error processing {0}: {1}").format(config_name, e))
                    import traceback
                    traceback.print_exc()
                    
        except Exception as e:
            print(translate("❌ Queue worker error: {0}").format(e))
            import traceback
            traceback.print_exc()
        finally:
            # Always reset processing state
            print(translate("🏁 Queue worker finishing - resetting processing state"))
            self.is_processing = False
            self.current_config = None
            self.stop_processing = False
            print(translate("✅ Queue processing stopped - Processed: {0}, Errors: {1}").format(total_processed, total_errors))

    def load_config_from_processing(self, config_name: str) -> Tuple[bool, Dict, str]:

        try:
            config_file = os.path.join(self.processing_dir, f"{config_name}.json")
            if not os.path.exists(config_file):
                return False, {}, translate("Config file not found in processing: {0}").format(config_name)
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                
            return True, config_data, translate("Config loaded successfully")
            
        except Exception as e:
            return False, {}, translate("Error loading config from processing: {0}").format(str(e))

   
    def calculate_file_hash(self, file_path):

        try:
            hasher = hashlib.md5()
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()[:12]
        except Exception as e:
            print(translate("❌ Error calculating hash for {0}: {1}").format(file_path, e))
            return None


    # ==============================================================================
    # CONFIG FILE OPERATIONS
    # ==============================================================================

    def _init_directories(self):

        for dir_path in [self.configs_dir, self.queue_dir, self.processing_dir, 
                        self.completed_dir, self.error_dir]:
            os.makedirs(dir_path, exist_ok=True)
            
    def generate_config_name(self, custom_name: str = "") -> str:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name:
            # Remove invalid filename characters
            safe_name = "".join(c for c in custom_name if c.isalnum() or c in ('_', '-', ' ')).strip()
            safe_name = safe_name.replace(' ', '_')
            return f"{safe_name}_{timestamp}"
        else:
            return f"config_{timestamp}"
                    
    def save_config_with_timestamp_option(self, config_name: str, image_path: str, prompt: str, 
                                        lora_settings: Dict, add_timestamp: bool = True, 
                                        other_params: Dict = None) -> Tuple[bool, str]:

        try:
            # Validate inputs
            if not image_path or not os.path.exists(image_path):
                return False, translate("Invalid image path: {0}").format(image_path)
                
            # Validate LoRA files if enabled
            if lora_settings.get('use_lora', False):
                lora_files = lora_settings.get('lora_files', [])
                for lora_file in lora_files:
                    if lora_file and not os.path.exists(lora_file):
                        return False, translate("LoRA file not found: {0}").format(lora_file)
            
            # Handle config name generation
            if not config_name:
                final_config_name = self.generate_config_name()
            else:
                if add_timestamp:
                    final_config_name = self.generate_config_name(config_name)
                else:
                    final_config_name = config_name
                    
                    # CASE-SENSITIVITY FIX: Use generic helper
                    self._remove_file_case_insensitive(self.configs_dir, final_config_name)
            
        # IMPROVED IMAGE STORAGE
            final_image_path = image_path
            image_storage_note = ""
            
            if self.is_gradio_temp_file(image_path):
                print(translate("🔄 Detected Gradio temp file, copying with deduplication..."))
                permanent_path, message = self.copy_image_to_permanent_storage(image_path, final_config_name)
                if permanent_path:
                    final_image_path = permanent_path
                    image_storage_note = f" ({message})"
                    print(f"✅ {message}")
                else:
                    print(translate("❌ Failed to copy image: {0}").format(message))
                    return False, translate("Failed to store image permanently: {0}").format(message)
            else:
                print(translate("📁 Using existing permanent image path: {0}").format(image_path))
            
            # PORTABILITY: Convert permanent image path to relative
            relative_image_path = self._get_relative_path(final_image_path)
            
            # Keep original_image_path as absolute since it's typically a temp file
            # that won't exist after the session anyway
            original_absolute_path = image_path
            
            # Store additional image metadata for better recovery
            original_filename = os.path.basename(image_path)
            
            # Convert LoRA paths to relative if they exist
            portable_lora_settings = lora_settings.copy()
            if portable_lora_settings.get('lora_files'):
                portable_lora_files = []
                for lora_path in portable_lora_settings['lora_files']:
                    if lora_path:
                        portable_lora_files.append(self._get_relative_path(lora_path))
                    else:
                        portable_lora_files.append(lora_path)
                portable_lora_settings['lora_files'] = portable_lora_files
            
            # Create config data with relative paths (except original_image_path)
            config_data = {
                "config_name": final_config_name,
                "timestamp": datetime.now().isoformat(),
                "image_path": relative_image_path,  # Store as relative
                "original_image_path": original_absolute_path,  # Keep as absolute (temp file)
                "original_filename": original_filename,
                "prompt": prompt,
                "lora_settings": portable_lora_settings,  # With relative paths
                "other_params": other_params or {},
                "format_version": "2.0"  # Mark new format for future reference
            }
            
            # Save config file
            config_file = os.path.join(self.configs_dir, f"{final_config_name}.json")
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # Verify the file was created with correct casing
            if os.path.exists(config_file):
                # Double-check the actual filename on disk
                actual_files = os.listdir(self.configs_dir)
                for actual in actual_files:
                    if actual.lower() == f"{final_config_name.lower()}.json":
                        if actual == f"{final_config_name}.json":
                            print(translate("✅ Config saved with correct casing: {0}").format(final_config_name))
                        else:
                            print(translate("⚠️ File system preserved different casing: {0} (requested: {1})").format(actual[:-5], final_config_name))
                        break
            
            # CRITICAL FIX: Return ONLY the clean config name
            return True, translate("Config saved: {0}").format(final_config_name)
            
        except Exception as e:
            return False, translate("Error saving config: {0}").format(str(e))

    def _remove_file_case_insensitive(self, directory: str, filename: str) -> bool:

        removed = False
        try:
            if os.path.exists(directory):
                for existing in os.listdir(directory):
                    if existing.lower() == f"{filename.lower()}.json":
                        existing_name = existing[:-5]  # Remove .json
                        if existing_name != filename:
                            # Different casing detected
                            print(translate("🔄 Detected case difference: '{0}' vs '{1}'").format(existing_name, filename))
                        file_path = os.path.join(directory, existing)
                        os.remove(file_path)
                        print(translate("🗑️ Removed file: {0}").format(existing))
                        removed = True
            return removed
        except Exception as e:
            print(translate("❌ Error removing file: {0}").format(e))
            return False
        
    def load_config_for_editing(self, config_name: str) -> Tuple[bool, Dict, str]:

        try:
            config_file = os.path.join(self.configs_dir, f"{config_name}.json")
            if not os.path.exists(config_file):
                return False, {}, translate("Config file not found: {0}").format(config_name)
                
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # PORTABILITY: Resolve image path (works with both relative and absolute)
            stored_image_path = config_data.get('image_path')
            config_images_dir = os.path.join(self.base_path, 'config_images')
            
            # Try to resolve the image path
            resolved_image_path = self._resolve_path(
                stored_image_path, 
                search_dirs=[config_images_dir, self.base_path]
            )
            
            image_status = "available"
            
            if not resolved_image_path:
                # Try recovery strategies
                recovered_path = None
                
                # Strategy 1: Try original path (it's absolute, so just check if it exists)
                original_path = config_data.get('original_image_path')
                if original_path and os.path.exists(original_path):
                    recovered_path = original_path
                    print(translate("🔄 Recovered image from original path: {0}").format(recovered_path))
                
                # Strategy 2: Search by original filename
                if not recovered_path:
                    original_filename = config_data.get('original_filename')
                    if original_filename:
                        found_path = self.find_image_by_original_name(original_filename)
                        if found_path:
                            recovered_path = found_path
                            print(translate("🔍 Found image by original name: {0}").format(found_path))
                
                if recovered_path:
                    config_data['image_path'] = recovered_path
                    resolved_image_path = recovered_path
                    image_status = "recovered"
                else:
                    # Image not found - still allow loading for editing
                    image_status = "missing"
                    print(translate("⚠️ Image missing but config loaded for editing: {0}").format(stored_image_path))
            else:
                # Update config data with resolved absolute path for processing
                config_data['image_path'] = resolved_image_path
            
            # Add image status to config data for UI feedback
            config_data['_image_status'] = image_status
            
            # PORTABILITY: Resolve LoRA file paths
            lora_settings = config_data.get('lora_settings', {})
            if lora_settings.get('use_lora', False):
                lora_files = lora_settings.get('lora_files', [])
                resolved_lora_files = []
                missing_lora = []
                
                lora_dir = os.path.join(self.base_path, 'lora')
                
                for lora_file in lora_files:
                    if lora_file:
                        resolved_path = self._resolve_path(
                            lora_file,
                            search_dirs=[lora_dir, self.base_path]
                        )
                        if resolved_path:
                            resolved_lora_files.append(resolved_path)
                        else:
                            missing_lora.append(lora_file)
                            resolved_lora_files.append(None)
                    else:
                        resolved_lora_files.append(None)
                
                if missing_lora:
                    return False, {}, translate("LoRA files not found: {0}").format(', '.join(missing_lora))
                
                # Update lora_settings with resolved paths
                lora_settings['lora_files'] = resolved_lora_files
                config_data['lora_settings'] = lora_settings
                        
            success_message = translate("Config loaded successfully")
            if image_status == "missing":
                success_message += translate(" (image missing - generation will be blocked until image is replaced)")
            elif image_status == "recovered":
                success_message += translate(" (image recovered from alternative path)")
            
            return True, config_data, success_message
            
        except Exception as e:
            return False, {}, translate("Error loading config: {0}").format(str(e))

    def load_config_for_generation(self, config_name: str) -> Tuple[bool, Dict, str]:

        try:
            success, config_data, message = self.load_config_for_editing(config_name)
            
            if not success:
                return success, config_data, message
            
            # For generation, image MUST be available
            image_status = config_data.get('_image_status', 'unknown')
            if image_status == 'missing':
                return False, {}, translate("Cannot generate video: Image file missing for config '{0}'").format(config_name)
            
            # Remove internal status before returning
            if '_image_status' in config_data:
                del config_data['_image_status']
            
            return True, config_data, translate("Config ready for generation")
            
        except Exception as e:
            return False, {}, translate("Error preparing config for generation: {0}").format(str(e))  

    def config_exists(self, config_name: str) -> bool:

        config_file = os.path.join(self.configs_dir, f"{config_name}.json")
        return os.path.exists(config_file) 

    def delete_config(self, config_name: str) -> Tuple[bool, str]:

        try:
            config_file = os.path.join(self.configs_dir, f"{config_name}.json")
            
            if not os.path.exists(config_file):
                return False, translate("Config file not found: {0}").format(config_name)
            
            # Remove the file
            os.remove(config_file)
            
            return True, translate("Config deleted: {0}").format(config_name)
            
        except Exception as e:
            return False, translate("Error deleting config: {0}").format(str(e))

    def get_available_configs(self) -> List[str]:

        try:
            configs = []
            seen_lowercase = {}  # Track lowercase versions for duplicate detection
            
            if os.path.exists(self.configs_dir):
                # Always do a fresh directory scan
                for file in sorted(os.listdir(self.configs_dir)):
                    if file.endswith('.json'):
                        config_name = file[:-5]  # Remove .json extension
                        # Verify file is readable
                        config_file = os.path.join(self.configs_dir, file)
                        try:
                            with open(config_file, 'r', encoding='utf-8') as f:
                                json.load(f)  # Basic JSON validation
                            
                            # Check for case-insensitive duplicates on Windows
                            lowercase_name = config_name.lower()
                            if lowercase_name in seen_lowercase:
                                print(translate("⚠️ Warning: Case-sensitive duplicate found: '{0}' vs '{1}'").format(config_name, seen_lowercase[lowercase_name]))
                            else:
                                seen_lowercase[lowercase_name] = config_name
                            
                            configs.append(config_name)
                        except:
                            # Skip corrupted files
                            print(translate("Warning: Skipping corrupted config file: {0}").format(file))
            return configs
        except Exception as e:
            print(translate("Error getting available configs: {0}").format(e))
            return []
    
    # ==============================================================================
    # QUEUE MANAGEMENT OPERATIONS
    # ==============================================================================

    def queue_config(self, config_name: str) -> Tuple[bool, str]:

        try:
            source_file = os.path.join(self.configs_dir, f"{config_name}.json")
            dest_file = os.path.join(self.queue_dir, f"{config_name}.json")
            
            if not os.path.exists(source_file):
                return False, translate("Config file not found: {0}").format(config_name)
            
            # CASE-SENSITIVITY FIX: Use generic helper
            self._remove_file_case_insensitive(self.queue_dir, config_name)
                
            shutil.copy2(source_file, dest_file)
            return True, translate("Config queued: {0}").format(config_name)
            
        except Exception as e:
            return False, translate("Error queueing config: {0}").format(str(e))

        

            






        
