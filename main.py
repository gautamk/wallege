#!/usr/bin/env python3
"""
Background Removal and Seamless Combination CLI Tool

A professional command-line tool for removing backgrounds, creating masks,
isolating backgrounds, and combining them into seamless panoramic images.

Author: CLI Background Tool
Version: 1.0.0
"""

import os
import sys
import cv2
import numpy as np
from rembg import remove
from PIL import Image
import glob
import argparse
from pathlib import Path
import io
from typing import List, Tuple, Optional, Union

# Try to import tkinter with fallback for headless systems
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("⚠️  tkinter not available - using fallback resolution detection")

class BackgroundProcessor:
    """Main class for background processing operations"""
    
    def __init__(self):
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
    def is_image_file(self, filepath: str) -> bool:
        """Check if file is a supported image format"""
        return Path(filepath).suffix.lower() in self.supported_extensions
    
    def get_image_files(self, path: str) -> List[str]:
        """Get all image files from a path (file or directory)"""
        path_obj = Path(path).resolve()  # Resolve to absolute path
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path does not exist: {path_obj}")
        
        # Check if we have read permissions
        if not os.access(str(path_obj), os.R_OK):
            raise PermissionError(f"No read permission for: {path_obj}")
        
        if path_obj.is_file():
            if self.is_image_file(str(path_obj)):
                return [str(path_obj)]
            else:
                raise ValueError(f"File is not a supported image format: {path_obj}")
        
        elif path_obj.is_dir():
            image_files = []
            
            try:
                # Use pathlib for cross-platform glob patterns
                for ext in self.supported_extensions:
                    # Case insensitive search using both cases
                    image_files.extend(path_obj.glob(f"*{ext}"))
                    image_files.extend(path_obj.glob(f"*{ext.upper()}"))
                
                # Convert to strings and sort
                image_files = [str(f) for f in image_files]
                image_files.sort()
                
                if not image_files:
                    raise ValueError(f"No supported images found in directory: {path_obj}")
                
                return image_files
                
            except PermissionError:
                raise PermissionError(f"No permission to read directory: {path_obj}")
        
        else:
            raise ValueError(f"Path is neither file nor directory: {path_obj}")

    def remove_background(self, image_path: str) -> Image.Image:
        """Remove background from an image using rembg library"""
        with open(image_path, 'rb') as input_file:
            input_data = input_file.read()
        
        output_data = remove(input_data)
        output_image = Image.open(io.BytesIO(output_data))
        return output_image

    def create_mask_from_removed_bg(self, removed_bg_image: Image.Image) -> np.ndarray:
        """Create a binary mask from the background-removed image"""
        img_array = np.array(removed_bg_image)
        
        if img_array.shape[2] == 4:  # RGBA
            alpha_channel = img_array[:, :, 3]
        else:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            alpha_channel = (gray > 0).astype(np.uint8) * 255
        
        mask = (alpha_channel > 0).astype(np.uint8) * 255
        return mask

    def isolate_background(self, original_image_path: str, mask: np.ndarray) -> np.ndarray:
        """Isolate the background using the mask"""
        original_img = cv2.imread(original_image_path)
        
        if original_img.shape[:2] != mask.shape:
            mask = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
        
        background_mask = cv2.bitwise_not(mask)
        background_only = cv2.bitwise_and(original_img, original_img, mask=background_mask)
        
        return background_only

class ResolutionManager:
    """Manages screen resolution detection and presets"""
    
    @staticmethod
    def get_screen_resolution() -> Tuple[int, int]:
        """Get the current screen resolution with fallback for headless systems"""
        
        # Method 1: Try tkinter (works on systems with GUI)
        if TKINTER_AVAILABLE:
            try:
                # Check if DISPLAY is available on Unix-like systems
                if os.name == 'posix' and 'DISPLAY' not in os.environ:
                    print("⚠️  No DISPLAY environment variable - using fallback resolution")
                    return 1920, 1080
                
                root = tk.Tk()
                root.withdraw()  # Hide the window
                
                screen_width = root.winfo_screenwidth()
                screen_height = root.winfo_screenheight()
                
                root.destroy()
                
                # Validate reasonable screen dimensions
                if screen_width > 0 and screen_height > 0 and screen_width <= 16384 and screen_height <= 16384:
                    return screen_width, screen_height
                else:
                    print(f"⚠️  Invalid screen dimensions detected: {screen_width}x{screen_height}")
                    return 1920, 1080
                    
            except Exception as e:
                print(f"⚠️  tkinter screen detection failed: {str(e)}")
        
        # Method 2: Try xrandr on Linux
        if os.name == 'posix':
            try:
                import subprocess
                result = subprocess.run(['xrandr'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if '*' in line and '+' in line:  # Current resolution line
                            parts = line.split()
                            for part in parts:
                                if 'x' in part and part.replace('x', '').replace('.', '').isdigit():
                                    try:
                                        w, h = part.split('x')
                                        width, height = int(float(w)), int(float(h.split('.')[0]))
                                        if width > 0 and height > 0:
                                            print(f"📺 Detected resolution via xrandr: {width}x{height}")
                                            return width, height
                                    except:
                                        continue
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass
        
        # Method 3: Try system_profiler on macOS
        if sys.platform == 'darwin':
            try:
                import subprocess
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    lines = result.stdout.split('\n')
                    for i, line in enumerate(lines):
                        if 'Resolution:' in line:
                            # Look for resolution pattern like "2560 x 1440"
                            import re
                            match = re.search(r'(\d+)\s*x\s*(\d+)', line)
                            if match:
                                width, height = int(match.group(1)), int(match.group(2))
                                print(f"📺 Detected resolution via system_profiler: {width}x{height}")
                                return width, height
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                pass
        
        # Method 4: Environment variable fallback
        if 'SCREEN_WIDTH' in os.environ and 'SCREEN_HEIGHT' in os.environ:
            try:
                width = int(os.environ['SCREEN_WIDTH'])
                height = int(os.environ['SCREEN_HEIGHT'])
                if width > 0 and height > 0:
                    print(f"📺 Using environment variables: {width}x{height}")
                    return width, height
            except ValueError:
                pass
        
        # Final fallback
        print("⚠️  Could not detect screen resolution, using Full HD (1920x1080)")
        return 1920, 1080

    @staticmethod
    def get_resolution_presets() -> dict:
        """Get common resolution presets with descriptions"""
        return {
            'auto': 'Auto-detect screen size',
            'hd': {'width': 1280, 'height': 720, 'desc': 'HD (1280x720)'},
            'fhd': {'width': 1920, 'height': 1080, 'desc': 'Full HD (1920x1080)'},
            'qhd': {'width': 2560, 'height': 1440, 'desc': '2K QHD (2560x1440)'},
            '4k': {'width': 3840, 'height': 2160, 'desc': '4K UHD (3840x2160)'},
            '5k': {'width': 5120, 'height': 2880, 'desc': '5K (5120x2880)'},
            '8k': {'width': 7680, 'height': 4320, 'desc': '8K (7680x4320)'},
            'mobile_portrait': {'width': 1080, 'height': 1920, 'desc': 'Mobile Portrait (1080x1920)'},
            'mobile_landscape': {'width': 1920, 'height': 1080, 'desc': 'Mobile Landscape (1920x1080)'},
            'ultrawide': {'width': 3440, 'height': 1440, 'desc': 'Ultrawide (3440x1440)'},
            'cinema': {'width': 4096, 'height': 2160, 'desc': 'Cinema 4K (4096x2160)'},
            'instagram_square': {'width': 1080, 'height': 1080, 'desc': 'Instagram Square (1080x1080)'},
            'instagram_story': {'width': 1080, 'height': 1920, 'desc': 'Instagram Story (1080x1920)'}
        }

    def choose_max_dimensions(self, preset: str = 'auto', custom_width: Optional[int] = None, 
                            custom_height: Optional[int] = None) -> Tuple[int, int]:
        """Choose maximum dimensions based on screen size or preset"""
        presets = self.get_resolution_presets()
        
        if custom_width and custom_height:
            return custom_width, custom_height
        
        if preset == 'auto':
            return self.get_screen_resolution()
        
        elif preset in presets and preset != 'auto':
            preset_info = presets[preset]
            return preset_info['width'], preset_info['height']
        
        else:
            print(f"⚠️  Unknown preset '{preset}', using auto-detection")
            return self.get_screen_resolution()

class SeamlessBlender:
    """Handles generative fill and seamless blending operations"""
    
    @staticmethod
    def create_blend_mask(img1: np.ndarray, img2: np.ndarray, overlap_width: int = 50, 
                         blend_type: str = 'linear') -> np.ndarray:
        """Create a blending mask for seamless transition between two images"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        if blend_type == 'linear':
            mask = np.zeros((max(h1, h2), overlap_width), dtype=np.float32)
            for i in range(overlap_width):
                mask[:, i] = i / (overlap_width - 1)
        
        elif blend_type == 'gaussian':
            mask = np.zeros((max(h1, h2), overlap_width), dtype=np.float32)
            center = overlap_width // 2
            sigma = overlap_width / 6
            for i in range(overlap_width):
                gaussian_val = np.exp(-((i - center) ** 2) / (2 * sigma ** 2))
                mask[:, i] = gaussian_val
        
        else:  # linear default
            mask = np.zeros((max(h1, h2), overlap_width), dtype=np.float32)
            for i in range(overlap_width):
                mask[:, i] = i / (overlap_width - 1)
        
        return mask

    @staticmethod
    def apply_alpha_blending(img1: np.ndarray, img2: np.ndarray, mask: np.ndarray, 
                           overlap_region: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply alpha blending in the overlap region"""
        x, y, w, h = overlap_region
        
        region1 = img1[y:y+h, x:x+w]
        region2 = img2[y:y+h, x:x+w]
        
        if mask.shape[:2] != (h, w):
            mask_resized = cv2.resize(mask, (w, h))
        else:
            mask_resized = mask
        
        if len(mask_resized.shape) == 2:
            mask_resized = cv2.merge([mask_resized, mask_resized, mask_resized])
        
        mask_resized = mask_resized.astype(np.float32) / 255.0
        
        blended_region = (region1.astype(np.float32) * (1 - mask_resized) + 
                         region2.astype(np.float32) * mask_resized)
        
        result = img1.copy()
        result[y:y+h, x:x+w] = blended_region.astype(np.uint8)
        
        return result

    @staticmethod
    def generative_fill_inpaint(image: np.ndarray, mask_region: np.ndarray, 
                              method: str = 'telea') -> np.ndarray:
        """Apply inpainting to fill gaps and create smooth transitions"""
        if len(mask_region.shape) == 3:
            mask_region = cv2.cvtColor(mask_region, cv2.COLOR_BGR2GRAY)
        
        if method == 'telea':
            result = cv2.inpaint(image, mask_region, 3, cv2.INPAINT_TELEA)
        else:  # ns
            result = cv2.inpaint(image, mask_region, 3, cv2.INPAINT_NS)
        
        return result

    def combine_horizontal_with_fill(self, images: List[np.ndarray], 
                                   use_generative_fill: bool = True) -> np.ndarray:
        """Combine images horizontally with seamless blending"""
        if not use_generative_fill:
            min_height = min(img.shape[0] for img in images)
            resized_images = []
            for img in images:
                h, w = img.shape[:2]
                new_w = int(w * min_height / h)
                resized_img = cv2.resize(img, (new_w, min_height))
                resized_images.append(resized_img)
            return np.hstack(resized_images)
        
        print("🎨 Applying generative fill for horizontal combination...")
        
        min_height = min(img.shape[0] for img in images)
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            new_w = int(w * min_height / h)
            resized_img = cv2.resize(img, (new_w, min_height))
            resized_images.append(resized_img)
        
        result = resized_images[0].copy()
        overlap_width = 100
        
        for i in range(1, len(resized_images)):
            current_img = resized_images[i]
            h, w = current_img.shape[:2]
            
            new_width = result.shape[1] + w - overlap_width
            expanded = np.zeros((h, new_width, 3), dtype=np.uint8)
            
            expanded[:, :result.shape[1]] = result
            
            overlap_start = result.shape[1] - overlap_width
            overlap_end = result.shape[1]
            
            overlap_existing = result[:, overlap_start:overlap_end]
            overlap_new = current_img[:, :overlap_width]
            
            blend_mask = self.create_blend_mask(overlap_existing, overlap_new, overlap_width, 'gaussian')
            
            blended_overlap = self.apply_alpha_blending(
                overlap_existing, overlap_new, 
                (blend_mask * 255).astype(np.uint8),
                (0, 0, overlap_width, h)
            )
            
            expanded[:, overlap_start:overlap_end] = blended_overlap
            expanded[:, overlap_end:overlap_end + w - overlap_width] = current_img[:, overlap_width:]
            
            inpaint_mask = np.zeros((h, new_width), dtype=np.uint8)
            inpaint_mask[:, overlap_start-10:overlap_end+10] = 255
            
            expanded = self.generative_fill_inpaint(expanded, inpaint_mask, 'telea')
            
            result = expanded
            print(f"   ✓ Blended image {i+1}/{len(resized_images)}")
        
        return result

    def combine_vertical_with_fill(self, images: List[np.ndarray], 
                                 use_generative_fill: bool = True) -> np.ndarray:
        """Combine images vertically with seamless blending"""
        if not use_generative_fill:
            min_width = min(img.shape[1] for img in images)
            resized_images = []
            for img in images:
                h, w = img.shape[:2]
                new_h = int(h * min_width / w)
                resized_img = cv2.resize(img, (min_width, new_h))
                resized_images.append(resized_img)
            return np.vstack(resized_images)
        
        print("🎨 Applying generative fill for vertical combination...")
        
        min_width = min(img.shape[1] for img in images)
        resized_images = []
        for img in images:
            h, w = img.shape[:2]
            new_h = int(h * min_width / w)
            resized_img = cv2.resize(img, (min_width, new_h))
            resized_images.append(resized_img)
        
        result = resized_images[0].copy()
        overlap_height = 100
        
        for i in range(1, len(resized_images)):
            current_img = resized_images[i]
            h, w = current_img.shape[:2]
            
            new_height = result.shape[0] + h - overlap_height
            expanded = np.zeros((new_height, w, 3), dtype=np.uint8)
            
            expanded[:result.shape[0], :] = result
            
            overlap_start = result.shape[0] - overlap_height
            overlap_end = result.shape[0]
            
            overlap_existing = result[overlap_start:overlap_end, :]
            overlap_new = current_img[:overlap_height, :]
            
            blend_mask = self.create_blend_mask(overlap_existing, overlap_new, overlap_height, 'gaussian')
            blend_mask = np.transpose(blend_mask, (1, 0))
            
            blended_overlap = self.apply_alpha_blending(
                overlap_existing, overlap_new, 
                (blend_mask * 255).astype(np.uint8),
                (0, 0, w, overlap_height)
            )
            
            expanded[overlap_start:overlap_end, :] = blended_overlap
            expanded[overlap_end:overlap_end + h - overlap_height, :] = current_img[overlap_height:, :]
            
            inpaint_mask = np.zeros((new_height, w), dtype=np.uint8)
            inpaint_mask[overlap_start-10:overlap_end+10, :] = 255
            
            expanded = self.generative_fill_inpaint(expanded, inpaint_mask, 'telea')
            
            result = expanded
            print(f"   ✓ Blended image {i+1}/{len(resized_images)}")
        
        return result

    def combine_grid_with_fill(self, images: List[np.ndarray], 
                             use_generative_fill: bool = True) -> np.ndarray:
        """Combine images in grid layout with seamless blending"""
        num_images = len(images)
        grid_cols = int(np.ceil(np.sqrt(num_images)))
        grid_rows = int(np.ceil(num_images / grid_cols))
        
        avg_height = int(np.mean([img.shape[0] for img in images]))
        avg_width = int(np.mean([img.shape[1] for img in images]))
        
        resized_images = []
        for img in images:
            resized_img = cv2.resize(img, (avg_width, avg_height))
            resized_images.append(resized_img)
        
        while len(resized_images) < grid_rows * grid_cols:
            black_img = np.zeros((avg_height, avg_width, 3), dtype=np.uint8)
            resized_images.append(black_img)
        
        if not use_generative_fill:
            rows = []
            for i in range(grid_rows):
                start_idx = i * grid_cols
                end_idx = start_idx + grid_cols
                row_images = resized_images[start_idx:end_idx]
                row = np.hstack(row_images)
                rows.append(row)
            return np.vstack(rows)
        
        print("🎨 Applying generative fill for grid combination...")
        
        rows = []
        for i in range(grid_rows):
            start_idx = i * grid_cols
            end_idx = start_idx + grid_cols
            row_images = resized_images[start_idx:end_idx]
            row = self.combine_horizontal_with_fill(row_images, True)
            rows.append(row)
        
        result = self.combine_vertical_with_fill(rows, True)
        return result

class BackgroundCombiner:
    """Main class for combining backgrounds with advanced options"""
    
    def __init__(self):
        self.processor = BackgroundProcessor()
        self.resolution_manager = ResolutionManager()
        self.blender = SeamlessBlender()

    def combine_backgrounds(self, background_path: str, output_path: str, 
                          arrangement: str = 'horizontal', resolution_preset: str = 'auto',
                          custom_width: Optional[int] = None, custom_height: Optional[int] = None,
                          generative_fill: bool = True, maintain_aspect_ratio: bool = True) -> bool:
        """Combine multiple background images into a single contiguous image"""
        
        try:
            # Get image files
            bg_files = self.processor.get_image_files(background_path)
            print(f"📁 Found {len(bg_files)} background images")
            
            # Get target dimensions
            max_width, max_height = self.resolution_manager.choose_max_dimensions(
                resolution_preset, custom_width, custom_height)
            print(f"🎯 Target resolution: {max_width}x{max_height}")
            
            # Load and resize images
            images = self._load_and_resize_images(bg_files, arrangement, max_width, max_height, 
                                                maintain_aspect_ratio)
            
            if not images:
                print("❌ No valid images to combine")
                return False
            
            # Combine images based on arrangement
            if arrangement == 'horizontal':
                combined = self.blender.combine_horizontal_with_fill(images, generative_fill)
            elif arrangement == 'vertical':
                combined = self.blender.combine_vertical_with_fill(images, generative_fill)
            elif arrangement == 'grid':
                combined = self.blender.combine_grid_with_fill(images, generative_fill)
            else:
                print(f"❌ Unknown arrangement: {arrangement}")
                return False
            
            # Final resize if needed
            if combined.shape[1] > max_width or combined.shape[0] > max_height:
                print(f"📐 Resizing final image to fit within {max_width}x{max_height}")
                
                if maintain_aspect_ratio:
                    scale_w = max_width / combined.shape[1]
                    scale_h = max_height / combined.shape[0]
                    scale = min(scale_w, scale_h)
                    
                    final_width = int(combined.shape[1] * scale)
                    final_height = int(combined.shape[0] * scale)
                else:
                    final_width = max_width
                    final_height = max_height
                
                combined = cv2.resize(combined, (final_width, final_height))
            
            # Ensure output directory exists with proper permissions
            output_dir = Path(output_path).parent
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = output_dir / '.write_test'
                try:
                    test_file.touch()
                    test_file.unlink()
                except PermissionError:
                    raise PermissionError(f"No write permission for output directory: {output_dir}")
                    
            except OSError as e:
                raise OSError(f"Cannot create output directory {output_dir}: {str(e)}")
            
            # Save combined image
            success = cv2.imwrite(output_path, combined)
            
            if success:
                print(f"✅ Combined image saved: {output_path}")
                print(f"📏 Final image shape: {combined.shape}")
                return True
            else:
                print(f"❌ Failed to save combined image")
                return False
                
        except Exception as e:
            print(f"❌ Error combining backgrounds: {str(e)}")
            return False

    def _load_and_resize_images(self, bg_files: List[str], arrangement: str, 
                              max_width: int, max_height: int, maintain_aspect_ratio: bool) -> List[np.ndarray]:
        """Load and resize images based on arrangement and target resolution"""
        images = []
        
        for bg_file in bg_files:
            img = cv2.imread(bg_file)
            if img is not None:
                # Resize based on arrangement and target resolution
                if arrangement == 'horizontal':
                    target_height = min(max_height, img.shape[0])
                    if maintain_aspect_ratio:
                        scale = target_height / img.shape[0]
                        target_width = int(img.shape[1] * scale)
                        max_width_per_image = max_width // len(bg_files)
                        if target_width > max_width_per_image:
                            scale = max_width_per_image / img.shape[1]
                            target_width = max_width_per_image
                            target_height = int(img.shape[0] * scale)
                    else:
                        target_width = max_width // len(bg_files)
                    
                    img = cv2.resize(img, (target_width, target_height))
                    
                elif arrangement == 'vertical':
                    target_width = min(max_width, img.shape[1])
                    if maintain_aspect_ratio:
                        scale = target_width / img.shape[1]
                        target_height = int(img.shape[0] * scale)
                        max_height_per_image = max_height // len(bg_files)
                        if target_height > max_height_per_image:
                            scale = max_height_per_image / img.shape[0]
                            target_height = max_height_per_image
                            target_width = int(img.shape[1] * scale)
                    else:
                        target_height = max_height // len(bg_files)
                    
                    img = cv2.resize(img, (target_width, target_height))
                    
                else:  # grid
                    num_images = len(bg_files)
                    grid_cols = int(np.ceil(np.sqrt(num_images)))
                    grid_rows = int(np.ceil(num_images / grid_cols))
                    
                    target_width = max_width // grid_cols
                    target_height = max_height // grid_rows
                    
                    if maintain_aspect_ratio:
                        scale_w = target_width / img.shape[1]
                        scale_h = target_height / img.shape[0]
                        scale = min(scale_w, scale_h)
                        
                        new_width = int(img.shape[1] * scale)
                        new_height = int(img.shape[0] * scale)
                        img = cv2.resize(img, (new_width, new_height))
                    else:
                        img = cv2.resize(img, (target_width, target_height))
                
                images.append(img)
                print(f"   ✓ Loaded: {Path(bg_file).name} - Shape: {img.shape}")
        
        return images

class CLI:
    """Command Line Interface for the Background Processing Tool"""
    
    def __init__(self):
        self.processor = BackgroundProcessor()
        self.combiner = BackgroundCombiner()
        self.resolution_manager = ResolutionManager()

    def create_parser(self) -> argparse.ArgumentParser:
        """Create and configure the argument parser"""
        parser = argparse.ArgumentParser(
            description="🎨 Professional Background Removal and Seamless Combination Tool",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  # Process single image
  %(prog)s process image.jpg -o output/
  
  # Process directory with 4K output
  %(prog)s process photos/ -o output/ --resolution 4k
  
  # Combine backgrounds horizontally with auto screen detection
  %(prog)s combine backgrounds/ -o panorama.png --arrangement horizontal
  
  # Create Instagram story format
  %(prog)s combine photos/ -o story.png --resolution instagram_story --arrangement vertical
  
  # Custom resolution with generative fill
  %(prog)s combine images/ -o result.png --width 2560 --height 1440 --generative-fill
  
  # Show available resolution presets
  %(prog)s list-resolutions
            """)
        
        subparsers = parser.add_subparsers(dest='command', help='Available commands')
        
        # Process command
        process_parser = subparsers.add_parser('process', help='Remove backgrounds and create masks')
        process_parser.add_argument('input', help='Input image file or directory')
        process_parser.add_argument('-o', '--output', required=True, help='Output directory')
        process_parser.add_argument('--no-combine', action='store_true', 
                                  help='Skip combining backgrounds')
        process_parser.add_argument('--arrangement', choices=['horizontal', 'vertical', 'grid'],
                                  default='horizontal', help='How to arrange combined backgrounds')
        process_parser.add_argument('--resolution', default='auto',
                                  help='Resolution preset (auto, hd, fhd, qhd, 4k, etc.)')
        process_parser.add_argument('--width', type=int, help='Custom width')
        process_parser.add_argument('--height', type=int, help='Custom height')
        process_parser.add_argument('--no-generative-fill', action='store_true',
                                  help='Disable generative fill for seamless blending (enabled by default)')
        process_parser.add_argument('--generative-fill', action='store_true',
                                  help='Enable generative fill for seamless blending (default: enabled)')
        
        # Combine command
        combine_parser = subparsers.add_parser('combine', help='Combine existing background images')
        combine_parser.add_argument('input', help='Background images file or directory')
        combine_parser.add_argument('-o', '--output', required=True, help='Output image file')
        combine_parser.add_argument('--arrangement', choices=['horizontal', 'vertical', 'grid'],
                                  default='horizontal', help='How to arrange images')
        combine_parser.add_argument('--resolution', default='auto',
                                  help='Resolution preset (auto, hd, fhd, qhd, 4k, etc.)')
        combine_parser.add_argument('--width', type=int, help='Custom width')
        combine_parser.add_argument('--height', type=int, help='Custom height')
        combine_parser.add_argument('--no-generative-fill', action='store_true',
                                  help='Disable generative fill for seamless blending (enabled by default)')
        combine_parser.add_argument('--generative-fill', action='store_true',
                                  help='Enable generative fill for seamless blending (default: enabled)')
        combine_parser.add_argument('--no-aspect-ratio', action='store_true',
                                  help='Don\'t maintain aspect ratios')
        
        # List resolutions command
        subparsers.add_parser('list-resolutions', help='Show available resolution presets')
        
        # Validate command
        subparsers.add_parser('validate', help='Validate platform compatibility')
        
        return parser

    def print_resolution_options(self):
        """Print all available resolution presets"""
        presets = self.resolution_manager.get_resolution_presets()
        
        print("\n🖥️  Available Resolution Presets:")
        print("=" * 60)
        
        # Current screen info
        screen_w, screen_h = self.resolution_manager.get_screen_resolution()
        print(f"📱 Your Screen: {screen_w}x{screen_h}")
        print()
        
        # Standard resolutions
        print("📺 Standard Resolutions:")
        standard = ['hd', 'fhd', 'qhd', '4k', '5k', '8k']
        for preset in standard:
            info = presets[preset]
            print(f"   {preset.upper():12} - {info['desc']}")
        
        print()
        
        # Special formats
        print("📱 Mobile & Social:")
        mobile = ['mobile_portrait', 'mobile_landscape', 'instagram_square', 'instagram_story']
        for preset in mobile:
            info = presets[preset]
            print(f"   {preset:15} - {info['desc']}")
        
        print()
        
        # Cinema formats
        print("🎬 Cinema & Ultrawide:")
        cinema = ['ultrawide', 'cinema']
        for preset in cinema:
            info = presets[preset]
            print(f"   {preset:12} - {info['desc']}")
        
        print("=" * 60)

    def process_images(self, args):
        """Process images: remove backgrounds, create masks, isolate backgrounds"""
        try:
            # Get image files
            image_files = self.processor.get_image_files(args.input)
            print(f"📁 Found {len(image_files)} images to process")
            
            # Create output directories with proper error handling
            output_path = Path(args.output).resolve()
            
            try:
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Test write permissions
                test_file = output_path / '.write_test'
                test_file.touch()
                test_file.unlink()
                
            except PermissionError:
                print(f"❌ No write permission for output directory: {output_path}")
                return False
            except OSError as e:
                print(f"❌ Cannot create output directory: {str(e)}")
                return False
            
            removed_bg_dir = output_path / 'removed_bg'
            masks_dir = output_path / 'masks'
            backgrounds_dir = output_path / 'backgrounds'
            
            removed_bg_dir.mkdir(exist_ok=True)
            masks_dir.mkdir(exist_ok=True)
            backgrounds_dir.mkdir(exist_ok=True)
            
            print(f"📂 Processing to: {output_path}")
            
            # Process each image
            for i, image_path in enumerate(image_files):
                try:
                    filename = Path(image_path).name
                    name_without_ext = Path(image_path).stem
                    
                    print(f"🔄 Processing {i+1}/{len(image_files)}: {filename}")
                    
                    # Step 1: Remove background
                    removed_bg = self.processor.remove_background(image_path)
                    removed_bg_path = removed_bg_dir / f"{name_without_ext}_no_bg.png"
                    removed_bg.save(str(removed_bg_path))
                    
                    # Step 2: Create mask
                    mask = self.processor.create_mask_from_removed_bg(removed_bg)
                    mask_path = masks_dir / f"{name_without_ext}_mask.png"
                    cv2.imwrite(str(mask_path), mask)
                    
                    # Step 3: Isolate background
                    background_only = self.processor.isolate_background(image_path, mask)
                    bg_path = backgrounds_dir / f"{name_without_ext}_background.png"
                    cv2.imwrite(str(bg_path), background_only)
                    
                    print(f"   ✅ Completed: {filename}")
                    
                except Exception as e:
                    print(f"   ❌ Error processing {filename}: {str(e)}")
            
            print("🎉 Processing complete!")
            
            # Combine backgrounds if requested
            if not args.no_combine:
                print("\n🎨 Combining backgrounds...")
                combined_output_path = output_path / f'combined_backgrounds_{args.arrangement}.png'
                
                success = self.combiner.combine_backgrounds(
                    str(backgrounds_dir),
                    str(combined_output_path),
                    arrangement=args.arrangement,
                    resolution_preset=args.resolution,
                    custom_width=args.width,
                    custom_height=args.height,
                    generative_fill=not args.no_generative_fill,  # Default True, disable with --no-generative-fill
                    maintain_aspect_ratio=True
                )
                
                if success:
                    print("🎉 Background combination complete!")
                else:
                    print("⚠️  Background combination failed")
            
        except Exception as e:
            print(f"❌ Error processing images: {str(e)}")
            return False
        
        return True

    def combine_backgrounds_cmd(self, args):
        """Combine background images command"""
        try:
            success = self.combiner.combine_backgrounds(
                args.input,
                args.output,
                arrangement=args.arrangement,
                resolution_preset=args.resolution,
                custom_width=args.width,
                custom_height=args.height,
                generative_fill=not args.no_generative_fill,
                maintain_aspect_ratio=not args.no_aspect_ratio
            )
            
            if success:
                print("🎉 Background combination complete!")
            else:
                print("❌ Background combination failed")
                
            return success
            
        except Exception as e:
            print(f"❌ Error combining backgrounds: {str(e)}")
            return False

    def run(self):
        """Main CLI entry point"""
        parser = self.create_parser()
        
        if len(sys.argv) == 1:
            parser.print_help()
            return
        
        args = parser.parse_args()
        
        if args.command == 'process':
            self.process_images(args)
            
        elif args.command == 'combine':
            self.combine_backgrounds_cmd(args)
            
        elif args.command == 'list-resolutions':
            self.print_resolution_options()
            
        elif args.command == 'validate':
            success = print_platform_info()
            sys.exit(0 if success else 1)
            
        else:
            parser.print_help()

def validate_platform() -> dict:
    """Validate platform compatibility and available features"""
    validation = {
        'platform': sys.platform,
        'python_version': sys.version,
        'tkinter_available': TKINTER_AVAILABLE,
        'display_available': True,
        'required_modules': {},
        'warnings': [],
        'errors': []
    }
    
    # Check if running on supported platform
    if sys.platform not in ['linux', 'darwin', 'linux2']:
        validation['warnings'].append(f"Platform {sys.platform} not explicitly tested")
    
    # Check DISPLAY on Unix systems
    if os.name == 'posix' and 'DISPLAY' not in os.environ:
        validation['display_available'] = False
        validation['warnings'].append("No DISPLAY environment variable (headless system)")
    
    # Check required modules
    required_modules = ['cv2', 'numpy', 'PIL', 'rembg', 'pathlib']
    
    for module_name in required_modules:
        try:
            if module_name == 'cv2':
                import cv2
                validation['required_modules'][module_name] = f"✅ {cv2.__version__}"
            elif module_name == 'numpy':
                import numpy
                validation['required_modules'][module_name] = f"✅ {numpy.__version__}"
            elif module_name == 'PIL':
                from PIL import Image
                validation['required_modules'][module_name] = f"✅ {Image.__version__}"
            elif module_name == 'rembg':
                import rembg
                validation['required_modules'][module_name] = "✅ Available"
            elif module_name == 'pathlib':
                from pathlib import Path
                validation['required_modules'][module_name] = "✅ Built-in"
        except ImportError as e:
            validation['required_modules'][module_name] = f"❌ Missing"
            validation['errors'].append(f"Required module {module_name} not available: {str(e)}")
    
    return validation

def print_platform_info():
    """Print platform compatibility information"""
    validation = validate_platform()
    
    print(f"\n🖥️  Platform Information:")
    print("=" * 50)
    print(f"Platform: {validation['platform']}")
    print(f"Python: {validation['python_version'].split()[0]}")
    print(f"Display Available: {'✅' if validation['display_available'] else '❌'}")
    print(f"Tkinter Available: {'✅' if validation['tkinter_available'] else '❌'}")
    
    print(f"\n📦 Dependencies:")
    for module, status in validation['required_modules'].items():
        print(f"   {module}: {status}")
    
    if validation['warnings']:
        print(f"\n⚠️  Warnings:")
        for warning in validation['warnings']:
            print(f"   • {warning}")
    
    if validation['errors']:
        print(f"\n❌ Errors:")
        for error in validation['errors']:
            print(f"   • {error}")
        return False
    
    print("\n✅ Platform validation passed!")
    return True

def main():
    """Main entry point"""
    print("🎨 Background Removal & Seamless Combination CLI Tool")
    print("=" * 55)
    
    # Validate platform on startup
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        success = print_platform_info()
        sys.exit(0 if success else 1)
    
    try:
        cli = CLI()
        cli.run()
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        print("\n💡 Try running 'python bgremove.py validate' to check your system")
        sys.exit(1)

if __name__ == "__main__":
    main()