#!/usr/bin/env python3
"""
Wallege - Professional background removal and seamless image combination CLI tool
"""

import argparse
import io
import sys
from pathlib import Path
from typing import List, Union
from PIL import Image, ImageOps, ImageFilter, ImageDraw
from rembg import remove
import numpy as np
import cv2
import math


class ImageCombiner:
    """Handles image input processing and combination orchestration"""
    
    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, debug: bool = False):
        self.debug = debug
    
    def get_images_from_input(self, input_path: Union[str, List[str]]) -> List[Path]:
        """
        Process input to get list of image files.
        
        Args:
            input_path: Either a directory path, single file path, or list of file paths
            
        Returns:
            List of Path objects for valid image files
            
        Raises:
            FileNotFoundError: If path doesn't exist
            ValueError: If no valid images found
        """
        image_paths = []
        
        if isinstance(input_path, list):
            # Handle list of file paths
            for path_str in input_path:
                path = Path(path_str)
                if not path.exists():
                    print(f"Warning: File {path} does not exist, skipping")
                    continue
                if self._is_supported_image(path):
                    image_paths.append(path)
                else:
                    print(f"Warning: Unsupported format {path.suffix}, skipping {path}")
        else:
            # Handle single path (file or directory)
            path = Path(input_path)
            if not path.exists():
                raise FileNotFoundError(f"Path does not exist: {path}")
            
            if path.is_file():
                if self._is_supported_image(path):
                    image_paths.append(path)
                else:
                    raise ValueError(f"Unsupported image format: {path.suffix}")
            elif path.is_dir():
                # Get all supported images from directory
                for file_path in path.iterdir():
                    if file_path.is_file() and self._is_supported_image(file_path):
                        image_paths.append(file_path)
            else:
                raise ValueError(f"Invalid path type: {path}")
        
        if not image_paths:
            raise ValueError("No valid images found in input")
        
        # Sort for consistent ordering
        image_paths.sort()
        return image_paths
    
    def _is_supported_image(self, path: Path) -> bool:
        """Check if file has supported image extension"""
        return path.suffix.lower() in self.SUPPORTED_FORMATS
    
    def remove_backgrounds(self, prescaled_images: List[Image.Image], debug_dir: Path = None) -> List[Image.Image]:
        """
        Remove backgrounds from list of pre-scaled images using rembg.
        
        Args:
            prescaled_images: List of pre-scaled PIL Images
            debug_dir: Optional directory to save debug output
            
        Returns:
            List of PIL Images with backgrounds removed
        """
        processed_images = []
        
        print(f"Step 1: Removing backgrounds from {len(prescaled_images)} pre-scaled images...")
        
        for i, prescaled_image in enumerate(prescaled_images, 1):
            print(f"  Processing {i}/{len(prescaled_images)}: Background removal")
            
            # Convert PIL image to bytes for rembg
            img_bytes = io.BytesIO()
            prescaled_image.save(img_bytes, format='PNG')
            input_data = img_bytes.getvalue()
            
            # Remove background using rembg
            output_data = remove(input_data)
            
            # Convert to PIL Image
            processed_image = Image.open(io.BytesIO(output_data))
            processed_images.append(processed_image)
            
            # Save debug output if requested
            if debug_dir and self.debug:
                debug_path = debug_dir / f"step1_bg_removed_{i:03d}.png"
                processed_image.save(debug_path)
                print(f"    Debug: Saved background-removed image to {debug_path}")
        
        print(f"Step 1 complete: {len(processed_images)} images processed")
        return processed_images
    
    def extract_backgrounds(self, prescaled_images: List[Image.Image], processed_images: List[Image.Image], debug_dir: Path = None) -> List[Image.Image]:
        """
        Extract backgrounds from pre-scaled images using processed images as masks.
        
        Args:
            prescaled_images: List of pre-scaled original images
            processed_images: List of background-removed images to use as masks
            debug_dir: Optional directory to save debug output
            
        Returns:
            List of PIL Images containing extracted backgrounds
        """
        background_images = []
        
        print(f"\nStep 2: Extracting backgrounds from {len(prescaled_images)} pre-scaled images...")
        
        for i, (prescaled_image, processed_image) in enumerate(zip(prescaled_images, processed_images), 1):
            print(f"  Processing {i}/{len(prescaled_images)}: Background extraction")
            
            # Ensure both images are RGBA
            original_image = prescaled_image.convert('RGBA')
            if processed_image.mode != 'RGBA':
                processed_image = processed_image.convert('RGBA')
            
            # Extract alpha channel as mask
            mask = processed_image.split()[-1]  # Get alpha channel
            
            # Invert mask to get background areas (where alpha was 0)
            inverted_mask = ImageOps.invert(mask)
            
            # Create background image by applying inverted mask
            background = Image.new('RGBA', original_image.size, (0, 0, 0, 0))
            
            # Convert images to numpy arrays for easier manipulation
            orig_array = np.array(original_image)
            mask_array = np.array(inverted_mask)
            bg_array = np.array(background)
            
            # Apply mask to extract background
            # Where mask is white (255), keep original pixel
            # Where mask is black (0), keep transparent
            for c in range(3):  # RGB channels
                bg_array[:, :, c] = orig_array[:, :, c] * (mask_array / 255.0)
            
            # Set alpha channel based on inverted mask
            bg_array[:, :, 3] = mask_array
            
            # Convert back to PIL Image
            background_image = Image.fromarray(bg_array.astype('uint8'), 'RGBA')
            background_images.append(background_image)
            
            # Save debug output if requested
            if debug_dir and self.debug:
                # Save mask
                mask_path = debug_dir / f"step2_mask_{i:03d}.png"
                mask.save(mask_path)
                
                # Save inverted mask
                inverted_mask_path = debug_dir / f"step2_inverted_mask_{i:03d}.png"
                inverted_mask.save(inverted_mask_path)
                
                # Save extracted background
                bg_path = debug_dir / f"step2_background_{i:03d}.png"
                background_image.save(bg_path)
                
                print(f"    Debug: Saved mask to {mask_path}")
                print(f"    Debug: Saved inverted mask to {inverted_mask_path}")
                print(f"    Debug: Saved background to {bg_path}")
        
        print(f"Step 2 complete: {len(background_images)} backgrounds extracted")
        return background_images
    
    def inpaint_backgrounds(self, background_images: List[Image.Image], processed_images: List[Image.Image], debug_dir: Path = None) -> List[Image.Image]:
        """
        Fill in blank spaces left by masks in background images using inpainting.
        
        Args:
            background_images: List of extracted background images with gaps
            processed_images: List of background-removed images to create inpainting masks
            debug_dir: Optional directory to save debug output
            
        Returns:
            List of PIL Images with inpainted (filled) backgrounds
        """
        inpainted_images = []
        
        print(f"\nStep 3: Inpainting {len(background_images)} background images...")
        
        for i, (bg_image, processed_image) in enumerate(zip(background_images, processed_images), 1):
            print(f"  Processing {i}/{len(background_images)}: Inpainting background gaps")
            
            # Convert PIL images to OpenCV format (BGR)
            bg_cv = cv2.cvtColor(np.array(bg_image.convert('RGB')), cv2.COLOR_RGB2BGR)
            
            # Create inpainting mask from processed image alpha channel
            if processed_image.mode != 'RGBA':
                processed_image = processed_image.convert('RGBA')
            
            # Extract alpha channel and convert to mask for inpainting
            alpha_mask = processed_image.split()[-1]
            inpaint_mask = np.array(alpha_mask)
            
            # Inpainting mask should be white (255) where we want to fill
            # Alpha channel is 255 where subject was, 0 where background was
            # So we use the alpha channel directly as inpainting mask
            
            # Apply inpainting using OpenCV
            # Using INPAINT_TELEA algorithm (fast, good for texture)
            inpainted_cv = cv2.inpaint(bg_cv, inpaint_mask, 3, cv2.INPAINT_TELEA)
            
            # Alternative: INPAINT_NS (Navier-Stokes, better for smooth areas)
            # inpainted_cv = cv2.inpaint(bg_cv, inpaint_mask, 3, cv2.INPAINT_NS)
            
            # Convert back to PIL format (RGB)
            inpainted_rgb = cv2.cvtColor(inpainted_cv, cv2.COLOR_BGR2RGB)
            inpainted_image = Image.fromarray(inpainted_rgb, 'RGB')
            inpainted_images.append(inpainted_image)
            
            # Save debug output if requested
            if debug_dir and self.debug:
                # Save inpainting mask
                mask_path = debug_dir / f"step3_inpaint_mask_{i:03d}.png"
                Image.fromarray(inpaint_mask, 'L').save(mask_path)
                
                # Save original background with gaps
                bg_path = debug_dir / f"step3_bg_with_gaps_{i:03d}.png"
                bg_image.convert('RGB').save(bg_path)
                
                # Save inpainted result
                inpainted_path = debug_dir / f"step3_inpainted_{i:03d}.png"
                inpainted_image.save(inpainted_path)
                
                print(f"    Debug: Saved inpaint mask to {mask_path}")
                print(f"    Debug: Saved original background to {bg_path}")
                print(f"    Debug: Saved inpainted result to {inpainted_path}")
        
        print(f"Step 3 complete: {len(inpainted_images)} backgrounds inpainted")
        return inpainted_images
    
    def determine_background_type(self, inpainted_images: List[Image.Image], background_type: str = 'auto') -> str:
        """
        Automatically determine the best background type or return specified type.
        
        Args:
            inpainted_images: List of inpainted background images
            background_type: Specified background type or 'auto'
            
        Returns:
            Background type to use: 'concat', 'texture', 'pattern', or 'gradient'
        """
        if background_type != 'auto':
            return background_type
        
        num_images = len(inpainted_images)
        
        # Auto-determination logic
        if num_images == 1:
            return 'texture'  # Single image works well as texture
        elif num_images <= 4:
            return 'concat'   # Small number works well concatenated
        else:
            return 'gradient'  # Many images work better as gradient
    
    def create_seamless_blend_mask(self, size: tuple, direction: str, blend_width: int = 50) -> Image.Image:
        """
        Create a gradient mask for seamless blending.
        
        Args:
            size: (width, height) of the mask
            direction: 'horizontal' or 'vertical'
            blend_width: Width of the blend zone in pixels
            
        Returns:
            Grayscale mask image for blending
        """
        mask = Image.new('L', size, 255)
        draw = ImageDraw.Draw(mask)
        
        if direction == 'horizontal':
            # Create horizontal gradient for left edge
            for x in range(min(blend_width, size[0])):
                alpha = int(255 * (x / blend_width))
                draw.line([(x, 0), (x, size[1])], fill=alpha)
        elif direction == 'vertical':
            # Create vertical gradient for top edge
            for y in range(min(blend_width, size[1])):
                alpha = int(255 * (y / blend_width))
                draw.line([(0, y), (size[0], y)], fill=alpha)
        
        return mask
    
    def create_concatenated_background(self, inpainted_images: List[Image.Image], arrangement: str, target_size: tuple) -> Image.Image:
        """
        Create background by concatenating inpainted images with seamless blending.
        
        Args:
            inpainted_images: List of inpainted background images
            arrangement: 'horizontal', 'vertical', or 'grid'
            target_size: (width, height) for output
            
        Returns:
            Combined background image with seamless transitions
        """
        if not inpainted_images:
            return Image.new('RGB', target_size, (128, 128, 128))
        
        # Blend width for seamless transitions
        blend_width = min(50, target_size[0] // 20, target_size[1] // 20)
        
        if arrangement == 'horizontal':
            # Calculate individual image size with overlap
            base_width = target_size[0] // len(inpainted_images)
            img_width = base_width + blend_width  # Extra width for blending
            img_height = target_size[1]
            
            # Create combined image
            combined = Image.new('RGB', target_size, (0, 0, 0))
            
            for i, img in enumerate(inpainted_images):
                # Resize image to fit with blend overlap
                resized = img.convert('RGB').resize((img_width, img_height), Image.Resampling.LANCZOS)
                
                # Calculate position (overlap with previous image)
                x_pos = i * base_width - (blend_width if i > 0 else 0)
                x_pos = max(0, x_pos)  # Don't go negative
                
                if i == 0:
                    # First image - paste directly
                    combined.paste(resized, (x_pos, 0))
                else:
                    # Create blend mask for left edge
                    mask = self.create_seamless_blend_mask((img_width, img_height), 'horizontal', blend_width)
                    
                    # Composite with blending
                    combined.paste(resized, (x_pos, 0), mask)
            
            return combined
            
        elif arrangement == 'vertical':
            # Calculate individual image size with overlap
            base_height = target_size[1] // len(inpainted_images)
            img_width = target_size[0]
            img_height = base_height + blend_width  # Extra height for blending
            
            # Create combined image
            combined = Image.new('RGB', target_size, (0, 0, 0))
            
            for i, img in enumerate(inpainted_images):
                # Resize image to fit with blend overlap
                resized = img.convert('RGB').resize((img_width, img_height), Image.Resampling.LANCZOS)
                
                # Calculate position (overlap with previous image)
                y_pos = i * base_height - (blend_width if i > 0 else 0)
                y_pos = max(0, y_pos)  # Don't go negative
                
                if i == 0:
                    # First image - paste directly
                    combined.paste(resized, (0, y_pos))
                else:
                    # Create blend mask for top edge
                    mask = self.create_seamless_blend_mask((img_width, img_height), 'vertical', blend_width)
                    
                    # Composite with blending
                    combined.paste(resized, (0, y_pos), mask)
            
            return combined
            
        elif arrangement == 'grid':
            # Calculate grid dimensions
            num_images = len(inpainted_images)
            grid_cols = math.ceil(math.sqrt(num_images))
            grid_rows = math.ceil(num_images / grid_cols)
            
            # Calculate individual image size with overlap
            base_width = target_size[0] // grid_cols
            base_height = target_size[1] // grid_rows
            img_width = base_width + blend_width
            img_height = base_height + blend_width
            
            # Create combined image
            combined = Image.new('RGB', target_size, (0, 0, 0))
            
            for i, img in enumerate(inpainted_images):
                row = i // grid_cols
                col = i % grid_cols
                
                # Resize image to fit with blend overlap
                resized = img.convert('RGB').resize((img_width, img_height), Image.Resampling.LANCZOS)
                
                # Calculate position with overlap
                x_pos = col * base_width - (blend_width if col > 0 else 0)
                y_pos = row * base_height - (blend_width if row > 0 else 0)
                x_pos = max(0, x_pos)
                y_pos = max(0, y_pos)
                
                if row == 0 and col == 0:
                    # Top-left image - paste directly
                    combined.paste(resized, (x_pos, y_pos))
                else:
                    # Create appropriate blend mask
                    mask = Image.new('L', (img_width, img_height), 255)
                    
                    if col > 0:
                        # Blend left edge
                        left_mask = self.create_seamless_blend_mask((img_width, img_height), 'horizontal', blend_width)
                        mask = ImageOps.multiply(mask, left_mask)
                    
                    if row > 0:
                        # Blend top edge
                        top_mask = self.create_seamless_blend_mask((img_width, img_height), 'vertical', blend_width)
                        mask = ImageOps.multiply(mask, top_mask)
                    
                    # Composite with blending
                    combined.paste(resized, (x_pos, y_pos), mask)
            
            return combined
        
        return inpainted_images[0].convert('RGB').resize(target_size, Image.Resampling.LANCZOS)
    
    def create_texture_background(self, inpainted_images: List[Image.Image], target_size: tuple) -> Image.Image:
        """
        Create seamless texture background from inpainted images.
        
        Args:
            inpainted_images: List of inpainted background images
            target_size: (width, height) for output
            
        Returns:
            Texture-based background image
        """
        if not inpainted_images:
            return Image.new('RGB', target_size, (128, 128, 128))
        
        # Use first image as base texture
        base_img = inpainted_images[0].convert('RGB')
        
        # Create tileable texture
        tile_size = min(base_img.size[0], base_img.size[1], 256)
        
        # Extract central square as tile
        center_x = base_img.size[0] // 2
        center_y = base_img.size[1] // 2
        left = center_x - tile_size // 2
        top = center_y - tile_size // 2
        tile = base_img.crop((left, top, left + tile_size, top + tile_size))
        
        # Create tiled background
        combined = Image.new('RGB', target_size, (0, 0, 0))
        
        for x in range(0, target_size[0], tile_size):
            for y in range(0, target_size[1], tile_size):
                combined.paste(tile, (x, y))
        
        return combined
    
    
    def create_gradient_background(self, inpainted_images: List[Image.Image], target_size: tuple) -> Image.Image:
        """
        Create gradient background based on dominant colors from images.
        
        Args:
            inpainted_images: List of inpainted background images
            target_size: (width, height) for output
            
        Returns:
            Gradient-based background image
        """
        if not inpainted_images:
            return Image.new('RGB', target_size, (128, 128, 128))
        
        # Extract dominant colors from images
        colors = []
        for img in inpainted_images:
            # Resize for faster processing
            small_img = img.convert('RGB').resize((50, 50), Image.Resampling.LANCZOS)
            pixels = list(small_img.getdata())
            
            # Calculate average color
            avg_r = sum(p[0] for p in pixels) // len(pixels)
            avg_g = sum(p[1] for p in pixels) // len(pixels)
            avg_b = sum(p[2] for p in pixels) // len(pixels)
            colors.append((avg_r, avg_g, avg_b))
        
        # Create gradient between colors
        combined = Image.new('RGB', target_size, (0, 0, 0))
        draw = ImageDraw.Draw(combined)
        
        if len(colors) == 1:
            # Single color with slight gradient
            color = colors[0]
            for y in range(target_size[1]):
                brightness = 0.7 + 0.3 * (y / target_size[1])
                line_color = tuple(int(c * brightness) for c in color)
                draw.line([(0, y), (target_size[0], y)], fill=line_color)
        else:
            # Multi-color gradient
            for y in range(target_size[1]):
                progress = y / target_size[1]
                color_index = progress * (len(colors) - 1)
                color_idx = int(color_index)
                color_blend = color_index - color_idx
                
                if color_idx >= len(colors) - 1:
                    line_color = colors[-1]
                else:
                    # Blend between two colors
                    c1 = colors[color_idx]
                    c2 = colors[color_idx + 1]
                    line_color = tuple(
                        int(c1[i] * (1 - color_blend) + c2[i] * color_blend)
                        for i in range(3)
                    )
                
                draw.line([(0, y), (target_size[0], y)], fill=line_color)
        
        return combined
    
    def create_combined_background(self, inpainted_images: List[Image.Image], background_type: str, arrangement: str, target_size: tuple, debug_dir: Path = None) -> Image.Image:
        """
        Create combined background using specified type and parameters.
        
        Args:
            inpainted_images: List of inpainted background images
            background_type: Type of background to create
            arrangement: Arrangement for concatenation mode
            target_size: (width, height) for output
            debug_dir: Optional directory to save debug output
            
        Returns:
            Combined background image
        """
        print(f"\nStep 4: Creating {background_type} background ({target_size[0]}x{target_size[1]})...")
        
        # Determine background type
        actual_type = self.determine_background_type(inpainted_images, background_type)
        if actual_type != background_type:
            print(f"  Auto-determined type: {actual_type}")
        
        # Create background based on type
        if actual_type == 'concat':
            combined = self.create_concatenated_background(inpainted_images, arrangement, target_size)
        elif actual_type == 'texture':
            combined = self.create_texture_background(inpainted_images, target_size)
        elif actual_type == 'gradient':
            combined = self.create_gradient_background(inpainted_images, target_size)
        else:
            # Fallback to concatenation
            combined = self.create_concatenated_background(inpainted_images, arrangement, target_size)
        
        # Save debug output if requested
        if debug_dir and self.debug:
            debug_path = debug_dir / f"step4_combined_{actual_type}_background.png"
            combined.save(debug_path)
            print(f"  Debug: Saved combined background to {debug_path}")
        
        print(f"Step 4 complete: {actual_type} background created")
        return combined
    
    def get_subject_bounding_box(self, foreground_image: Image.Image) -> tuple:
        """
        Get bounding box of the subject in a foreground image (ignoring transparent pixels).
        
        Args:
            foreground_image: PIL Image with transparent background
            
        Returns:
            (left, top, right, bottom) bounding box of the subject
        """
        if foreground_image.mode != 'RGBA':
            foreground_image = foreground_image.convert('RGBA')
        
        # Get alpha channel
        alpha = foreground_image.split()[-1]
        
        # Find bounding box of non-transparent pixels
        bbox = alpha.getbbox()
        
        # Return bbox or full image bounds if no transparency found
        if bbox is None:
            return (0, 0, foreground_image.size[0], foreground_image.size[1])
        
        return bbox
    
    def calculate_subject_positions_and_scale(self, foreground_images: List[Image.Image], background_size: tuple, arrangement: str = 'horizontal') -> List[tuple]:
        """
        Calculate evenly spaced positions and optimal scaling for subjects to fit background.
        
        Args:
            foreground_images: List of foreground images with subjects
            background_size: (width, height) of the background
            arrangement: 'horizontal' or 'vertical' spacing
            
        Returns:
            List of (scaled_image, x, y) tuples for each subject
        """
        if not foreground_images:
            return []
        
        results = []
        subject_boxes = []
        
        # Get bounding boxes for all subjects
        for img in foreground_images:
            bbox = self.get_subject_bounding_box(img)
            subject_width = bbox[2] - bbox[0]
            subject_height = bbox[3] - bbox[1]
            subject_boxes.append((bbox, subject_width, subject_height))
        
        # Calculate optimal scaling to fit all subjects
        if arrangement == 'horizontal':
            # For horizontal arrangement, scale to fit height and distribute width
            max_height = max(box[2] for box in subject_boxes)  # Max subject height
            available_height = background_size[1] * 0.8  # Use 80% of background height
            height_scale = available_height / max_height if max_height > 0 else 1.0
            
            # Calculate total width after scaling
            total_scaled_width = sum(box[1] * height_scale for box in subject_boxes)
            available_width = background_size[0] * 0.9  # Use 90% of background width
            
            # If total width exceeds available width, scale down further
            if total_scaled_width > available_width:
                width_scale = available_width / total_scaled_width
                final_scale = height_scale * width_scale
            else:
                final_scale = height_scale
            
            # Calculate spacing
            scaled_widths = [box[1] * final_scale for box in subject_boxes]
            total_width = sum(scaled_widths)
            
            if len(foreground_images) > 1:
                spacing = (background_size[0] - total_width) / (len(foreground_images) + 1)
            else:
                spacing = (background_size[0] - total_width) / 2
            
            current_x = spacing
            
            for img, (bbox, subject_width, subject_height) in zip(foreground_images, subject_boxes):
                # Scale the image
                scaled_width = int(img.size[0] * final_scale)
                scaled_height = int(img.size[1] * final_scale)
                scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                
                # Calculate position
                scaled_bbox = self.get_subject_bounding_box(scaled_img)
                scaled_subject_width = scaled_bbox[2] - scaled_bbox[0]
                scaled_subject_height = scaled_bbox[3] - scaled_bbox[1]
                
                x = current_x - scaled_bbox[0]
                y = (background_size[1] - scaled_subject_height) // 2 - scaled_bbox[1]
                
                results.append((scaled_img, int(x), int(y)))
                current_x += scaled_subject_width + spacing
                
        elif arrangement == 'vertical':
            # For vertical arrangement, scale to fit width and distribute height
            max_width = max(box[1] for box in subject_boxes)  # Max subject width
            available_width = background_size[0] * 0.8  # Use 80% of background width
            width_scale = available_width / max_width if max_width > 0 else 1.0
            
            # Calculate total height after scaling
            total_scaled_height = sum(box[2] * width_scale for box in subject_boxes)
            available_height = background_size[1] * 0.9  # Use 90% of background height
            
            # If total height exceeds available height, scale down further
            if total_scaled_height > available_height:
                height_scale = available_height / total_scaled_height
                final_scale = width_scale * height_scale
            else:
                final_scale = width_scale
            
            # Calculate spacing
            scaled_heights = [box[2] * final_scale for box in subject_boxes]
            total_height = sum(scaled_heights)
            
            if len(foreground_images) > 1:
                spacing = (background_size[1] - total_height) / (len(foreground_images) + 1)
            else:
                spacing = (background_size[1] - total_height) / 2
            
            current_y = spacing
            
            for img, (bbox, subject_width, subject_height) in zip(foreground_images, subject_boxes):
                # Scale the image
                scaled_width = int(img.size[0] * final_scale)
                scaled_height = int(img.size[1] * final_scale)
                scaled_img = img.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
                
                # Calculate position
                scaled_bbox = self.get_subject_bounding_box(scaled_img)
                scaled_subject_width = scaled_bbox[2] - scaled_bbox[0]
                scaled_subject_height = scaled_bbox[3] - scaled_bbox[1]
                
                x = (background_size[0] - scaled_subject_width) // 2 - scaled_bbox[0]
                y = current_y - scaled_bbox[1]
                
                results.append((scaled_img, int(x), int(y)))
                current_y += scaled_subject_height + spacing
        
        else:
            # Default to horizontal if arrangement not recognized
            return self.calculate_subject_positions_and_scale(foreground_images, background_size, 'horizontal')
        
        return results
    
    def compose_final_image(self, background: Image.Image, foreground_images: List[Image.Image], arrangement: str = 'horizontal', debug_dir: Path = None) -> Image.Image:
        """
        Compose final image by placing scaled foreground subjects on background with even spacing.
        
        Args:
            background: Combined background image
            foreground_images: List of foreground images with subjects
            arrangement: How to arrange subjects ('horizontal' or 'vertical')
            debug_dir: Optional directory to save debug output
            
        Returns:
            Final composite image
        """
        print(f"\nStep 5: Composing final image with {len(foreground_images)} subjects...")
        
        if not foreground_images:
            return background
        
        # Create final composite
        final_image = background.copy()
        
        # Calculate positions and scaling for subjects
        subject_data = self.calculate_subject_positions_and_scale(foreground_images, background.size, arrangement)
        
        # Composite each subject onto the background
        for i, (scaled_img, x, y) in enumerate(subject_data):
            print(f"  Placing subject {i+1}/{len(foreground_images)} at position ({x}, {y}) with size {scaled_img.size}")
            
            # Ensure foreground image has alpha channel
            if scaled_img.mode != 'RGBA':
                scaled_img = scaled_img.convert('RGBA')
            
            # Paste foreground onto final image using alpha channel as mask
            final_image.paste(scaled_img, (x, y), scaled_img)
            
            # Save debug output if requested
            if debug_dir and self.debug:
                # Save individual subject bounding box visualization
                debug_img = scaled_img.copy()
                bbox = self.get_subject_bounding_box(scaled_img)
                draw = ImageDraw.Draw(debug_img)
                draw.rectangle(bbox, outline=(255, 0, 0, 255), width=2)
                
                debug_path = debug_dir / f"step5_subject_{i+1}_scaled_bbox.png"
                debug_img.save(debug_path)
                print(f"    Debug: Saved scaled subject bbox to {debug_path}")
        
        # Save debug output for final composite
        if debug_dir and self.debug:
            debug_path = debug_dir / "step5_final_composite.png"
            final_image.save(debug_path)
            print(f"  Debug: Saved final composite to {debug_path}")
        
        print(f"Step 5 complete: Final composite created with {arrangement} arrangement")
        return final_image
    
    def resize_to_fit(self, image: Image.Image, target_width: int = None, target_height: int = None, debug_dir: Path = None) -> Image.Image:
        """
        Resize image to fit specified width or height while maintaining aspect ratio.
        
        Args:
            image: Input image to resize
            target_width: Target width (if specified)
            target_height: Target height (if specified)
            debug_dir: Optional directory to save debug output
            
        Returns:
            Resized image maintaining aspect ratio
        """
        print(f"\nStep 6: Resizing final composite to fit target dimensions...")
        
        current_width, current_height = image.size
        current_aspect_ratio = current_width / current_height
        
        if target_width and target_height:
            # Both specified - use as-is (shouldn't happen in this context)
            new_width, new_height = target_width, target_height
        elif target_width:
            # Width specified - calculate height maintaining aspect ratio
            new_width = target_width
            new_height = int(target_width / current_aspect_ratio)
            print(f"  Resizing to width {new_width}, calculated height {new_height}")
        elif target_height:
            # Height specified - calculate width maintaining aspect ratio
            new_height = target_height
            new_width = int(target_height * current_aspect_ratio)
            print(f"  Resizing to calculated width {new_width}, height {new_height}")
        else:
            # Neither specified - return original
            return image
        
        # Resize image
        resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Save debug output if requested
        if debug_dir and self.debug:
            debug_path = debug_dir / f"step6_resized_{new_width}x{new_height}.png"
            resized_image.save(debug_path)
            print(f"  Debug: Saved resized image to {debug_path}")
        
        print(f"Step 6 complete: Resized from {current_width}x{current_height} to {new_width}x{new_height}")
        return resized_image
    
    def prescale_images(self, image_paths: List[Path], target_size: tuple, debug_dir: Path = None) -> List[Image.Image]:
        """
        Pre-scale input images to fit within final composite size for optimal processing.
        
        Args:
            image_paths: List of input image file paths
            target_size: (width, height) of final composite
            debug_dir: Optional directory to save debug output
            
        Returns:
            List of pre-scaled PIL Images
        """
        print(f"\nStep 0: Pre-scaling {len(image_paths)} images to fit within {target_size[0]}x{target_size[1]}...")
        
        prescaled_images = []
        
        # Calculate maximum size per image based on arrangement and number of images
        num_images = len(image_paths)
        if num_images <= 2:
            # For 1-2 images, allow them to be larger
            max_width = target_size[0] // 2
            max_height = target_size[1] // 2
        elif num_images <= 4:
            # For 3-4 images, medium size
            max_width = target_size[0] // 3
            max_height = target_size[1] // 3
        else:
            # For many images, smaller size
            max_width = target_size[0] // 4
            max_height = target_size[1] // 4
        
        for i, image_path in enumerate(image_paths, 1):
            print(f"  Scaling {i}/{num_images}: {image_path.name}")
            
            # Load image
            original_image = Image.open(image_path)
            original_width, original_height = original_image.size
            
            # Calculate scaling factor to fit within max dimensions
            width_scale = max_width / original_width
            height_scale = max_height / original_height
            scale_factor = min(width_scale, height_scale, 1.0)  # Don't upscale
            
            if scale_factor < 1.0:
                # Scale down the image
                new_width = int(original_width * scale_factor)
                new_height = int(original_height * scale_factor)
                scaled_image = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                print(f"    Scaled from {original_width}x{original_height} to {new_width}x{new_height} (factor: {scale_factor:.2f})")
            else:
                # Keep original size
                scaled_image = original_image
                print(f"    Kept original size {original_width}x{original_height}")
            
            prescaled_images.append(scaled_image)
            
            # Save debug output if requested
            if debug_dir and self.debug:
                debug_path = debug_dir / f"step0_prescaled_{i:03d}_{image_path.stem}.png"
                scaled_image.save(debug_path)
                print(f"    Debug: Saved pre-scaled image to {debug_path}")
        
        print(f"Step 0 complete: {len(prescaled_images)} images pre-scaled")
        return prescaled_images
    
    def combine_images(self, image_paths: List[Path], output_path: str, **kwargs) -> None:
        """
        Combine multiple images into a single composite.
        
        Args:
            image_paths: List of Path objects for input images
            output_path: Output file path for combined image
            **kwargs: Additional parameters for combination process
        """
        output_dir = Path(output_path).parent
        debug_dir = None
        
        if self.debug:
            debug_dir = output_dir / "debug"
            debug_dir.mkdir(exist_ok=True)
            print(f"Debug mode enabled: outputs will be saved to {debug_dir}")
        
        # Determine target size with proper fallbacks (needed for pre-scaling)
        specified_width = kwargs.get('width')
        specified_height = kwargs.get('height')
        
        # Default size for composition
        default_width, default_height = 1920, 1080
        
        # Handle resolution parameter if provided
        resolution = kwargs.get('resolution')
        if resolution:
            if resolution.lower() == 'hd':
                default_width, default_height = 1280, 720
            elif resolution.lower() == 'fhd' or resolution.lower() == '1080p':
                default_width, default_height = 1920, 1080
            elif resolution.lower() == 'qhd' or resolution.lower() == '1440p':
                default_width, default_height = 2560, 1440
            elif resolution.lower() == '4k':
                default_width, default_height = 3840, 2160
            elif 'x' in resolution.lower():
                # Parse custom resolution like "1920x1080"
                try:
                    w, h = resolution.lower().split('x')
                    default_width, default_height = int(w), int(h)
                except ValueError:
                    print(f"Warning: Invalid resolution format '{resolution}', using defaults")
        
        # Determine working size for composition (use defaults if single dimension specified)
        if specified_width and specified_height:
            # Both dimensions specified - use as-is
            target_width, target_height = specified_width, specified_height
            final_resize_needed = False
        elif specified_width or specified_height:
            # Only one dimension specified - use defaults for composition, resize later
            target_width, target_height = default_width, default_height
            final_resize_needed = True
        else:
            # No dimensions specified - use defaults
            target_width, target_height = default_width, default_height
            final_resize_needed = False
        
        target_size = (target_width, target_height)
        
        # Step 0: Pre-scale images to fit within final composite size
        prescaled_images = self.prescale_images(image_paths, target_size, debug_dir)
        
        # Step 1: Remove backgrounds
        processed_images = self.remove_backgrounds(prescaled_images, debug_dir)
        
        # Step 2: Extract backgrounds using masks
        background_images = self.extract_backgrounds(prescaled_images, processed_images, debug_dir)
        
        # Step 3: Inpaint backgrounds to fill gaps
        inpainted_images = self.inpaint_backgrounds(background_images, processed_images, debug_dir)
        
        # Step 4: Create combined background
        combined_background = self.create_combined_background(
            inpainted_images,
            kwargs.get('background_type', 'auto'),
            kwargs.get('arrangement', 'horizontal'),
            target_size,
            debug_dir
        )
        
        # Step 5: Compose final image with subjects on background
        final_composite = self.compose_final_image(
            combined_background,
            processed_images,  # Use background-removed images (foreground subjects)
            kwargs.get('arrangement', 'horizontal'),
            debug_dir
        )
        
        # Step 6: Resize final composite if only one dimension was specified
        if final_resize_needed:
            final_composite = self.resize_to_fit(final_composite, specified_width, specified_height, debug_dir)
        
        # Save final result
        final_composite.save(output_path)
        print(f"\nFinal composite saved to {output_path}")


def setup_argparse() -> argparse.ArgumentParser:
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Wallege - Professional background removal and image combination",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Combine command
    combine_parser = subparsers.add_parser(
        'combine', 
        help='Combine multiple images into a composite'
    )
    combine_parser.add_argument(
        'input',
        help='Input directory or list of image files'
    )
    combine_parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output file path for combined image'
    )
    combine_parser.add_argument(
        '--arrangement',
        choices=['horizontal', 'vertical', 'grid'],
        default='horizontal',
        help='How to arrange images (default: horizontal)'
    )
    combine_parser.add_argument(
        '--resolution',
        help='Target resolution (e.g., "1920x1080", "4k", "hd")'
    )
    combine_parser.add_argument(
        '--width',
        type=int,
        help='Custom output width in pixels'
    )
    combine_parser.add_argument(
        '--height', 
        type=int,
        help='Custom output height in pixels'
    )
    combine_parser.add_argument(
        '--generative-fill',
        action='store_true',
        help='Use generative fill for seamless blending'
    )
    combine_parser.add_argument(
        '--no-generative-fill',
        action='store_true', 
        help='Disable generative fill for faster processing'
    )
    combine_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode with intermediate output files'
    )
    combine_parser.add_argument(
        '--background-type',
        choices=['auto', 'concat', 'texture', 'gradient'],
        default='auto',
        help='Type of background to generate (default: auto)'
    )
    
    return parser


def main():
    """Main entry point"""
    parser = setup_argparse()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        debug_mode = getattr(args, 'debug', False)
        combiner = ImageCombiner(debug=debug_mode)
        
        if args.command == 'combine':
            # Get image files from input
            image_paths = combiner.get_images_from_input(args.input)
            
            # Prepare combination options
            options = {
                'arrangement': args.arrangement,
                'resolution': args.resolution,
                'width': args.width,
                'height': args.height,
                'generative_fill': args.generative_fill and not args.no_generative_fill,
                'background_type': args.background_type,
                'debug': args.debug
            }
            
            # Combine images
            combiner.combine_images(image_paths, args.output, **options)
            
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())