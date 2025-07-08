#!/usr/bin/env python3
"""
Wallege - Professional background removal and seamless image combination CLI tool
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image
import glob


class SeamlessBlender:
    """Advanced image blending with generative fill techniques"""
    
    def __init__(self, blend_width: int = 50):
        self.blend_width = blend_width
    
    def create_blend_mask(self, width: int, height: int, overlap_width: int) -> np.ndarray:
        """Create a gradient mask for seamless blending"""
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Create gradient from 0 to 1 over the overlap region
        for x in range(overlap_width):
            alpha = x / float(overlap_width - 1)
            mask[:, x] = alpha
        
        return mask
    
    def inpaint_seam(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Use OpenCV inpainting to fill gaps at seams"""
        # Convert mask to uint8 if needed
        if mask.dtype != np.uint8:
            mask = (mask * 255).astype(np.uint8)
        
        # Use Telea inpainting algorithm for better texture preservation
        inpainted = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
        return inpainted
    
    def blend_images(self, img1: np.ndarray, img2: np.ndarray, overlap_width: int) -> np.ndarray:
        """Combine two images with raw data concatenation - minimal processing for maximum quality"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Use maximum height to preserve resolution - resize only if necessary
        target_height = max(h1, h2)
        
        # Only resize if heights don't match, using high-quality interpolation
        if h1 != target_height:
            img1 = cv2.resize(img1, (w1, target_height), interpolation=cv2.INTER_LANCZOS4)
        if h2 != target_height:
            img2 = cv2.resize(img2, (w2, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # For maximum quality, use smaller overlap or no overlap
        if overlap_width > 0:
            # Very minimal overlap for seamless look
            overlap_width = min(overlap_width, w1 // 8, w2 // 8, 20)  # Maximum 20 pixels overlap
        
        # Achieve perfect quality preservation except in blend regions
        if overlap_width == 0:
            # Pure concatenation - zero quality loss
            result = np.concatenate([img1, img2], axis=1)
        else:
            # Strategy: Use direct array slicing to preserve original data perfectly
            overlap_start = w1 - overlap_width
            
            # Create result by concatenating non-overlapping parts + blended overlap
            # This preserves original image data perfectly in non-overlap regions
            
            # Part 1: First image up to overlap (perfect quality preservation)
            part1 = img1[:, :overlap_start]
            
            # Part 2: Blended overlap region (only this part has quality modification)
            overlap_img1 = img1[:, overlap_start:]
            overlap_img2 = img2[:, :overlap_width]
            
            # High-quality blend only in overlap - use uint16 to prevent overflow
            blended_overlap = ((overlap_img1.astype(np.uint16) + 
                              overlap_img2.astype(np.uint16)) // 2).astype(np.uint8)
            
            # Part 3: Second image after overlap (perfect quality preservation)  
            part3 = img2[:, overlap_width:]
            
            # Concatenate parts - original data is preserved perfectly everywhere except overlap
            result = np.concatenate([part1, blended_overlap, part3], axis=1)
        
        return result


class BackgroundCombiner:
    """Main orchestrator for combining multiple background images"""
    
    def __init__(self, generative_fill: bool = True, blend_width: int = 50):
        self.generative_fill = generative_fill
        self.blender = SeamlessBlender(blend_width)
    
    def load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """Load and validate images preserving full quality"""
        images = []
        supported_formats = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        
        for path in image_paths:
            path_obj = Path(path)
            if path_obj.suffix.lower() not in supported_formats:
                print(f"Warning: Unsupported format {path}, skipping...")
                continue
            
            try:
                # Load directly with OpenCV for best quality preservation
                cv_img = cv2.imread(path, cv2.IMREAD_COLOR)
                if cv_img is None:
                    print(f"Error: Could not load {path}")
                    continue
                    
                images.append(cv_img)
                print(f"Loaded: {path} ({cv_img.shape[1]}x{cv_img.shape[0]})")
            except Exception as e:
                print(f"Error loading {path}: {e}")
        
        return images
    
    def combine_horizontal(self, images: List[np.ndarray]) -> np.ndarray:
        """Combine images horizontally with seamless blending"""
        if not images:
            raise ValueError("No images provided")
        
        if len(images) == 1:
            return images[0]
        
        # Start with first image
        result = images[0]
        
        # Progressively blend each subsequent image
        for i in range(1, len(images)):
            overlap_width = min(self.blender.blend_width, result.shape[1] // 4, images[i].shape[1] // 4)
            result = self.blender.blend_images(result, images[i], overlap_width)
        
        # Apply generative fill if enabled
        if self.generative_fill:
            result = self.apply_generative_fill(result)
        
        return result
    
    def apply_generative_fill(self, image: np.ndarray) -> np.ndarray:
        """Apply generative fill to smooth out any remaining seams"""
        # Create a mask for potential seam areas (simplified approach)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to create inpainting mask
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(edges, kernel, iterations=1)
        
        # Apply inpainting
        result = self.blender.inpaint_seam(image, mask)
        return result


def generate_default_output_filename(output_dir: str = ".") -> str:
    """Generate default output filename with numbering (wall1.jpg, wall2.jpg, etc.)"""
    base_name = "wall"
    extension = ".jpg"
    counter = 1
    
    while True:
        filename = os.path.join(output_dir, f"{base_name}{counter}{extension}")
        if not os.path.exists(filename):
            return filename
        counter += 1


def is_single_directory_input(inputs: List[str]) -> bool:
    """Check if input is a single directory"""
    return len(inputs) == 1 and os.path.isdir(inputs[0])


def get_image_paths(inputs: List[str]) -> List[str]:
    """Get list of image paths from directories or file list"""
    all_image_paths = []
    
    for input_path in inputs:
        if os.path.isfile(input_path):
            all_image_paths.append(input_path)
        elif os.path.isdir(input_path):
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
            
            for ext in image_extensions:
                pattern = os.path.join(input_path, ext)
                all_image_paths.extend(glob.glob(pattern, recursive=False))
                # Also check uppercase
                pattern = os.path.join(input_path, ext.upper())
                all_image_paths.extend(glob.glob(pattern, recursive=False))
        else:
            print(f"Warning: Invalid input path: {input_path}")
    
    return sorted(all_image_paths)


def main():
    parser = argparse.ArgumentParser(
        description="Wallege - Horizontal image concatenation with generative fill",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py photos/ -o result.png
  python main.py image1.jpg image2.jpg -o panorama.png
  python main.py photos/ -o result.png --no-generative-fill
  python main.py photos/ -o result.png --blend-width 100
  python main.py photos/ -o result.png --no-blend
        """
    )
    
    parser.add_argument(
        'input',
        nargs='+',
        help='Input directory containing images or individual image files'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Output file path for the combined image (default: wall1.jpg, wall2.jpg, etc.)'
    )
    
    parser.add_argument(
        '--no-generative-fill',
        action='store_true',
        help='Disable generative fill for faster processing'
    )
    
    parser.add_argument(
        '--blend-width',
        type=int,
        default=50,
        help='Width of the blending overlap region (default: 50, use 0 for no blending)'
    )
    
    parser.add_argument(
        '--no-blend',
        action='store_true',
        help='Disable all blending - pure concatenation for maximum quality'
    )
    
    args = parser.parse_args()
    
    try:
        # Get image paths
        image_paths = get_image_paths(args.input)
        
        # Handle output argument logic
        if not args.output:
            # No output argument given
            if is_single_directory_input(args.input):
                # Input is a directory, use same directory for output
                output_dir = args.input[0]
            else:
                # Input is list of images, use current directory
                output_dir = "."
            args.output = generate_default_output_filename(output_dir)
        elif os.path.isdir(args.output):
            # Output argument is a directory, use generate_default_output_filename
            args.output = generate_default_output_filename(args.output)
        
        print(f"Output filename: {args.output}")
        
        if not image_paths:
            print("No valid images found in the input path")
            sys.exit(1)
        
        print(f"Found {len(image_paths)} images")
        
        # Initialize combiner
        blend_width = 0 if args.no_blend else args.blend_width
        combiner = BackgroundCombiner(
            generative_fill=not args.no_generative_fill and not args.no_blend,
            blend_width=blend_width
        )
        
        # Load images
        images = combiner.load_images(image_paths)
        
        if not images:
            print("No images could be loaded")
            sys.exit(1)
        
        print(f"Successfully loaded {len(images)} images")
        
        # Combine images
        print("Combining images horizontally...")
        result = combiner.combine_horizontal(images)
        
        # Save result with high quality
        output_path = Path(args.output)
        if output_path.suffix.lower() in ['.jpg', '.jpeg']:
            # Use OpenCV for JPG to preserve quality with high compression settings
            cv2.imwrite(args.output, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif output_path.suffix.lower() == '.png':
            # Use OpenCV for PNG with minimal compression
            cv2.imwrite(args.output, result, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        else:
            # For other formats, use PIL with high quality
            result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            pil_result = Image.fromarray(result_rgb)
            pil_result.save(args.output, quality=95, optimize=False)
        
        print(f"Combined image saved to: {args.output}")
        print(f"Final dimensions: {result.shape[1]}x{result.shape[0]}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()