# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wallege is a Python CLI tool for professional background removal and seamless image combination. It uses machine learning (rembg with ONNX runtime) to remove backgrounds from images, creates composite backgrounds, and intelligently places subjects with even spacing for professional results.

### Key Components

- **ImageCombiner**: Main orchestrator handling the complete 6-step pipeline
- **Background Removal**: ML-powered segmentation using rembg library
- **Background Extraction**: Inverse masking to isolate original backgrounds
- **Inpainting**: OpenCV-based gap filling for seamless backgrounds
- **Background Combination**: Multiple generation modes (concat, texture, gradient)
- **Subject Placement**: Intelligent positioning with bounding box detection
- **CLI**: Command-line interface with comprehensive options

## Development Commands

### Package Management
```bash
# Install dependencies using uv (modern Python package manager)
uv sync

# Run the tool
python main.py <command>
```

### Core Commands
```bash
# Basic combination with auto background type
python main.py combine photos/ -o result.png

# Debug mode with intermediate outputs
python main.py combine photos/ -o result.png --debug

# Specific background types
python main.py combine photos/ -o result.png --background-type concat --arrangement horizontal
python main.py combine photos/ -o result.png --background-type texture
python main.py combine photos/ -o result.png --background-type gradient

# Resolution control
python main.py combine photos/ -o result.png --resolution 4k
python main.py combine photos/ -o result.png --width 3840 --height 2160
python main.py combine photos/ -o result.png --width 2560  # Height calculated automatically

# Arrangement options (affects concat mode and subject placement)
python main.py combine photos/ -o result.png --arrangement horizontal
python main.py combine photos/ -o result.png --arrangement vertical
python main.py combine photos/ -o result.png --arrangement grid

# Combined examples
python main.py combine photos/ -o instagram.png --width 1080 --background-type gradient --debug
```

## Code Architecture

### Core Processing Pipeline
1. **Background Removal**: ML-based segmentation using rembg to create transparent foreground subjects
2. **Background Extraction**: Inverse masking to isolate original backgrounds where subjects were removed
3. **Inpainting**: OpenCV algorithms (Telea/NS) fill gaps left by removed subjects for seamless backgrounds
4. **Background Combination**: Combine multiple backgrounds using concat, texture, or gradient modes
5. **Subject Placement**: Intelligent positioning based on actual subject bounding boxes (ignoring transparency)
6. **Final Resize**: Aspect ratio-preserving scaling when single dimension specified

### Key Technical Details

- **Supported formats**: JPG, JPEG, PNG, BMP, TIFF, WEBP (case-insensitive)
- **Background types**: 
  - `concat`: Seamless concatenation with gradient blending transitions
  - `texture`: Tileable texture generation from first image
  - `gradient`: Color-based gradients extracted from dominant colors
  - `auto`: Intelligent type selection (1 image→texture, ≤4→concat, >4→gradient)
- **Inpainting algorithms**: OpenCV Telea (fast, textured) and NS (smooth areas) methods
- **Seamless blending**: Gradient masks with configurable blend widths for smooth transitions
- **Bounding box detection**: Alpha channel analysis for precise subject positioning
- **Resolution handling**: Presets (HD, FHD, QHD, 4K) and custom "WIDTHxHEIGHT" format
- **Aspect ratio preservation**: Single dimension scaling maintains proportions
- **Debug mode**: Comprehensive intermediate outputs for each processing step
- **Memory efficiency**: Stream processing for large image sets

### Dependencies

- **rembg**: ML background removal (requires ONNX runtime)
- **opencv-python**: Image processing, inpainting, and color space conversion
- **PIL/Pillow**: Image I/O, format handling, alpha channel operations, and drawing
- **numpy**: Array operations for efficient image data manipulation
- **math**: Grid layout calculations and aspect ratio handling

### Error Handling Patterns

- Comprehensive file existence and permission validation
- Graceful handling of unsupported image formats with warnings
- Fallback resolution defaults when invalid custom formats provided
- Protected division operations with None value checking
- Alpha channel validation with automatic RGBA conversion

## Workflow and Use Cases

### Primary Use Cases
- **Professional Composites**: Create studio-quality images with subjects on custom backgrounds
- **Social Media Content**: Generate posts with consistent subject spacing and branded backgrounds
- **Product Photography**: Combine multiple product shots on seamless backgrounds
- **Event Photography**: Create group composites with even subject distribution
- **Marketing Materials**: Professional layouts with controlled background aesthetics

### Single-Command Workflow
The tool now uses a streamlined single-command approach:
```bash
python main.py combine input_photos/ -o final_result.png [options]
```
This automatically handles the complete pipeline from raw photos to final composite.

## Common Development Tasks

### Adding New Background Types
1. Add new type to CLI choices in `setup_argparse()` around main.py:360
2. Create new generation method like `create_X_background()` in ImageCombiner class
3. Add case to `create_combined_background()` method around main.py:512
4. Update auto-determination logic in `determine_background_type()` around main.py:278

### Modifying Inpainting Algorithms
Update `inpaint_backgrounds()` method around main.py:194-259, specifically the cv2.inpaint() calls

### Adding New Resolution Presets
Add resolution cases in the resolution handling section around main.py:770-784

### Customizing Subject Placement
Modify `calculate_subject_positions()` method around main.py:606-675 for different spacing algorithms

### Debug Output Customization
Each step has debug output sections - search for `debug_dir and self.debug` patterns throughout the code