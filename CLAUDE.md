# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Wallege is a Python CLI tool for professional background removal and seamless image combination. It uses machine learning (rembg with ONNX runtime) to remove backgrounds from images and combines them into panoramic layouts using advanced blending techniques.

### Key Components

- **BackgroundProcessor**: Core ML-powered background removal using rembg library
- **ResolutionManager**: Cross-platform screen resolution detection and preset management
- **SeamlessBlender**: Advanced image blending with generative fill techniques
- **BackgroundCombiner**: Main orchestrator for combining multiple background images
- **CLI**: Command-line interface with multiple subcommands

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
# Validate platform compatibility and dependencies
python main.py validate

# Show all available resolution presets
python main.py list-resolutions

# Process single image (creates removed_bg/, masks/, backgrounds/ subdirs)
python main.py process input.jpg -o output/

# Process directory with custom resolution
python main.py process photos/ -o output/ --resolution 4k

# Combine backgrounds with different arrangements
python main.py combine backgrounds/ -o panorama.png --arrangement horizontal
python main.py combine backgrounds/ -o grid.png --arrangement grid

# Social media formats
python main.py combine photos/ -o story.png --resolution instagram_story --arrangement vertical

# Custom resolution with generative fill
python main.py combine images/ -o result.png --width 2560 --height 1440 --generative-fill

# Disable automatic background combination during processing
python main.py process images/ -o output/ --no-combine

# Disable generative fill for faster processing
python main.py combine backgrounds/ -o result.png --no-generative-fill
```

## Code Architecture

### Core Processing Pipeline
1. **Input validation**: File/directory existence, permissions, supported formats
2. **Background removal**: ML-based segmentation using rembg
3. **Mask generation**: Binary masks from alpha channels or grayscale conversion
4. **Background isolation**: Inverse masking to extract original backgrounds
5. **Image combination**: Seamless blending with generative fill techniques

### Key Technical Details

- **Supported formats**: JPG, JPEG, PNG, BMP, TIFF, WEBP (case-insensitive)
- **Cross-platform resolution detection**: Uses tkinter, xrandr (Linux), system_profiler (macOS) with fallbacks
- **Generative fill**: OpenCV inpainting (Telea/NS algorithms) for seamless transitions
- **Blending modes**: Linear and Gaussian blending masks with configurable overlap widths
- **Memory management**: Processes images individually to handle large datasets
- **Output structure**: Creates organized subdirectories (removed_bg/, masks/, backgrounds/)
- **Resolution presets**: HD, FHD, QHD, 4K, 5K, 8K, mobile formats, Instagram formats, ultrawide, cinema
- **Arrangement options**: horizontal, vertical, grid layouts with intelligent sizing
- **Aspect ratio handling**: Maintains or forces aspect ratios based on user preference

### Dependencies

- **rembg**: ML background removal (requires ONNX runtime)
- **opencv-python**: Image processing and blending
- **PIL/Pillow**: Image I/O and format handling
- **numpy**: Array operations for image data
- **tkinter**: Optional GUI for resolution detection (with fallbacks)

### Error Handling Patterns

- Comprehensive permission checking for input/output paths
- Graceful fallbacks for headless systems (no DISPLAY)
- Platform-specific resolution detection with multiple fallback methods
- Detailed validation with user-friendly error messages

## Workflow and Use Cases

### Primary Use Cases
- **Content Creation**: Remove backgrounds for social media posts, presentations
- **Panoramic Photography**: Combine multiple background images into seamless panoramas
- **Batch Processing**: Handle entire directories of images efficiently
- **Social Media**: Generate content in specific formats (Instagram story, square posts)
- **Professional Photography**: Create composite backgrounds and environmental scenes

### Two-Step Workflow
1. **Process**: `python main.py process` - Removes backgrounds, creates masks, isolates backgrounds
2. **Combine**: `python main.py combine` - Blends isolated backgrounds into seamless compositions

## Common Development Tasks

### Adding New Resolution Presets
Edit `ResolutionManager.get_resolution_presets()` in main.py:209-225

### Modifying Blending Algorithms  
Update `SeamlessBlender` class methods in main.py:246-482

### Adding New Arrangement Types
Extend `combine_backgrounds()` method in main.py:492 and CLI parser in main.py:675-709