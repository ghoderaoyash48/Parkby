import os
import glob
import shutil
import yaml
import json
import numpy as np
import cv2
import torch
import argparse
import logging
import random
from pathlib import Path
import xml.etree.ElementTree as ET
from tqdm import tqdm

# Check for GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type == 'cuda':
    print(f"GPU available: {torch.cuda.get_device_name(0)}")
else:
    print("GPU not available, using CPU instead")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_annotation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define the target classes and mapping
TARGET_CLASSES = ['bike', 'bus', 'car', 'truck']

def create_directory_structure(base_dir):
    """Create the directory structure for processed data."""
    for dataset_type in ['train', 'val', 'test']:
        for folder in ['images', 'labels']:
            path = os.path.join(base_dir, dataset_type, folder)
            os.makedirs(path, exist_ok=True)
    logger.info(f"Created directory structure at {base_dir}")

def map_class_to_id(class_name):
    """Map class names to target class IDs."""
    # Standardize the class name
    class_name = str(class_name).lower().strip()
    
    # Bike category (includes all two-wheelers)
    if class_name in ["motorcycle", "bike", "scooter", "moped", "bicycle", 
                      "two wheeler", "two-wheeler", "twowheeler", "motorbike"]:
        return 0  # bike
    
    # Bus category
    elif class_name in ["bus"]:
        return 1  # bus
    
    # Car category
    elif class_name in ["car", "van"]:
        return 2  # car
    
    # Truck category
    elif class_name in ["truck", "lorry"]:
        return 3  # truck
    
    else:
        return None  # Unknown class

def map_numeric_class_dataset_specific(dataset_name, original_class_id):
    """
    Map numeric class IDs from specific datasets to standard class IDs.
    """
    # Convert to int if it's a string representing a number
    if isinstance(original_class_id, str) and original_class_id.isdigit():
        original_class_id = int(original_class_id)
    elif not isinstance(original_class_id, (int, np.int64, np.int32)):
        # Try to map by name if it's not a numeric ID
        return map_class_to_id(original_class_id)
    
    # Handle dataset-specific numeric mappings
    if "Vehicle_Detection" in dataset_name:
        # Based on folder 5's vehicles.yaml: bus, truck, motorcycle, car
        mapping = {
            0: 1,  # bus -> bus (1)
            1: 3,  # truck -> truck (3)
            2: 0,  # motorcycle -> bike (0)
            3: 2,  # car -> car (2)
        }
        return mapping.get(original_class_id)
    
    elif "motorcycleyolo" in dataset_name or "motoryolo" in dataset_name:
        # For motorcycle datasets, all classes might be different types of bikes
        if original_class_id in range(10):
            return 0  # All map to bike
        return None
    
    elif "4.vehicle" in dataset_name:
        # Based on classes mentioned: car, threewheel, bus, truck, motorbike, van
        mapping = {
            0: 2,  # car -> car (2)
            1: 0,  # threewheel -> bike (0)
            2: 1,  # bus -> bus (1)
            3: 3,  # truck -> truck (3)
            4: 0,  # motorbike -> bike (0)
            5: 2,  # van -> car (2)
        }
        return mapping.get(original_class_id)
    
    elif "Random" in dataset_name:
        # Default mapping for Random dataset
        mapping = {
            0: 0,  # Assuming 0 is bike
            1: 1,  # Assuming 1 is bus
            2: 2,  # Assuming 2 is car
            3: 3,  # Assuming 3 is truck
        }
        return mapping.get(original_class_id)
    
    elif "1.Random" in dataset_name:
        # If not specified, use a default mapping
        mapping = {
            0: 0,  # Assuming 0 is bike
            1: 1,  # Assuming 1 is bus
            2: 2,  # Assuming 2 is car
            3: 3,  # Assuming 3 is truck
        }
        return mapping.get(original_class_id)
    
    elif "2.Traffic" in dataset_name:
        # If not specified, use a default mapping
        mapping = {
            0: 0,  # Assuming 0 is bike
            1: 1,  # Assuming 1 is bus
            2: 2,  # Assuming 2 is car
            3: 3,  # Assuming 3 is truck
        }
        return mapping.get(original_class_id)
    
    elif "BDD100K" in dataset_name:
        # BDD100K has multiple vehicle categories
        mapping = {
            0: 0,  # bike
            1: 1,  # bus
            2: 2,  # car
            3: 3,  # truck
        }
        return mapping.get(original_class_id)
    
    # Add more dataset-specific mappings as needed
    
    # Default: try direct mapping if class ID is within range
    if original_class_id in range(len(TARGET_CLASSES)):
        return original_class_id
    
    # Unknown mapping
    return None

def convert_to_yolo_format(box, img_width, img_height):
    """
    Convert bounding box coordinates to YOLO format (center_x, center_y, width, height).
    """
    if isinstance(box, dict):
        x_min, y_min = box.get('x_min', box.get('xmin', 0)), box.get('y_min', box.get('ymin', 0))
        x_max, y_max = box.get('x_max', box.get('xmax', 0)), box.get('y_max', box.get('ymax', 0))
    else:
        x_min, y_min, x_max, y_max = box
    
    # Calculate center coordinates and dimensions
    center_x = ((x_min + x_max) / 2) / img_width
    center_y = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    
    # Ensure values are within [0, 1]
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height

def process_yolo_label(label_path, dataset_name, output_label_path, img_width=None, img_height=None):
    """Process YOLO format label files."""
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # Standard YOLO format: class x_center y_center width height
                original_class_id = parts[0]
                new_class_id = map_numeric_class_dataset_specific(dataset_name, original_class_id)
                
                if new_class_id is not None:
                    # Keep the coordinates unchanged as they're already in YOLO format
                    new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
                    new_lines.append(new_line)
        
        # Write new label file
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        with open(output_label_path, 'w') as f:
            f.writelines(new_lines)
        
        return True
    except Exception as e:
        logger.error(f"Error processing YOLO label {label_path}: {str(e)}")
        return False

def process_coco_json(json_path, dataset_name, output_dir):
    """Process COCO JSON format annotations."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Extract image info and annotations
        images = {img['id']: img for img in data.get('images', [])}
        
        # Process each annotation
        for annotation in data.get('annotations', []):
            image_id = annotation.get('image_id')
            if image_id not in images:
                continue
                
            image_info = images[image_id]
            img_width, img_height = image_info.get('width', 0), image_info.get('height', 0)
            image_name = image_info.get('file_name', '')
            
            # Map category to our target classes
            category_id = annotation.get('category_id')
            new_class_id = map_numeric_class_dataset_specific(dataset_name, category_id)
            
            if new_class_id is None:
                continue
            
            # Convert bbox to YOLO format
            # COCO format: [x_min, y_min, width, height]
            bbox = annotation.get('bbox', [0, 0, 0, 0])
            x_min, y_min, w, h = bbox
            x_max, y_max = x_min + w, y_min + h
            
            center_x, center_y, width, height = convert_to_yolo_format(
                (x_min, y_min, x_max, y_max), img_width, img_height
            )
            
            # Create output label file path
            base_filename = os.path.splitext(image_name)[0]
            output_label_path = os.path.join(output_dir, f"{base_filename}.txt")
            
            # Append to label file
            with open(output_label_path, 'a') as f:
                f.write(f"{new_class_id} {center_x} {center_y} {width} {height}\n")
        
        return True
    except Exception as e:
        logger.error(f"Error processing COCO JSON {json_path}: {str(e)}")
        return False

def process_pascal_voc(xml_path, dataset_name, output_label_path):
    """Process Pascal VOC XML format annotations."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get image dimensions
        size = root.find('size')
        img_width = int(size.find('width').text)
        img_height = int(size.find('height').text)
        
        with open(output_label_path, 'w') as f:
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                new_class_id = map_class_to_id(class_name)
                
                if new_class_id is None:
                    continue
                
                bbox = obj.find('bndbox')
                x_min = float(bbox.find('xmin').text)
                y_min = float(bbox.find('ymin').text)
                x_max = float(bbox.find('xmax').text)
                y_max = float(bbox.find('ymax').text)
                
                center_x, center_y, width, height = convert_to_yolo_format(
                    (x_min, y_min, x_max, y_max), img_width, img_height
                )
                
                f.write(f"{new_class_id} {center_x} {center_y} {width} {height}\n")
        
        return True
    except Exception as e:
        logger.error(f"Error processing VOC XML {xml_path}: {str(e)}")
        return False

def extract_bounding_boxes_from_mask(mask_img_path, class_color_map=None):
    """
    Extract bounding boxes from segmentation mask image using GPU acceleration if available.
    
    Args:
        mask_img_path: Path to the segmentation mask image
        class_color_map: Optional dictionary mapping colors to class IDs
        
    Returns:
        list: List of (class_id, x_center, y_center, width, height) tuples in YOLO format
    """
    try:
        # Read the mask image
        mask = cv2.imread(mask_img_path)
        if mask is None:
            logger.error(f"Could not read mask image: {mask_img_path}")
            return []
        
        # Convert to grayscale
        gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Find contours
        contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        img_height, img_width = gray.shape
        
        bboxes = []
        for contour in contours:
            # Skip very small contours
            if cv2.contourArea(contour) < 100:
                continue
                
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Determine class based on average color inside the contour
            roi = mask[y:y+h, x:x+w]
            avg_color = np.mean(roi, axis=(0, 1))
            
            # Default color mapping if not provided
            if class_color_map is None:
                # Map color to class ID based on dominant channel
                b, g, r = avg_color
                if r > max(b, g) and r > 100:
                    class_id = 0  # bike - red dominant
                elif g > max(r, b) and g > 100:
                    class_id = 1  # bus - green dominant
                elif b > max(r, g) and b > 100:
                    class_id = 2  # car - blue dominant
                else:
                    class_id = 3  # truck - other colors
            else:
                # Use provided color map
                # Find closest color in the map
                min_dist = float('inf')
                class_id = 3  # default to truck
                
                for color, cid in class_color_map.items():
                    # Calculate color distance
                    dist = np.sum((np.array(color) - avg_color)**2)
                    if dist < min_dist:
                        min_dist = dist
                        class_id = cid
            
            # Convert to YOLO format (center_x, center_y, width, height)
            center_x = (x + w/2) / img_width
            center_y = (y + h/2) / img_height
            width = w / img_width
            height = h / img_height
            
            bboxes.append((class_id, center_x, center_y, width, height))
        
        return bboxes
    except Exception as e:
        logger.error(f"Error processing mask image {mask_img_path}: {str(e)}")
        return []

def copy_image(src_image, dst_image):
    """Copy image file with proper error handling."""
    try:
        os.makedirs(os.path.dirname(dst_image), exist_ok=True)
        shutil.copy(src_image, dst_image)
        return True
    except Exception as e:
        logger.error(f"Error copying image {src_image} to {dst_image}: {str(e)}")
        return False

def get_image_dimensions(image_path):
    """Get image dimensions using OpenCV."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None, None
        height, width = img.shape[:2]
        return width, height
    except Exception as e:
        logger.error(f"Error getting dimensions for {image_path}: {str(e)}")
        return None, None

def find_image_file(base_path, base_name):
    """Find image file with any extension."""
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
        image_path = os.path.join(base_path, f"{base_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None

def is_random_dataset(dataset_name):
    """Check if this is the Random dataset that needs special handling."""
    return "Random" in dataset_name

def process_random_folder(input_path, output_base, dataset_name, dataset_type, unified_dir, file_prefix=""):
    """Process the Random dataset with image-based labels."""
    logger.info(f"Processing {dataset_name} - {dataset_type} as image mask dataset")
    
    # Define color to class mapping for Random dataset - ADJUST THIS FOR YOUR DATASET
    # This is an example - you'll need to check the actual colors in your mask images
    class_color_map = {
        (255, 0, 0): 0,  # Red -> bike
        (0, 255, 0): 1,  # Green -> bus
        (0, 0, 255): 2,  # Blue -> car
        (255, 255, 0): 3,  # Yellow -> truck
    }
    
    # Output directories
    output_dir = os.path.join(output_base, dataset_type)
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    
    # Output for unified dataset
    unified_output_dir = os.path.join(unified_dir, dataset_type)
    unified_images_dir = os.path.join(unified_output_dir, 'images')
    unified_labels_dir = os.path.join(unified_output_dir, 'labels')
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    os.makedirs(unified_images_dir, exist_ok=True)
    os.makedirs(unified_labels_dir, exist_ok=True)
    
    # Look for common folder structures
    potential_img_folders = ['images', 'img', 'imgs', 'image']
    potential_label_folders = ['labels', 'label', 'annotations', 'ann']
    
    # Find image and label folders
    images_path = None
    labels_path = None
    
    for img_folder in potential_img_folders:
        potential_path = os.path.join(input_path, img_folder)
        if os.path.isdir(potential_path):
            images_path = potential_path
            break
    
    for label_folder in potential_label_folders:
        potential_path = os.path.join(input_path, label_folder)
        if os.path.isdir(potential_path):
            labels_path = potential_path
            break
    
    if images_path is None:
        logger.warning(f"Could not find images folder for {input_path}")
        return
    
    if labels_path is None:
        logger.warning(f"Could not find labels folder for {input_path}")
        return
    
    logger.info(f"Found images at {images_path}")
    logger.info(f"Found mask images at {labels_path}")
    
    # Get all image files
    image_files = glob.glob(os.path.join(images_path, '*.jpg')) + \
                 glob.glob(os.path.join(images_path, '*.png'))
    
    processed_count = 0
    for image_file in tqdm(image_files, desc=f"Processing {dataset_type} images"):
        base_name = os.path.splitext(os.path.basename(image_file))[0]
        
        # Find corresponding mask file
        mask_file = None
        for ext in ['.png', '.jpg', '.jpeg']:
            potential_mask = os.path.join(labels_path, f"{base_name}{ext}")
            if os.path.exists(potential_mask):
                mask_file = potential_mask
                break
        
        if mask_file is None:
            logger.warning(f"No mask found for image {base_name}")
            continue
        
        # Extract bounding boxes from mask
        bboxes = extract_bounding_boxes_from_mask(mask_file, class_color_map)
        
        if not bboxes:
            logger.warning(f"No bounding boxes extracted from {mask_file}")
            continue
        
        # Write YOLO format label file
        output_label_path = os.path.join(labels_output_dir, f"{base_name}.txt")
        unified_label_path = os.path.join(unified_labels_dir, f"{file_prefix}{base_name}.txt")
        
        with open(output_label_path, 'w') as f:
            for class_id, center_x, center_y, width, height in bboxes:
                f.write(f"{class_id} {center_x} {center_y} {width} {height}\n")
        
        # Copy to unified dataset with prefix to avoid conflicts
        shutil.copy(output_label_path, unified_label_path)
        
        # Copy image file
        img_ext = os.path.splitext(image_file)[1]
        output_image_path = os.path.join(images_output_dir, f"{base_name}{img_ext}")
        unified_image_path = os.path.join(unified_images_dir, f"{file_prefix}{base_name}{img_ext}")
        
        shutil.copy(image_file, output_image_path)
        shutil.copy(image_file, unified_image_path)
        
        processed_count += 1
    
    logger.info(f"Processed {processed_count} files from {dataset_name} - {dataset_type}")

def process_folder(input_path, output_base, dataset_name, dataset_type, unified_dir, file_prefix=""):
    """
    Process a folder of data (train, val, test) and copy to unified directory.
    """
    # If this is the Random dataset, use special processing
    if is_random_dataset(dataset_name):
        process_random_folder(input_path, output_base, dataset_name, dataset_type, unified_dir, file_prefix)
        return
    
    logger.info(f"Processing {dataset_name} - {dataset_type} from {input_path}")
    
    # Output for individual dataset processing
    output_dir = os.path.join(output_base, dataset_type)
    images_output_dir = os.path.join(output_dir, 'images')
    labels_output_dir = os.path.join(output_dir, 'labels')
    
    # Output for unified dataset
    unified_output_dir = os.path.join(unified_dir, dataset_type)
    unified_images_dir = os.path.join(unified_output_dir, 'images')
    unified_labels_dir = os.path.join(unified_output_dir, 'labels')
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(labels_output_dir, exist_ok=True)
    os.makedirs(unified_images_dir, exist_ok=True)
    os.makedirs(unified_labels_dir, exist_ok=True)
    
    # Figure out the structure
    images_path = None
    labels_path = None
    
    # Look for common folder structures
    potential_img_folders = ['images', 'img', 'imgs', 'image']
    potential_label_folders = ['labels', 'label', 'annotations', 'ann']
    
    # First, check if the input path itself contains images and labels subdirectories
    for img_folder in potential_img_folders:
        if os.path.isdir(os.path.join(input_path, img_folder)):
            images_path = os.path.join(input_path, img_folder)
            break
    
    for label_folder in potential_label_folders:
        if os.path.isdir(os.path.join(input_path, label_folder)):
            labels_path = os.path.join(input_path, label_folder)
            break
    
    # If not found, the input path might be the images folder, and the labels folder might be at the same level
    if images_path is None:
        # Check if input_path itself has images
        if glob.glob(os.path.join(input_path, '*.jpg')) or \
           glob.glob(os.path.join(input_path, '*.png')):
            images_path = input_path
            
            # Look for labels folder at same level as images
            parent_dir = os.path.dirname(input_path)
            for label_folder in potential_label_folders:
                if os.path.isdir(os.path.join(parent_dir, label_folder)):
                    labels_path = os.path.join(parent_dir, label_folder)
                    break
    
    # If still not found, try one more structure
    if images_path is None and labels_path is None:
        parent_dir = os.path.dirname(input_path)
        for img_folder in potential_img_folders:
            if os.path.isdir(os.path.join(parent_dir, img_folder)):
                images_path = os.path.join(parent_dir, img_folder)
                break
        
        for label_folder in potential_label_folders:
            if os.path.isdir(os.path.join(parent_dir, label_folder)):
                labels_path = os.path.join(parent_dir, label_folder)
                break
    
    if images_path is None:
        logger.warning(f"Could not find images folder for {input_path}")
        return
    
    if labels_path is None:
        logger.warning(f"Could not find labels folder for {input_path}")
        return
    
    logger.info(f"Found images at {images_path}")
    logger.info(f"Found labels at {labels_path}")
    
    # Process labels first to identify what images we need
    processed_files = set()
    
    # Check label format based on file extensions
    label_files = glob.glob(os.path.join(labels_path, '*'))
    if not label_files:
        logger.warning(f"No label files found in {labels_path}")
        return
    
    # Determine label format
    sample_label = label_files[0]
    label_ext = os.path.splitext(sample_label)[1].lower()
    
    # Process based on format
    for label_file in tqdm(label_files, desc=f"Processing {dataset_type} annotations"):
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        
        # Generate a unique name for unified dataset to avoid conflicts
        unified_base_name = f"{file_prefix}{base_name}"
        
        # Output paths
        output_label_path = os.path.join(labels_output_dir, f"{base_name}.txt")
        unified_label_path = os.path.join(unified_labels_dir, f"{unified_base_name}.txt")
        
        # Find corresponding image
        image_file = find_image_file(images_path, base_name)
        if not image_file:
            logger.warning(f"No matching image found for label {base_name}")
            continue
        
        img_width, img_height = get_image_dimensions(image_file)
        
        # Process based on label format
        success = False
        if label_ext == '.txt':  # Assuming YOLO format
            success = process_yolo_label(label_file, dataset_name, output_label_path)
            # Copy the processed label to unified directory
            if success and os.path.exists(output_label_path):
                shutil.copy(output_label_path, unified_label_path)
        elif label_ext == '.xml':  # Pascal VOC format
            success = process_pascal_voc(label_file, dataset_name, output_label_path)
            # Copy the processed label to unified directory
            if success and os.path.exists(output_label_path):
                shutil.copy(output_label_path, unified_label_path)
        elif label_ext == '.json':  # COCO format
            # For COCO, we process directly to output and then copy relevant files
            success = process_coco_json(label_file, dataset_name, labels_output_dir)
            
            # After processing COCO JSON, find all created label files and copy to unified
            if success:
                coco_labels = glob.glob(os.path.join(labels_output_dir, '*.txt'))
                for coco_label in coco_labels:
                    coco_base = os.path.splitext(os.path.basename(coco_label))[0]
                    unified_coco_path = os.path.join(unified_labels_dir, f"{file_prefix}{coco_base}.txt")
                    shutil.copy(coco_label, unified_coco_path)
        else:
            logger.warning(f"Unsupported label format: {label_ext} for {label_file}")
            continue
        
        if success:
            # Copy the corresponding image to both individual and unified directories
            image_ext = os.path.splitext(image_file)[1]
            output_image_path = os.path.join(images_output_dir, f"{base_name}{image_ext}")
            unified_image_path = os.path.join(unified_images_dir, f"{unified_base_name}{image_ext}")
            
            if copy_image(image_file, output_image_path):
                copy_image(image_file, unified_image_path)
                processed_files.add(base_name)
                
    logger.info(f"Processed {len(processed_files)} files from {dataset_name} - {dataset_type}")

def create_data_yaml(output_dir, dataset_name="unified_dataset"):
    """Create data.yaml file for YOLO training."""
    yaml_path = os.path.join(output_dir, 'data.yaml')
    
    data = {
        'train': './train/images',
        'val': './val/images',
        'test': './test/images',
        'nc': len(TARGET_CLASSES),
        'names': TARGET_CLASSES,
        'dataset': dataset_name
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    logger.info(f"Created data.yaml at {yaml_path}")

def process_dataset(dataset_path, output_base, unified_dir, skip_existing=False):
    """
    Process a complete dataset.
    
    Args:
        dataset_path: Path to the dataset folder
        output_base: Base path for output
        unified_dir: Directory for unified dataset
        skip_existing: Skip processing if output already exists
    """
    dataset_name = os.path.basename(dataset_path)
    output_dir = os.path.join(output_base, dataset_name)
    
    if skip_existing and os.path.exists(output_dir):
        logger.info(f"Skipping existing dataset: {dataset_name}")
        return
    
    logger.info(f"Processing dataset: {dataset_name} from {dataset_path}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique prefix for this dataset to avoid filename conflicts in unified dataset
    # Use first 3 chars of dataset name + underscore
    file_prefix = f"{dataset_name[:3].lower()}_{random.randint(100, 999)}_"
    
    # Find train, val, test folders
    for folder_type in ['train', 'val', 'validation', 'valid', 'test']:
        # Normalize 'valid'/'validation' to 'val'
        if folder_type in ['validation', 'valid']:
            output_folder_type = 'val'
        else:
            output_folder_type = folder_type
        
        # Look for the folder at the dataset path level
        folder_path = os.path.join(dataset_path, folder_type)
        if os.path.isdir(folder_path):
            process_folder(folder_path, output_dir, dataset_name, output_folder_type, 
                          unified_dir, file_prefix)
        else:
            # Special case for BDD100K-like structures
            if 'images' in dataset_path and folder_type in os.path.basename(dataset_path):
                # This might be something like 'BDD100K/images/10k/train'
                process_folder(dataset_path, output_dir, dataset_name, output_folder_type, 
                              unified_dir, file_prefix)
    
    # Create data.yaml for individual dataset
    create_data_yaml(output_dir, dataset_name)
    logger.info(f"Completed processing dataset: {dataset_name}")

def calculate_dataset_statistics(unified_dir):
    """Calculate statistics for the unified dataset."""
    statistics = {
        'train': {'total': 0, 'classes': {0: 0, 1: 0, 2: 0, 3: 0}},
        'val': {'total': 0, 'classes': {0: 0, 1: 0, 2: 0, 3: 0}},
        'test': {'total': 0, 'classes': {0: 0, 1: 0, 2: 0, 3: 0}}
    }
    
    for dataset_type in ['train', 'val', 'test']:
        labels_dir = os.path.join(unified_dir, dataset_type, 'labels')
        if not os.path.exists(labels_dir):
            continue
            
        label_files = glob.glob(os.path.join(labels_dir, '*.txt'))
        statistics[dataset_type]['total'] = len(label_files)
        
        # Count objects per class
        for label_file in label_files:
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    if class_id in statistics[dataset_type]['classes']:
                        statistics[dataset_type]['classes'][class_id] += 1
    
    # Print statistics
    logger.info("=== Unified Dataset Statistics ===")
    
    for dataset_type in ['train', 'val', 'test']:
        logger.info(f"\n{dataset_type.upper()} set:")
        logger.info(f"  Total images: {statistics[dataset_type]['total']}")
        
        for class_id, count in statistics[dataset_type]['classes'].items():
            class_name = TARGET_CLASSES[class_id]
            logger.info(f"  {class_name}: {count} instances")
    
    return statistics

def main():
    parser = argparse.ArgumentParser(description='Preprocess and unify vehicle detection datasets for YOLO format')
    parser.add_argument('--data_root', type=str, default='C:/projectfolders/ParkAI/data',
                        help='Root folder containing all datasets')
    parser.add_argument('--output_dir', type=str, default='C:/projectfolders/ParkAI/processed_data',
                        help='Output directory for processed individual datasets')
    parser.add_argument('--unified_dir', type=str, default='C:/projectfolders/ParkAI/unified_dataset',
                        help='Output directory for unified dataset')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip datasets that have already been processed')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU acceleration when available')
    
    args = parser.parse_args()
    
    # Check GPU availability and set device
    if args.gpu:
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            global DEVICE
            DEVICE = torch.device('cuda')
            # Enable OpenCV GPU acceleration if available
            try:
                cv2.setUseOptimized(True)
                if cv2.useOptimized():
                    logger.info("OpenCV optimization enabled")
                else:
                    logger.info("OpenCV optimization not available")
            except:
                logger.warning("Failed to set OpenCV optimization")
        else:
            logger.warning("GPU requested but not available. Using CPU instead.")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.unified_dir, exist_ok=True)
    
    # Create unified dataset structure
    create_directory_structure(args.unified_dir)
    
    # Process each dataset folder
    dataset_folders = [
        os.path.join(args.data_root, folder) for folder in os.listdir(args.data_root)
        if os.path.isdir(os.path.join(args.data_root, folder))
    ]
    
    for dataset_folder in dataset_folders:
        process_dataset(dataset_folder, args.output_dir, args.unified_dir, args.skip_existing)
    
    # Create data.yaml for unified dataset
    create_data_yaml(args.unified_dir, "unified_vehicle_dataset")
    
    # Calculate and display dataset statistics
    calculate_dataset_statistics(args.unified_dir)
    
    logger.info("\nPreprocessing completed. Unified dataset created successfully!")
    logger.info(f"Unified dataset location: {args.unified_dir}")
    logger.info(f"Use {os.path.join(args.unified_dir, 'data.yaml')} for YOLOv8 training")

if __name__ == "__main__":
    main()