import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import xml.etree.ElementTree as ET


def augment_image(image_path, xml_path, save_dir, num_augmentations=5):
    # Create directories if they don't exist
    if not os.path.exists(os.path.join(save_dir, "images")):
        os.makedirs(os.path.join(save_dir, "images"))
    if not os.path.exists(os.path.join(save_dir, "annotations")):
        os.makedirs(os.path.join(save_dir, "annotations"))

    # Load image in RGB format
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image.shape

    # Convert VOC annotations to YOLO format
    yolo_annotations = convert_voc_to_yolo(xml_path, width, height)

    # Extract bboxes and labels separately
    bboxes = np.array([anno[1:] for anno in yolo_annotations], dtype=np.float32)
    labels = np.array([anno[0] for anno in yolo_annotations], dtype=np.int64)

    # Define augmentation pipeline
    augmentation_pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.Rotate(limit=15, p=0.2),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.2),
        A.RandomSizedBBoxSafeCrop(height=height, width=width, p=0.3),
    ], bbox_params=A.BboxParams(format='yolo', min_visibility=0.3, label_fields=['category_id']))

    for i in range(num_augmentations):
        # Apply augmentation
        augmented = augmentation_pipeline(image=image, bboxes=bboxes, category_id=labels)

        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']

        # Ensure augmented image is in RGB format
        augmented_image_rgb = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)

        # Construct save paths
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        augmented_image_path = os.path.join(save_dir, "images", f"{base_name}_aug_{i}.jpg")
        augmented_annotation_path = os.path.join(save_dir, "annotations", f"{base_name}_aug_{i}.xml")

        # Save augmented image in RGB format
        cv2.imwrite(augmented_image_path, augmented_image_rgb)

        # Convert augmented YOLO bboxes back to VOC format and save as XML
        augmented_voc_annotations = convert_yolo_to_voc(augmented_bboxes, width, height, labels)

        save_as_voc_xml(augmented_annotation_path, augmented_voc_annotations, augmented_image.shape, base_name)


def convert_voc_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    yolo_annotations = []
    for obj in root.findall('object'):
        class_id = 0  # Assuming all objects are of the same class (you might need to adjust this)
        bbox = obj.find('bndbox')
        xmin = float(bbox.find('xmin').text)
        ymin = float(bbox.find('ymin').text)
        xmax = float(bbox.find('xmax').text)
        ymax = float(bbox.find('ymax').text)

        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        yolo_annotations.append((class_id, x_center, y_center, width, height))

    return yolo_annotations


def convert_yolo_to_voc(yolo_bboxes, img_width, img_height, labels):
    voc_annotations = []
    for bbox, label in zip(yolo_bboxes, labels):
        x_center, y_center, bbox_width, bbox_height = bbox
        xmin = int((x_center - bbox_width / 2) * img_width)
        xmax = int((x_center + bbox_width / 2) * img_width)
        ymin = int((y_center - bbox_height / 2) * img_height)
        ymax = int((y_center + bbox_height / 2) * img_height)
        voc_annotations.append((label, xmin, ymin, xmax, ymax))
    return voc_annotations


def save_as_voc_xml(file_path, annotations, image_shape, image_name):
    height, width, depth = image_shape

    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "augmented_images"
    ET.SubElement(annotation, "filename").text = f"{image_name}.jpg"
    ET.SubElement(annotation, "path").text = file_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = str(depth)

    ET.SubElement(annotation, "segmented").text = "0"

    for label, xmin, ymin, xmax, ymax in annotations:
        obj = ET.SubElement(annotation, "object")
        ET.SubElement(obj, "name").text = str(label)  # You can map class_id to class names if necessary
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = "0"
        ET.SubElement(obj, "difficult").text = "0"
        bndbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(xmin)
        ET.SubElement(bndbox, "ymin").text = str(ymin)
        ET.SubElement(bndbox, "xmax").text = str(xmax)
        ET.SubElement(bndbox, "ymax").text = str(ymax)

    tree = ET.ElementTree(annotation)
    tree.write(file_path)


def augment_dataset(dataset_dir, save_dir, num_augmentations=5):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.makedirs(os.path.join(save_dir, "images"))
        os.makedirs(os.path.join(save_dir, "annotations"))

    # Process each folder in the dataset directory
    for folder in os.listdir(dataset_dir):
        folder_path = os.path.join(dataset_dir, folder)
        if os.path.isdir(folder_path):
            image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg') or f.endswith('.png')]

            for image_file in tqdm(image_files, desc=f"Augmenting {folder}"):
                image_path = os.path.join(folder_path, image_file)
                xml_path = os.path.join(folder_path, image_file.replace('.jpg', '.xml').replace('.png', '.xml'))

                augment_image(image_path, xml_path, save_dir, num_augmentations)


# Define paths
dataset_directory = "./augmentdataxml"  # Update this to your dataset directory
augmented_save_directory = "./augmentdataxml"  # Update this to your desired save directory

# Apply augmentation
augment_dataset(dataset_directory, augmented_save_directory, num_augmentations=5)
