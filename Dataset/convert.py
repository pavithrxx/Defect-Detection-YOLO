import os
import xml.etree.ElementTree as ET

# Define your directories
xml_dir = 'labelled_images'  # Replace with the path to your XML files
output_dir = 'yolo_format'  # Replace with the desired output directory for YOLO files

# Make sure the output directory exists
os.makedirs(output_dir, exist_ok=True)

# List of defect names corresponding to your classes
classes = {
    "void": 0,
    "keyhole": 1,
    "cross-section reduction defect": 2,
    "crack": 3,
    "galling": 4,
    "flash": 5
}

# Process each XML file
for xml_file in os.listdir(xml_dir):
    if xml_file.endswith('.xml'):
        tree = ET.parse(os.path.join(xml_dir, xml_file))
        root = tree.getroot()

        # Get image dimensions
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        # Prepare a list to hold the YOLO formatted data
        yolo_data = []

        # Iterate through each object in the XML
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            class_id = classes[class_name]  # Get the class index
            
            # Get bounding box coordinates
            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / width
            y_center = (ymin + ymax) / 2 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height
            
            # Append formatted string to the list
            yolo_data.append(f"{class_id} {x_center} {y_center} {box_width} {box_height}")

        # Save to a TXT file
        txt_file_path = os.path.join(output_dir, xml_file.replace('.xml', '.txt'))
        with open(txt_file_path, 'w') as f:
            f.write("\n".join(yolo_data))
