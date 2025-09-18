from PIL import Image
import xml.etree.ElementTree as ET
import unicodedata
import pandas as pd
from pathlib import Path
from datasets import Dataset, load_from_disk, concatenate_datasets, DatasetInfo, DatasetDict

def normalize_chinese_text(text):
    """
    Normalize text to standard Chinese Unicode form.
    Converts variant Unicode characters (e.g., Kangxi Radicals) into normal forms.
    """
    return unicodedata.normalize("NFKC", text)


def create_images_from_regions(image_path, regions,
                                          buffer_above=10, buffer_below=10,
                                          buffer_left=10, buffer_right=10):
    image = Image.open(image_path)
    cropped_images = []

    for region_id, coords_str in zip(regions["region_id"], regions["coord_str"]):
        points = [tuple(map(int, point.split(','))) for point in coords_str.split()]
        x_coords, y_coords = zip(*points)

        min_x = min(x_coords) - buffer_left
        max_x = max(x_coords) + buffer_right
        min_y = min(y_coords) - buffer_above
        max_y = max(y_coords) + buffer_below

        cropped_image = image.crop((min_x, min_y, max_x, max_y))
        cropped_images.append((cropped_image, region_id))

    return cropped_images


def parse_xml_index_ordering(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespace = {
        'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
    }

    regions = {
        "region_id": [],
        "coord_str": [],
        "ordered_text": [],
        "num_lines": []
    }

    for text_region in root.findall(".//ns:TextRegion", namespace):
        coords_elem = text_region.find("ns:Coords", namespace)
        if coords_elem is None:
            continue

        region_id = text_region.get("id")
        coords_str = coords_elem.get("points")

        text_lines = []
        for line in text_region.findall(".//ns:TextLine", namespace):
            unicode_elem = line.find(".//ns:Unicode", namespace)
            if unicode_elem is None:
                continue
            try:
                normalized = normalize_chinese_text(unicode_elem.text) if unicode_elem.text else ""
            except Exception:
                normalized = ""
            try:
                index = int(line.get("custom").split("{index:")[1].split(";}")[0])
            except Exception:
                continue
            text_lines.append((normalized, index))
        
        text_lines.sort(key=lambda x: x[1])
        ordered_text = " ".join(text for text, _ in text_lines)

        regions["region_id"].append(region_id)
        regions["coord_str"].append(coords_str)
        regions["ordered_text"].append(ordered_text)
        regions["num_lines"].append(len(text_lines))

    return regions


def extract_and_create(input_folder, output_folder):
    jpg_folder = input_folder / "jpg"
    xml_folder = input_folder / "xml"

    print(f"Processing images from: {jpg_folder}")
    print(f"Processing XMLs from: {xml_folder}")

    image_paths = sorted(jpg_folder.glob("*.jpg"), key=lambda p: p.stem)
    xml_paths = sorted(xml_folder.glob("*.xml"), key=lambda p: p.stem)

    print(f"Found {len(image_paths)} images and {len(xml_paths)} XML files.")

    for image_path, xml_path in zip(image_paths, xml_paths):
        if image_path.stem == xml_path.stem:
            base_name = image_path.stem
            print(f"Processing: {base_name}")

            regions = parse_xml_index_ordering(xml_path)
            pd.DataFrame({
                "text": regions["ordered_text"],
                "identifier": [f"{base_name}_{region_id}" for region_id in regions["region_id"]],
                "num_lines": regions["num_lines"]
            }).to_csv(f"{output_folder}/texts/{base_name}.csv", index=False)

            cropped_images = create_images_from_regions(image_path, regions,
                                                                  buffer_above=0, buffer_below=0, buffer_left=0, buffer_right=0)

            for cropped_image, region_id in cropped_images:
                cropped_image.save(f"{output_folder}/images/{base_name}_{region_id}.png")
                print(f"Saved image: {base_name}_{region_id}.png") 


# Function to process each dataframe and create a Hugging Face dataset
def process_dataframe(df, image_dir):
    dataset = Dataset.from_pandas(df)
    
    # # Function to map the image loading for each row
    # def process_images(example):
    #     if("singlecol" in image_dir):
    #         image_filename = example['id'] + '.jpg' 
    #     else:
    #         image_filename = example['identifier'] + '.png' 
    #     image_path = os.path.join(image_dir, image_filename)  # 'identifier' is the image file name
    #     example['image'] = Image.open(image_path)
    #     return example

    # Apply the image loading function
    dataset = dataset.map(process_images)
    return dataset

# Function to map the image loading for each row
def process_images(example):
    if("singlecol" in image_dir):
        image_filename = example['id'] + '.jpg' 
    else:
        image_filename = example['identifier'] + '.png' 
    image_path = os.path.join(image_dir, image_filename)  # 'identifier' is the image file name
    example['image'] = Image.open(image_path)
    return example



def create_images_from_lines_singlecol(image_path, lines,
                                          buffer_above=10, buffer_below=10,
                                          buffer_left=10, buffer_right=10):
    image = Image.open(image_path)
    cropped_images = []

    for line_id, coords_str in zip(lines["line_id"], lines["coord_str"]):
        points = [tuple(map(int, point.split(','))) for point in coords_str.split()]
        x_coords, y_coords = zip(*points)

        min_x = min(x_coords) - buffer_left
        max_x = max(x_coords) + buffer_right
        min_y = min(y_coords) - buffer_above
        max_y = max(y_coords) + buffer_below

        cropped_image = image.crop((min_x, min_y, max_x, max_y))
        cropped_images.append((cropped_image, line_id))

    return cropped_images


def parse_xml_line_ordering_singlecol(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    namespace = {
        'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
    }

    lines = {
        "line_id": [],
        "coord_str": [],
        "ordered_text": []
    }

    for text_region in root.findall(".//ns:TextRegion", namespace):
        for line in text_region.findall(".//ns:TextLine", namespace):
            coords_elem = line.find("ns:Coords", namespace)
            if coords_elem is None:
                continue

            line_id = line.get("id")
            coords_str = coords_elem.get("points")

            unicode_elem = line.find(".//ns:Unicode", namespace)
            if unicode_elem is None:
                continue
            try:
                normalized = normalize_chinese_text(unicode_elem.text) if unicode_elem.text else ""
            except Exception:
                normalized = ""

            lines["line_id"].append(line_id)
            lines["coord_str"].append(coords_str)
            lines["ordered_text"].append(normalized)

    return lines


def extract_and_create_singlecol(input_folder, output_folder):
    jpg_folder = input_folder / "jpg"
    xml_folder = input_folder / "xml"

    print(f"Processing images from: {jpg_folder}")
    print(f"Processing XMLs from: {xml_folder}")

    image_paths = sorted(jpg_folder.glob("*.jpg"), key=lambda p: p.stem)
    xml_paths = sorted(xml_folder.glob("*.xml"), key=lambda p: p.stem)

    print(f"Found {len(image_paths)} images and {len(xml_paths)} XML files.")

    for image_path, xml_path in zip(image_paths, xml_paths):
        if image_path.stem == xml_path.stem:
            base_name = image_path.stem
            print(f"Processing: {base_name}")

            lines = parse_xml_line_ordering_singlecol(xml_path)
            pd.DataFrame({
                "text": lines["ordered_text"],
                "identifier": [f"{base_name}_{line_id}" for line_id in lines["line_id"]]
            }).to_csv(f"{output_folder}/texts/{base_name}.csv", index=False)

            cropped_images = create_images_from_lines_singlecol(image_path, lines,
                                                                  buffer_above=0, buffer_below=0, buffer_left=0, buffer_right=0)

            for cropped_image, line_id in cropped_images:
                cropped_image.save(f"{output_folder}/images/{base_name}_{line_id}.png")
                print(f"Saved image: {base_name}_{line_id}.png")