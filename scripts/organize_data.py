#!/usr/bin/env python3
"""
Script to organize dog breed data from downloaded Kaggle datasets.
Matches breed names from CSV with image folders and copies to project structure.
"""

import os
import shutil
import csv
from pathlib import Path
import re

# Paths
PROJECT_ROOT = Path("/Users/Denise_Mendez/Documents/AI_architect/rag-document-analyzer")
CSV_PATH = PROJECT_ROOT / "data/documents/dog_breeds.csv"
IMAGE_SOURCE = Path("/Users/Denise_Mendez/Downloads/Dog Breeds Image Dataset")
IMAGE_DEST = PROJECT_ROOT / "data/images/raw"

# Create destination if not exists
IMAGE_DEST.mkdir(parents=True, exist_ok=True)


def normalize_breed_name(name):
    """Normalize breed name for matching (lowercase, remove spaces, underscores)."""
    return re.sub(r'[^a-z0-9]', '', name.lower())


def read_csv_breeds():
    """Read breeds from CSV file."""
    breeds = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            breeds.append({
                'name': row['Breed'],
                'normalized': normalize_breed_name(row['Breed']),
                'data': row
            })
    return breeds


def get_available_image_folders():
    """Get list of available image folders from source dataset."""
    folders = {}
    for folder in IMAGE_SOURCE.iterdir():
        if folder.is_dir() and not folder.name.startswith('.'):
            normalized = normalize_breed_name(folder.name)
            folders[normalized] = folder.name
    return folders


def match_breeds(csv_breeds, image_folders):
    """Match CSV breeds with image folders."""
    matches = []
    unmatched_csv = []

    for breed in csv_breeds:
        normalized = breed['normalized']
        if normalized in image_folders:
            matches.append({
                'csv_name': breed['name'],
                'folder_name': image_folders[normalized],
                'normalized': normalized,
                'data': breed['data']
            })
        else:
            unmatched_csv.append(breed['name'])

    return matches, unmatched_csv


def copy_breed_images(matches, max_images_per_breed=20):
    """Copy images for matched breeds to project structure."""
    copied_breeds = []

    for match in matches:
        source_folder = IMAGE_SOURCE / match['folder_name']
        dest_folder = IMAGE_DEST / match['folder_name']

        # Create destination folder
        dest_folder.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_files = list(source_folder.glob('*.jpg')) + \
                     list(source_folder.glob('*.jpeg')) + \
                     list(source_folder.glob('*.png'))

        # Copy up to max_images_per_breed
        copied_count = 0
        for img_file in image_files[:max_images_per_breed]:
            dest_file = dest_folder / img_file.name
            if not dest_file.exists():
                shutil.copy2(img_file, dest_file)
                copied_count += 1

        copied_breeds.append({
            'breed': match['csv_name'],
            'folder': match['folder_name'],
            'images_copied': copied_count
        })

        print(f"✓ Copied {copied_count} images for {match['csv_name']}")

    return copied_breeds


def create_breed_mapping(matches, copied_breeds):
    """Create a JSON mapping file of breeds with both text and images."""
    import json

    mapping = {
        'total_breeds': len(matches),
        'breeds': []
    }

    for i, match in enumerate(matches):
        copied_info = copied_breeds[i]
        mapping['breeds'].append({
            'name': match['csv_name'],
            'image_folder': match['folder_name'],
            'images_count': copied_info['images_copied'],
            'characteristics': match['data']
        })

    # Save mapping
    mapping_file = PROJECT_ROOT / "data/breed_mapping.json"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✓ Created breed mapping: {mapping_file}")
    return mapping_file


def main():
    """Main execution."""
    print("=" * 60)
    print("Dog Breed Dataset Organization")
    print("=" * 60)

    # Read CSV breeds
    print("\n1. Reading CSV breeds...")
    csv_breeds = read_csv_breeds()
    print(f"   Found {len(csv_breeds)} breeds in CSV")

    # Get available image folders
    print("\n2. Scanning image folders...")
    image_folders = get_available_image_folders()
    print(f"   Found {len(image_folders)} breed folders in dataset")

    # Match breeds
    print("\n3. Matching breeds...")
    matches, unmatched = match_breeds(csv_breeds, image_folders)
    print(f"   Matched: {len(matches)} breeds")
    print(f"   Unmatched: {len(unmatched)} breeds")

    if unmatched:
        print(f"\n   Unmatched breeds from CSV:")
        for breed in unmatched[:10]:
            print(f"     - {breed}")
        if len(unmatched) > 10:
            print(f"     ... and {len(unmatched) - 10} more")

    # Copy images
    print(f"\n4. Copying images (max 20 per breed)...")
    copied_breeds = copy_breed_images(matches, max_images_per_breed=20)

    total_images = sum(b['images_copied'] for b in copied_breeds)
    print(f"\n   Total images copied: {total_images}")

    # Create mapping file
    print("\n5. Creating breed mapping...")
    mapping_file = create_breed_mapping(matches, copied_breeds)

    print("\n" + "=" * 60)
    print("✓ Dataset organization complete!")
    print("=" * 60)
    print(f"\nSummary:")
    print(f"  - Breeds with images: {len(matches)}")
    print(f"  - Total images: {total_images}")
    print(f"  - Location: {IMAGE_DEST}")
    print(f"  - Mapping: {mapping_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
