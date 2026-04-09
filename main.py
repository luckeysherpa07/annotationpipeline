from pathlib import Path

dataset_folder = Path("dataset")
video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}

for file in dataset_folder.rglob("*"):
    if file.is_file() and file.suffix.lower() in video_extensions:
        if "rgb" in file.name.lower():
            print(file.name)