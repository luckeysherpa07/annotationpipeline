from pathlib import Path

def run():
    dataset_folder = Path("dataset")
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}

    for file in dataset_folder.rglob("*"):
        if file.is_file() and file.suffix.lower() in video_extensions:
            name = file.name.lower()
            if "rgb" in name and "night" in name:
                print(file.name)


if __name__ == "__main__":
    run()