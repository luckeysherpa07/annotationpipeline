import os
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("rgb-pipeline")
openai_secret = modal.Secret.from_name("annotationpipeline")
dataset_volume = modal.Volume.from_name("my-dataset", create_if_missing=True)

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_python_source("annotation_feature", "prompts")
    .add_local_file("main.py", "/root/main.py")
)


@app.function(
    image=image,
    secrets=[openai_secret],
    volumes={"/data": dataset_volume},
    timeout=60 * 60,
)
def run_pipeline():
    print(f"Starting pipeline execution on Modal at {datetime.now()}")
    print(f"Mounted dataset contents: {os.listdir('/data/dataset')}")

    from annotation_feature import pipeline

    pipeline.run(dataset_folder=Path("/data/dataset"))
    print(f"Pipeline execution completed on Modal at {datetime.now()}")

    with open("/root/execution_log.txt", "a", encoding="utf-8") as f:
        f.write(f"\nModal execution at {datetime.now()}\n")
    print("Execution logged on Modal")


@app.local_entrypoint()
def main():
    print(f"Local execution started at {datetime.now()}")
    print("Sending job to Modal servers...")
    run_pipeline.remote()
    print(f"Job result received from Modal at {datetime.now()}!")
    with open("local_run_timestamp.txt", "w", encoding="utf-8") as f:
        f.write(f"Last successful Modal run: {datetime.now()}\n")


if __name__ == "__main__":
    main()
