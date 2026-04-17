import modal
from datetime import datetime

app = modal.App("rgb-pipeline")

image = (
    modal.Image.debian_slim()
    .pip_install("openai")
    .add_local_python_source("annotation_feature", "prompts", "dataset")
    .add_local_file("main.py", "/root/main.py")
)

@app.function(
    image=image,
    timeout=60 * 60,
)
def run_pipeline():
    import json
    print(f"🚀 Starting pipeline execution on Modal at {datetime.now()}")
    from annotation_feature import pipeline
    pipeline.run()
    print(f"✅ Pipeline execution completed on Modal at {datetime.now()}")
    
    # Write execution metadata
    with open("/root/execution_log.txt", "a") as f:
        f.write(f"\nModal execution at {datetime.now()}\n")
    print("✨ Execution logged on Modal")

@app.local_entrypoint()
def main():
    print(f"📍 Local execution started at {datetime.now()}")
    print("📍 Sending job to Modal servers...")
    run_pipeline.remote()
    print(f"✨ Job result received from Modal at {datetime.now()}!")
    with open("local_run_timestamp.txt", "w") as f:
        f.write(f"Last successful Modal run: {datetime.now()}\n")

if __name__ == "__main__":
    main()