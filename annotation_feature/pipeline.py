from pathlib import Path
import json
from prompts.rgb_prompts import RGB_PROMPTS
from annotation_feature.demo_result import DEMO_RESULT

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def get_pair_key(file: Path) -> str:
    """
    Build a shared key for matching day/night RGB videos from the same scene.
    """
    stem = file.stem.lower()
    for token in ("_night", "_day", "night", "day"):
        stem = stem.replace(token, "")
    stem = stem.replace("__", "_").strip("_")
    return str(file.parent / stem)

def get_caption_from_openai(client, file, caption_prompt):
    """
    Call OpenAI API to generate a caption for a video file.
    
    Args:
        client: OpenAI client instance
        file: Path to the video file
        caption_prompt: The prompt to send to the API
        
    Returns:
        The caption text from the API response
    """
    with open(file, "rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="user_data"
        )

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded.id},
                    {"type": "input_text", "text": caption_prompt},
                ],
            }
        ],
    )

    return response.output_text


def get_question_from_openai(client, caption, question_prompt):
    """
    Call OpenAI API to generate a question from a caption.
    
    Args:
        client: OpenAI client instance
        caption: The caption text to generate a question from
        question_prompt: The prompt to send to the API
        
    Returns:
        The question text from the API response
    """
    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"{caption}\n\n{question_prompt}"},
                ],
            }
        ],
    )

    return response.output_text


def get_answer_from_openai(client, file, question, answering_prompt):
    """
    Call OpenAI API to generate an answer for a question based on video.
    
    Args:
        client: OpenAI client instance
        file: Path to the video file
        question: The question to answer
        answering_prompt: The prompt to send to the API
        
    Returns:
        The answer text from the API response
    """
    with open(file, "rb") as f:
        uploaded = client.files.create(
            file=f,
            purpose="user_data"
        )

    response = client.responses.create(
        model="gpt-5",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_file", "file_id": uploaded.id},
                    {"type": "input_text", "text": f"Question: {question}\n\n{answering_prompt}"},
                ],
            }
        ],
    )

    return response.output_text


def run():
    #### Uncomment for OpenAI version #####
    # client = OpenAI()

    dataset_folder = Path("dataset")
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}

    results = {}

    # First, collect day/night RGB files and group them into pairs.
    paired_files = {}

    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in video_extensions:
            continue

        name = file.name.lower()
        if "rgb" not in name:
            continue

        pair_key = get_pair_key(file)
        paired_files.setdefault(pair_key, {"day": None, "night": None})

        if "night" in name:
            paired_files[pair_key]["night"] = file
        elif "day" in name:
            paired_files[pair_key]["day"] = file

    # Process one combined result for each day/night pair.
    for pair_key, files in paired_files.items():
        night_file = files["night"]
        day_file = files["day"]

        if not night_file and not day_file:
            continue

        try:
            print(f"Processing pair: {pair_key}")

            file_results = {}

            # Process each annotation type from RGB_PROMPTS
            for annotation_type in RGB_PROMPTS.keys():
                caption_prompt = RGB_PROMPTS[annotation_type]["caption_prompt"]
                question_prompt = RGB_PROMPTS[annotation_type]["question_prompt"]
                answering_prompt = RGB_PROMPTS[annotation_type]["answering_prompt"]

                caption = None
                question = None
                answer = None

                # Step 1: Get caption from night file
                if night_file:
                    #### Uncomment for OpenAI version #####
                    # caption = get_caption_from_openai(client, night_file, caption_prompt)
                    caption = DEMO_RESULT[annotation_type]["caption"]
                
                # Step 2: Get question from caption
                if caption:
                    #### Uncomment for OpenAI version #####
                    # question = get_question_from_openai(client, caption, question_prompt)
                    question = DEMO_RESULT[annotation_type]["question"]
                
                # Step 3: Get answer from day file and question
                if day_file and question:
                    #### Uncomment for OpenAI version #####
                    # answer = get_answer_from_openai(client, day_file, question, answering_prompt)
                    answer = DEMO_RESULT[annotation_type]["answer"]

                file_results[annotation_type] = {
                    "caption": caption,
                    "question": question,
                    "answer": answer
                }

            results[pair_key] = {
                "night_file": str(night_file) if night_file else None,
                "day_file": str(day_file) if day_file else None,
                "annotations": file_results,
            }
            print(f"Done: {pair_key}")

        except Exception as e:
            results[pair_key] = f"ERROR: {e}"
            print(f"Failed: {pair_key} -> {e}")

    # Save results to JSON file at the project root
    output_file = Path("qa_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(results)
    return results


if __name__ == "__main__":
    run()
