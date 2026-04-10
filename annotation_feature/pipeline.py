from pathlib import Path
import json
from openai import OpenAI
from prompts import PROMPTS
from annotation_feature.demo_result import DEMO_RESULT

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
    # client = OpenAI()

    dataset_folder = Path("dataset")
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".mpeg", ".mpg"}

    results = {}
    
    # First, collect night and day files
    night_file = None
    day_file = None
    
    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in video_extensions:
            continue
        
        name = file.name.lower()
        if "rgb" not in name:
            continue
        
        if "night" in name:
            night_file = file
        elif "day" in name:
            day_file = file

    # Process all RGB files
    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in video_extensions:
            continue

        name = file.name.lower()
        if "rgb" not in name:
            continue

        try:
            print(f"Processing: {file}")

            file_results = {}
            
            # Process each annotation type from PROMPTS
            for annotation_type in PROMPTS.keys():
                caption_prompt = PROMPTS[annotation_type]["caption_prompt"]
                question_prompt = PROMPTS[annotation_type]["question_prompt"]
                answering_prompt = PROMPTS[annotation_type]["answering_prompt"]

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

            results[str(file)] = file_results
            print(f"Done: {file.name}")

        except Exception as e:
            results[str(file)] = f"ERROR: {e}"
            print(f"Failed: {file.name} -> {e}")

    # Save results to JSON file
    output_file = Path("dataset") / "results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    print(results)
    return results


if __name__ == "__main__":
    run()