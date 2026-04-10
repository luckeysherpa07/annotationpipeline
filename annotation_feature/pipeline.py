from pathlib import Path
from openai import OpenAI
from ..prompts import PROMPTS
from .demo_result import DEMO_RESULT

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
    caption_prompt = PROMPTS["object_recognition"]["caption_prompt"]
    question_prompt = PROMPTS["object_recognition"]["question_prompt"]
    answering_prompt = PROMPTS["object_recognition"]["answering_prompt"]

    results = {}

    for file in dataset_folder.rglob("*"):
        if not file.is_file() or file.suffix.lower() not in video_extensions:
            continue

        name = file.name.lower()
        if "rgb" not in name or "night" not in name:
            continue

        try:
            print(f"Processing: {file}")

            #### Uncomment for OpenAI version #####
            # caption = get_caption_from_openai(client, file, caption_prompt)
            # question = get_question_from_openai(client, caption, question_prompt)
            # answer = get_answer_from_openai(client, file, question, answering_prompt)

            caption = DEMO_RESULT["caption"]
            question = DEMO_RESULT["question"]
            answer = DEMO_RESULT["answer"]

            results[str(file)] = {
                "caption": caption,
                "question": question,
                "answer": answer
            }

            print(f"Done: {file.name}")

        except Exception as e:
            results[str(file)] = f"ERROR: {e}"
            print(f"Failed: {file.name} -> {e}")

    print(results)
    return results


if __name__ == "__main__":
    run()