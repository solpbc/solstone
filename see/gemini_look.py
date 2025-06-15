import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import ImageDraw

# Module-level globals to store the client and system instruction
_gemini_client = None
_system_instruction = None


def initialize():
    """
    Initialize the Gemini API client and load necessary configuration.
    Sets up the module-level globals.
    """
    global _gemini_client, _system_instruction

    # Skip if already initialized
    if _gemini_client is not None:
        return True

    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GOOGLE_API_KEY not found in environment. Please create a .env file.")

    # Load system instruction from external file
    try:
        system_instruction_path = os.path.join(os.path.dirname(__file__), "gemini_look.txt")
        with open(system_instruction_path, "r") as f:
            _system_instruction = f.read().strip()
    except FileNotFoundError:
        print(f"Warning: System instruction file not found at {system_instruction_path}")
        return False

    _gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    return _gemini_client is not None


def gemini_describe_region(image, box, models=None, entities=None):
    """
    Crops the image using native pixel coordinates from box,
    computes normalized coordinates once for the Gemini call, and then
    sends both full image and crop to Gemini.
    Retries with a fallback model if the primary model fails.
    """
    # Ensure the module is initialized
    if _gemini_client is None:
        initialize()

    # Draw bounding box on a copy of the image
    im_with_box = image.copy()
    native_y_min, native_x_min, native_y_max, native_x_max = box["box_2d"]
    draw = ImageDraw.Draw(im_with_box)
    draw.rectangle(
        ((native_x_min, native_y_min), (native_x_max, native_y_max)), outline="red", width=5
    )

    cropped = im_with_box.crop((native_x_min, native_y_min, native_x_max, native_y_max))

    prompt = "Here is the latest screenshot with the cropped region of interest, please return the complete JSON as instructed."
    if entities:
        try:
            with open(entities, "r") as ef:
                entities_text = ef.read().strip()
            if entities_text:
                prompt += (
                    " Here's an incomplete list of entity names you might encounter, they can be useful to help disambiguate some terms: "
                    + entities_text
                )
        except FileNotFoundError:
            print(f"Entities file not found: {entities}")

    models_to_try = (
        models
        if models is not None
        else ["gemini-2.5-flash-preview-05-20", "gemini-2.0-flash", "gemini-2.0-flash-lite"]
    )

    for model_name in models_to_try:
        try:
            response = _gemini_client.models.generate_content(
                model=model_name,
                contents=[prompt, im_with_box, cropped],
                config=types.GenerateContentConfig(
                    temperature=0.5,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                    system_instruction=_system_instruction,
                ),
            )
            print(response.text)
            return json.loads(response.text)
        except Exception as e:
            print(f"Error from Gemini API with model {model_name}: {e}")
            if model_name == models_to_try[-1]:  # If it's the last model in the list
                return None  # All retries failed
            # Otherwise, loop will continue to the next model
    return None
