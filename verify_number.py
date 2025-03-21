import argparse
import json
import os
from io import BytesIO
from PIL import Image
import requests

from picker_model import TargetModel
from gemini_model import GeminiInference
from config import Config

def verify_part_number(page_link, predicted_number, api_keys, gemini_model_name, car_brand):
    """
    Verifies a predicted part number against images from a given page link
    using the Gemini model.

    Args:
        page_link (str): URL of the product page.
        predicted_number (str): The part number predicted by AI.
        api_keys (list): List of API keys for Gemini.
        gemini_model_name (str): Name of the Gemini model to use.
        car_brand (str): Car brand for prompt customization.

    Returns:
        str: Validation result from Gemini model.
    """

    t = TargetModel() # Instantiate TargetModel to access its processor
    model = GeminiInference(api_keys=api_keys, model_name=gemini_model_name, car_brand=car_brand)

    page_img_links = t.processor.parse_images_from_page(page_link)
    page_img_links = list(set(page_img_links))

    if not page_img_links:
        return "No images found on the page."

    # For simplicity, let's just use the first image found on the page for validation.
    # In a real scenario, you might want to use the 'correct_image_link' from your
    # original encoding process if you have that information saved.
    target_image_link = page_img_links[0]

    try:
        response = requests.get(target_image_link, stream=True)
        img_data = BytesIO(response.content)
    except Exception as e:
        return f"Error downloading image: {e}"

    validation_result = model.final_validate_number(predicted_number, img_data, predicted_number)

    return validation_result


def parse_arguments():
    parser = argparse.ArgumentParser(description="Verify a predicted part number using Gemini.")
    parser.add_argument('--page-link', type=str, required=True, help="URL of the product page.")
    parser.add_argument('--predicted-number', type=str, required=True, help="The predicted part number to verify.")
    parser.add_argument('--api-keys', nargs='+', required=True, help="List of API keys for Gemini.")
    parser.add_argument('--gemini-model', type=str, default='gemini-1.5-pro', required=False, help="Gemini model name.")
    parser.add_argument('--car-brand', type=str, required=True, help="Car brand (e.g., audi, toyota).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    validation_result = verify_part_number(
        args.page_link,
        args.predicted_number,
        args.api_keys,
        args.gemini_model,
        args.car_brand
    )

    print("\nVerification Result:")
    print(f"Page Link: {args.page_link}")
    print(f"Predicted Number: {args.predicted_number}")
    print(f"Gemini Validation: {validation_result}")