# ! pip install -q google-generativeai

import google.generativeai as genai
from pathlib import Path
from time import sleep
import random
import logging
import time
import json
import os
from PIL import Image
import requests
import re
import io

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_PROMPT = """Identify the VAG (Volkswagen Audi Group) part number from the photo using this comprehensive algorithm:
1. **Scan the Image Thoroughly:**
   - Examine all text and numbers in the image, focusing on labels, stickers, or embossed areas.
   - Pay special attention to the upper part of labels, areas near barcodes, and any prominent alphanumeric sequences.
2. **Understand Detailed VAG Part Number Structure:**
   - Total length: Typically 11-13 characters (including spaces or hyphens)
   - Format: [First Number] [Middle Number] [Final Number] [Index] [Software Variant]
   
   Example: 5K0 937 087 AC Z15
   
   Detailed Breakdown:
   a) First Number (3 characters):
      - First two digits: Vehicle type (e.g., 3D = Phaeton, 1J = Golf IV, 8L = Audi A3)
      - Third digit: Body shape or variant
        0 = general, 1 = left-hand drive, 2 = right-hand drive, 3 = two-door, 4 = four-door,
        5 = notchback, 6 = hatchback, 7 = special shape, 8 = coupe, 9 = variant
   b) Middle Number (3 digits):
      - First digit: Main group (e.g., 1 = engine, 2 = fuel/exhaust, 3 = transmission, 4 = front axle, 5 = rear axle)
      - Last two digits: Subgroup within the main group
   c) Final Number (3 digits):
      - Identifies specific part within subgroup
      - Odd numbers often indicate left parts, even numbers right parts
   d) Index (1-2 LETTERS): Identifies variants, revisions, or colors
   e) Software Variant (2-3 characters): Often starts with Z (e.g., Z15, Z4)
3. **Identify and Verify with Precision:**
   - The first three parts (First, Middle, Final Numbers) are crucial and must be present.
   - Index and Software Variant may not always be visible or applicable.
   - Check for consistency with known vehicle types and component groups.
4. **Navigate Common Pitfalls and Special Cases:**
   - Character Confusion:
     '1' vs 'I', '0' vs 'O', '8' vs 'B', '5' vs 'S', '2' vs 'Z'
   - Upside-down numbers: Be vigilant for numbers that make sense when flipped.
   - Standard parts: May start with 9xx.xxx or 052.xxx
   - Exchange parts: Often marked with an 'X'
   - Color codes: e.g., GRU for primed parts requiring painting
5. **Context-Based Verification:**
   - Consider the part's apparent function in relation to its number.
   - Check for consistency with visible vehicle model or component type.
   - Look for supporting information like manufacturer logos or additional part descriptors.
Provide the response in this format:
- Valid part number identified: `<START> [VAG Part Number] <END>`
- No valid number found: `<START> NONE <END>`
Include spaces between number segments as shown in the example structure above.
If there are multiple numbers in the image, please identify the one that is most likely to be the correct part number.

**Response Format:**
- If a part number is identified: `<START> [Toyota Part Number] <END>`
- If no valid number is identified: `<START> NONE <END>`
"""


class GeminiInference():
  def __init__(self, api_keys, model_name='gemini-1.5-flash', car_brand=None):
    self.api_keys = api_keys
    self.current_key_index = 0
    self.car_brand = car_brand.lower() if car_brand else None
    self.prompts = self.load_prompts()
    with open("formats.json", "r") as file:
      self.formats = json.load(file)

    self.configure_api()
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
    ]

    self.system_prompt = self.prompts.get(self.car_brand, {}).get('main_prompt', DEFAULT_PROMPT)
    
    self.model = genai.GenerativeModel(
        model_name=model_name,
        generation_config=generation_config,
        safety_settings=safety_settings,
        system_instruction=self.system_prompt
    )

    self.validator_model = self.create_validator_model(model_name)
    self.identify_model = self.create_identify_model(model_name)
    self.incorrect_predictions = []
    self.message_history = []

  def load_prompts(self):
    try:
      with open('prompts.json', 'r') as f:
        return json.load(f)
    except FileNotFoundError:
      logging.warning("prompts.json not found. Using default prompts.")
      return {}

  def configure_api(self):
    genai.configure(api_key=self.api_keys[self.current_key_index])

  def switch_api_key(self):
    self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
    self.configure_api()
    logging.info(f"Switched to API key index: {self.current_key_index}")

  def create_identify_model(self, model_name):
    genai.configure(api_key=self.api_keys[self.current_key_index])
    
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
    ]
    return genai.GenerativeModel(model_name=model_name,
                                 generation_config=generation_config,
                                 safety_settings=safety_settings)

  def create_validator_model(self, model_name):
    genai.configure(api_key=self.api_keys[self.current_key_index])
    
    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_ONLY_HIGH"
        },
    ]
    return genai.GenerativeModel(model_name=model_name,
                                 generation_config=generation_config,
                                 safety_settings=safety_settings)

  def get_response(self, img_data, retry=False):
    max_retries = 10
    base_delay = 5
    
    for attempt in range(max_retries):
        try:
            image_parts = [
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": img_data.getvalue() if isinstance(img_data, io.BytesIO) else img_data.read_bytes()
                    }
                },
            ]
            
            prompt_parts = [' '] if not retry else [
                "It is not correct. Try again. Look for the numbers that are highly VAG number"
            ]
            
            full_prompt = image_parts + prompt_parts
            
            sleep(random.uniform(2, 6))
            
            chat = self.model.start_chat(history=self.message_history)
            response = chat.send_message(full_prompt)
            
            logging.info(f"Main model response: {response.text}")
            
            self.message_history.append({"role": "user", "parts": full_prompt})
            self.message_history.append({"role": "model", "parts": [response.text]})
            
            return response.text
            
        except Exception as e:
            if "quota" in str(e).lower():
                delay = base_delay * (2 ** attempt) + random.uniform(0, 1)
                if delay > 300:
                    self.switch_api_key()
                    delay = base_delay
                logging.warning(f"Rate limit reached. Attempt {attempt + 1}/{max_retries}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Error in get_response: {str(e)}")
                raise
    
    logging.error("Max retries reached. Unable to get a response.")
    raise Exception("Max retries reached. Unable to get a response.")

  def format_part_number(self, number):
    if self.car_brand == 'audi' and re.match(r'^[A-Z0-9]{3}[0-9]{3}[0-9]{3,5}[A-Z]?$', number.replace(' ', '').replace('-', '')):
        number = number.replace('-', '').replace(' ', '')
        
        formatted_number = f"{number[:3]} {number[3:6]} {number[6:9]}"
        
        if len(number) > 9:
            formatted_number += f" {number[9:]}"

        return formatted_number.strip()

    if self.car_brand == "bmw":
        number = number.replace('.', '').replace('-', '').replace(' ', '')
        if re.match(r'^\d{4}\d?\d{6}\d{0,2}$', number):
          formatted_number = f"{number[:2]}.{number[2:4]} {number[4]} {number[5:8]} {number[8:11]}"
          
          if len(number) > 10:
              if len(number) == 11:
                  formatted_number = f"{number[:2]}.{number[2:4]} {number[4]} {number[5:8]} {number[8:11]}"
              elif len(number) > 11:
                  formatted_number += f" {number[11:]}"
          
          return formatted_number.strip()
        else:
          return number

    else:
        return number

  def extract_number(self, response):
    number = response.split('<START>')[-1].split("<END>")[0].strip()
    if number.upper() != "NONE":
      return self.format_part_number(number)
    return number

  def validate_number(self, extracted_number, img_data, car_brand=None):
    genai.configure(api_key=self.api_keys[self.current_key_index])
    
    formatted_number = self.format_part_number(extracted_number)
    
    image_parts = [
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_data.getvalue() if isinstance(img_data, io.BytesIO) else img_data.read_bytes()
            }
        },
    ]
    
    if car_brand == None:
        validation_prompt = self.prompts.get(self.car_brand, {}).get('validation_prompt', "")
        
    else:
        validation_prompt = self.prompts.get(car_brand, {}).get('validation_prompt', "")

    incorrect_predictions_str = ", ".join(self.incorrect_predictions)
    prompt = validation_prompt.format(extracted_number=extracted_number, incorrect_predictions=incorrect_predictions_str)

    prompt_parts = [
        image_parts[0],
        prompt,
    ]
    
    response = self.validator_model.generate_content(prompt_parts)
    
    logging.info(f"Validator model response: {response.text}")
    return response.text

  def final_validate_number(self, extracted_number, img_data, predicted_number):
    #Checking format
    if self.car_brand in self.formats:
      brand_formats = self.formats[self.car_brand]["format"].split(",")
      number = extracted_number

      normalized_number = number.replace("-", " ")

      number_parts = normalized_number.split()
      flag = True #Extracted number is incorrect in any format
      for brand_format in brand_formats:
        format_parts = list(map(int, brand_format.split("-")))

        if len(number_parts) != len(format_parts):
          continue

        for part, expected_length in zip(number_parts, format_parts):
          if len(part) != expected_length:
            continue
        flag = False

      if flag:
        logging.info(f"Final Validator model response: Wrong Format")
        return "<START>NONE<END>"

    genai.configure(api_key=self.api_keys[self.current_key_index])
      
    formatted_number = self.format_part_number(extracted_number)
      
    image_parts = [
        {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": img_data.getvalue() if isinstance(img_data, io.BytesIO) else img_data.read_bytes()
            }
        },
    ]

    prompt = [
        f"Your task is to identify the number {predicted_number} on the provided image. ",
        "Check carefully to see if you can find that exact number in the picture. There may be errors in the number - check each character",
        "If you find the number clearly visible, return it as it is. ",
        "If you cannot find the number, return 'NONE'. ",
        "All segments must be clearly defined\n   - No mixing of 'O' (letter) with '0' (number)\n   - No mixing of 'B' (letter) with '8' (number)\n   - No mixing of 'S' (letter) with '5' (number)\n   - No mixing of 'I' (letter) with '1' (number)",
        f"If the sticker with the number is torn return '!{predicted_number}", 
        "IF THE PHOTO QUALITY IS POOR, OR THE STICKER IS NOT CLEARLY VISIBLE (THE STICKER MUST OCCUPY A LARGE AREA OF THE PHOTO) YOU MUST RETURN 'NONE'",
        "Explanation: [Brief explanation of why it's valid or invalid, including the number itself and any concerns about it being upside-down]",
        "Respond strictly in the format: <START>your_response<END>."
    ]

    prompt = "".join(prompt)

    prompt_parts = [
        image_parts[0],
        prompt,
    ]
      
    response = self.validator_model.generate_content(prompt_parts)
      
    logging.info(f"Final Validator model response: {response.text}")
    return response.text.split('<START>')[-1].split("<END>")[0].strip()

  def reset_incorrect_predictions(self):
    self.incorrect_predictions = []
    self.message_history = []

  def __call__(self, image_path):
    self.configure_api()
    
    if image_path.startswith('http'):
        response = requests.get(image_path, stream=True)
        img_data = io.BytesIO(response.content)
    else:
        img = Path(image_path)
        if not img.exists():
            raise FileNotFoundError(f"Could not find image: {img}")
        img_data = img

    self.message_history = []

    max_attempts = 2
    for attempt in range(max_attempts):
        answer = self.get_response(img_data, retry=(attempt > 0))
        extracted_number = self.extract_number(answer)
        
        logging.info(f"Attempt {attempt + 1}: Extracted number: {extracted_number}")
        
        if extracted_number.upper() != "NONE":
            validation_result = self.validate_number(extracted_number, img_data)
            if "<VALID>" in validation_result:
                extracted_number = self.final_validate_number(extracted_number, img_data, extracted_number)
                if extracted_number != "NONE" and extracted_number != "<START>NONE<END>": #extracted_number may be "NONE" after final validation
                  logging.info(f"Valid number found: {extracted_number}")
                  self.reset_incorrect_predictions()
                  return self.format_part_number(extracted_number)
            else:
                logging.warning(f"Validation failed")
                self.incorrect_predictions.append(extracted_number)
                if attempt < max_attempts - 1:
                    logging.info(f"Attempting to find another number (Attempt {attempt + 2}/{max_attempts})")
        else:
            logging.warning(f"No number found in attempt {attempt + 1}")
            if attempt < max_attempts - 1:
                logging.info(f"Attempting to find another number (Attempt {attempt + 2}/{max_attempts})")
    
    self.reset_incorrect_predictions()
    logging.warning("All attempts failed. Returning NONE.")
    return "NONE"
