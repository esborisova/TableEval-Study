import json
import os
import re
import argparse

import pathlib
import time
import zipfile
import google.generativeai as genai
import PIL.Image

"""
A script that uses Gemini API to get LaTeX code for the tables from the SciGen and NumericNLG datasets.
The model is prompted using only the image of the table. 
To run the script the following arguments are needed:
1. data_path: the path to where the SciGen or NumericNLG data is saved.
2. images_path: the path for the directory that contains the images of the tables from the dataset. 
3. gemini_api_key: the Google/Gemini API key to access and prompt the model.
"""

def extract_latex_code(text):
  """Extracts LaTeX code from text enclosed within ```."""
  if text == None:
    return None

  pattern = r'```latex\n(.*?)\n```'
  match = re.search(pattern, text, re.DOTALL)
  if match:
    return match.group(1)
  else:
    return text
    
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path for the SciGen dataset")
    parser.add_argument("--images_path", type=str, help="Path for the images of the SciGen dataset")
    parser.add_argument("--gemini_api_key", type=str, help="API key to access and prompt the Gemini model")
    args = parser.parse_args()

    GOOGLE_API_KEY=args.gemini_api_key
    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-1.5-flash')

    images_path = args.images_path
    data_path = args.data_path

    with open(data_path) as json_file:
        data = json.load(json_file)

    latex_img = []
    latex_img_src = []

    count = 1
    for key, value in data.items():
      
      if value['table_latex'] == None:
        print(count)
        count += 1
        caption = value['table_caption']
        image_id = value['image_id']
        
        try:
          image_path = os.path.join(images_path, image_id)
          img = PIL.Image.open(image_path)
          
          prompt_img = f"""Task: Given an image  of a table from a scholarly articl along with its caption, generate the latex code that creates the table.
          Caption: {caption}"""
          
          try:
            response_img = model.generate_content([prompt_img, img])
            latex_img.append((key, response_img.text))
            latex_img_src.append((key, 'gemini'))
            print("Got response from image")
          
          except:
            latex_img.append((key, None))
            
          time.sleep(10)
        except:
          latex_img.append((key, value['table_latex']))
          latex_img_src.append((key, value['table_latex_source']))
          print("No image found")
          
      else:
        latex_img.append((key, value['table_latex']))
        latex_img_src.append((key, value['table_latex_source']))

    for idx, item in enumerate(latex_img):
      key = item[0]
      value = item[1]
      data[key]['table_latex'] = extract_latex_code(value)
      
      try:
        data[key]['table_latex_source'] = latex_img_src[idx][1]
      
      except:
        data[key]['table_latex_source'] = "source_code"


if __name__ == "__main__":
    main()