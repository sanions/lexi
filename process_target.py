from google import genai
from google.genai import types
import json
from PIL import Image
from io import BytesIO

client = genai.Client(api_key="AIzaSyBHMDmFuuc2CpK23wH_oPd0ZqogbnFlSWI") 

def get_definition(target): 

    prompt = f"""
        You'll be given a word, along with the context around the word. 
        Provide a simple, kid-friendly definition of the word in the given context. 
        Your output should be in JSON format: 
        {{'word': <word>, 'definition': <your definition>}}

        The word is: {target['target_word']}.
        The context is: {target['context']}.
    """ 

    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=prompt
    )

    return response.text


def generate_image(target): 
    img_prompt =(f"""
        You'll be given a word, along with the context around the word. 
        Generate a descriptive, kid-friendly image that best depicts the word, in the context provided. 
                 
        Your output should be AN IMAGE. GENERATE an image.

        The word is: {target['target_word']}.
        The context is: {target['context']}.
        """
        )


    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=img_prompt,
        config=types.GenerateContentConfig(
        response_modalities=['TEXT', 'IMAGE']
        )
    )

    try:
        for part in response.candidates[0].content.parts:
            if part.text is not None:
                print(part.text)
            elif part.inline_data is not None:
                image = Image.open(BytesIO((part.inline_data.data)))
                image.save(f"images/{target['target_word']}.png")
                return image
    except Exception as e: 
        print(f"Unable to generate an image: {e}")


        

def process_target(target_fname):

    with open(target_fname, 'r') as f: 
        target = json.load(f)
    
    def_json = get_definition(target)
    print(def_json)

    json_str = def_json.strip().removeprefix("```json").removesuffix("```").strip()
    definition = json.loads(json_str)

    img = generate_image(target)

    return definition, img
    

if __name__ == "__main__":

    with open('saved_frames/triggered/ocr/targets.json', 'r') as f: 
        target = json.load(f)

    # ======= TESTING ========
    target = {
        "target_word": "pitcher", 
        "context": "Mrs. Arable put a pitcher of cream on the table."
    }
    # ========================

    definition = get_definition(target)
    print(definition)
    # img = generate_image(target)

