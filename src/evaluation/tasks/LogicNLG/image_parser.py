from PIL import Image

def parse(samples):
    inputs = []
    image_path = "/netscratch/borisova/TableEval/data/LogicNLG/images"
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB")
            inputs.append([image.copy(),f'Based on a given table, fill in the entity masked by [ENT] in the following sentence: {sample["template"]}. Output the sentence with filled in masked entities.\Table title: {sample["title"]}'])

           # inputs.append([image.copy(),f'Read the table and then fill in the entity masked by [ENT] in the sentence.\nSentence:{sample["template"]}'])
    return  inputs
