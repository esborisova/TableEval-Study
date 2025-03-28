from PIL import Image

def parse(samples, image_path='/netscratch/borisova/TableEval/data/ComTQA_data/pubmed/images/png'):
    inputs = []
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB") 
            inputs.append([image.copy(), f'Refer to the provided table and answer the question. Question: {sample["question"]}. Table caption: {sample["table_title"]} {sample["table_caption"]}. Table footnote: {sample["table_footnote"]}.'])
    return  inputs
