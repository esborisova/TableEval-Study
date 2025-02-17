from PIL import Image

def parse(samples):
    inputs = []
    image_path = '/netscratch/borisova/TableEval/data/ComTQA_data/pubmed/images/png'
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB")
            #inputs.append([image.copy(),f'{sample["question"]}'])
            inputs.append([image.copy(), f'Refer to the provided table and answer the question.\nTable caption: {sample["table_title"]} {sample["table_caption"]}\nTable footnote: {sample["table_footnote"]}\nQuestion: {sample["question"]}'])
    return  inputs
