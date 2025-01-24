from PIL import Image

def parse(samples):
    inputs = []
    image_path = '/netscratch/borisova/TableEval/data/ComTQA_data/pubmed/images/png'
    for sample in samples:
        image = Image.open(f'{image_path}/{sample["image_name"]}')
        
        inputs.append([image, f'Refer to the provided table image, its caption and footnote, and work through the question step by step.\nTable caption: {sample["table_title"]} {sample["table_caption"]}\nTable footnote: {sample["table_footnote"]}\nQuestion: {sample["question"]}'])
    return  inputs
