from PIL import Image

def parse(samples):
    inputs = []
    image_path = '/netscratch/borisova/TableEval/data/ComTQA_data/fintabnet/images'
    for sample in samples:
        image = Image.open(f'{image_path}/{sample["image_name"]}')
    
        inputs.append([image, f'Refer to the provided table image and work through the following question step by step: {sample["question"]}'])
    return  inputs
