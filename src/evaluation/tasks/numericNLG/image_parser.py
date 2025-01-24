from PIL import Image

def parse(samples):
    inputs = []
    image_path = '/netscratch/borisova/TableEval/data/numericNLG/generated_imgs'

    for sample in samples:
        image = Image.open(f'{image_path}/{sample["image_id"]}')
        
        inputs.append([image, ""])
    return  inputs
