from PIL import Image

def parse(samples):
    inputs = []
    image_path = "/netscratch/borisova/TableEval/data/Logic2Text/images"
    for sample in samples:
        image = Image.open(f'{image_path}/{sample["image_name"]}')

        inputs.append([image,""])
    return  inputs
