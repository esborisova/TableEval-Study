from PIL import Image

def parse(samples):
    inputs = []
    image_path = "/netscratch/user/TableEval/data/Logic2Text/images"
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB")
    
            inputs.append([image.copy(), f'Generate a one sentence statement based on the table and logical form. Logical form: {sample["logic_str"]}. Table title: {sample["title"]}'])
    return  inputs
