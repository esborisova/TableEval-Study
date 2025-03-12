from PIL import Image

def parse(samples):
    inputs = []
    other_image_path = '/netscratch/borisova/TableEval/data/SciGen/test-Other/generated_imgs_other'
    cl_image_path = '/netscratch/borisova/TableEval/data/SciGen/test-CL/generated_imgs_cl_update_2025_01_08'
    
    for sample in samples:
        if sample['subset'].startswith('other'):
            file_path = f'{other_image_path}/{sample["image_id"]}'
        else:
            file_path = f'{cl_image_path}/{sample["image_id"]}'
        with Image.open(file_path) as image:
            image = image.convert("RGB")
            inputs.append([image.copy(), f'You are an expert in the table-to-text generation task. Describe the given table focusing on the most important findings reported by reasoning over its content. The text must be coherent. Table caption: {sample["table_caption"]}'])
    return  inputs
