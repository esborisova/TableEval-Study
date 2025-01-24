from PIL import Image

def parse(samples):
    inputs = []
    other_image_path = '/netscratch/borisova/TableEval/data/SciGen/test-Other/generated_imgs_other'
    cl_image_path = '/netscratch/borisova/TableEval/data/SciGen/test-CL/generated_imgs_cl_update_2025_01_08'
    for sample in samples:
        if sample['subset'].startswith('other'):
            image = Image.open(f'{other_image_path}/{sample["image_id"]}')
        else:
            image = Image.open(f'{cl_image_path}/{sample["image_id"]}')

        inputs.append([image, ""])
    return  inputs
