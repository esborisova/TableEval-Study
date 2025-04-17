from PIL import Image

def parse(samples):
    inputs = []
    image_path = 'numericNLG/generated_imgs'

    for sample in samples:
        image = Image.open(f'{image_path}/{sample["image_id"]}')
        image = image.convert("RGB")
        inputs.append([image.copy(), f'Describe the given table focusing on the insights and trends revealed by the results. The summary must be factual, coherent, and well-written. Do not introduce new information or speculate. Table caption: {sample["caption"]}'])
    return  inputs
