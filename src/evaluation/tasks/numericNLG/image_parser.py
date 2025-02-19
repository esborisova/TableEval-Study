from PIL import Image

def parse(samples):
    inputs = []
    image_path = '/netscratch/borisova/TableEval/data/numericNLG/generated_imgs'

    for sample in samples:
        image = Image.open(f'{image_path}/{sample["image_id"]}')
        image = image.convert("RGB")
        inputs.append([image.copy(), f'You are an expert in the table-to-text generation task. Describe the given table focusing on the insights and trends revealed by the results. The summary should be factual and well-written. Do not introduce new information or speculate.\nTable caption: {sample["caption"]}'])
    return  inputs
