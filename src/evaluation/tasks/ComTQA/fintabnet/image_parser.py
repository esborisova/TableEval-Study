from PIL import Image

def parse(samples, image_path='/netscratch/borisova/TableEval/data/ComTQA_data/fintabnet/images'):
    inputs = []
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB")
            #  min_size = 28
            #  new_width = max(image.width, min_size)
            # new_height = max(image.height, min_size)
            # image = image.resize((new_width, new_height))
            inputs.append([image.copy(), f'Refer to the provided table and answer the question. Question: {sample["question"]}'])

    return inputs
