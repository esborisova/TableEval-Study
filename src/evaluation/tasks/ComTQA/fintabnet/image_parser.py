from PIL import Image

def parse(samples):
    inputs = []
    image_path = '/netscratch/borisova/TableEval/data/ComTQA_data/fintabnet/images'    
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB")
            inputs.append([image.copy(), f'Refer to the provided table and work through the question step by step. Output the final answer as JSON in the format {{"answer": "<YOUR ANSWER>"}}.\nQuestion: {sample["question"]}'])

    return inputs
