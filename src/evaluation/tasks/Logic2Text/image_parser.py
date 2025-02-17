from PIL import Image

def parse(samples):
    inputs = []
    image_path = "/netscratch/borisova/TableEval/data/Logic2Text/images"
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB")
    #        inputs.append([image.copy(), f'Generate a one sentence statement based on the table'])
            inputs.append([image.copy(), f'Generate a one sentence statement based on the table and logical form.\nLogical form: {sample["logic_str"]}\nTable title: {sample["title"]}'])
            #inputs.append([image.copy(), f'Generate a one sentence statement based on the table and logical form.\nOutput the sentence as JSON in the format {{"sentence": "<SENTENCE>"}}.\nLogical form: {sample["logic_str"]}'])
    return  inputs
