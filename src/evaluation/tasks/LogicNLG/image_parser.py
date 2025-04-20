from PIL import Image

def parse(samples, image_path="/netscratch/user/TableEval/data/LogicNLG/images"):
    inputs = []
    for sample in samples:
        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            image = image.convert("RGB")
            inputs.append([image.copy(),f'Based on a given table, fill in the entities masked by [ENT] in the following sentence: {sample["template"]}. Output the sentence with filled in masked entities. Table title: {sample["title"]}'])

    return  inputs
