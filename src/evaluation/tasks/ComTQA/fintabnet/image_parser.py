from PIL import Image

def parse(samples):
    inputs = []

    image_path_fin = '/netscratch/borisova/TableEval/data/ComTQA_data/fintabnet/images'    
    image_path_pmc = '/netscratch/borisova/TableEval/data/ComTQA_data/pubmed/images/png'
    for sample in samples:
        if sample["dataset"]=="FinTabNet":
            image_path = image_path_fin
        else:
            image_path = image_path_pmc

        with Image.open(f'{image_path}/{sample["image_name"]}') as image:
            inputs.append([image.copy(), f'Refer to the provided table image and work through the following question step by step: {sample["question"]}'])

    return inputs
