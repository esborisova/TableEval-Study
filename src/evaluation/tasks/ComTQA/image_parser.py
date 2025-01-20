from PIL import Image

def parse(samples):
    inputs = []
    pmc_image_path = '/netscratch/borisova/TableEval/data/ComTQA_data/pubmed/images/png'
    mgm_image_path = '/netscratch/borisova/TableEval/data/ComTQA_data/fintabnet/images'
    for sample in samples:
        if sample['image_name'].startswith('PMC'):
            image = Image.open(f'{pmc_image_path}/{sample["image_name"]}')
        else:
            image = Image.open(f'{mgm_image_path}/{sample["image_name"]}')
        inputs.append([image, sample['question']])
    return  inputs
