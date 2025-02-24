
def remove_continue(text):
  text = text.replace("[CONTINUE]", "")
  # remove white spaces
  text = ' '.join(text.split())
  return text

def preprocess(samples):
    gold_data = []
    for sample in samples:
        gold_data.append(remove_continue(sample['text']))
    
    return gold_data
