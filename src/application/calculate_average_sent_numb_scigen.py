import json
import spacy


def average_number_of_sentences(descriptions, nlp):
    total_sentences = 0

    for description in descriptions:
        doc = nlp(description)
        total_sentences += len(list(doc.sents))

    average_sentences = total_sentences / len(descriptions) if descriptions else 0
    return average_sentences


def main():
    nlp = spacy.load("en_core_web_lg")
    scigen_files = [
        "../../data/SciGen/test-CL/test_CL_all_meta_with_latex_updated_2024_10_15.json",
        "../../data/SciGen/test-Other/test_Other_all_meta_with_latex_update_2024_10_14.json",
    ]
    data_list = []

    for file in scigen_files:
        with open(file) as f:
            data = json.load(f)
        data_list.append(data)

    descriptions = [
        instance[key]["text"] for instance in data_list for key in instance.keys()
    ]
    # substituting the tag since it indicates that a sentence continues elsewhere
    cleaned_descriptions = [
        desc.replace("[CONTINUE]", ".").strip() for desc in descriptions
    ]
    average_sent_numb = average_number_of_sentences(cleaned_descriptions, nlp)
    print(f"Average number of sentences: {average_sent_numb:.2f}")


if __name__ == "__main__":
    main()
