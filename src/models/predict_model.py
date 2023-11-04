import time
from tqdm import tqdm
from train_model import *
import json


def translate(model, inference_request, tokenizer) -> str:
    input_ids = tokenizer(inference_request, return_tensors="pt").input_ids
    outputs = model.generate(input_ids=input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True, temperature=0)


test_references_texts = ['ARE YOU FUCKING IDIOT?',
                         "I hate my job because my boss is asshole!", "Honey, you're looking dusgusting today."]

# Test results for each model
test_results = {}

# loading the models and translating sentences with them
for model_name in MODEL_NAMES:
    print(f"Model: {model_name}")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        f'../models/{model_name}_best')
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=MAX_INPUT_LENGTH)

    model.eval()
    model.config.use_cache = False

    for text in test_references_texts:
        print(translate(model, text, tokenizer))
    print()

    # Detoxify test data
    model_test_translations = []
    model_test_labels = []

    tokenized_datasets = cropped_datasets.map(
        preprocess_function, batched=True)

    inference_time_sum = 0
    for i in tqdm(range(len(tokenized_datasets['test']))):

        # Inference
        start_time = time.time()
        model_translation = translate(
            model, tokenized_datasets['test'][i]['reference'], tokenizer)
        end_time = time.time()

        inference_time_sum += end_time - start_time

        model_test_translations.append(model_translation)
        model_test_labels.append(
            [tokenized_datasets['test'][i]['translation']])

    test_results[model_name] = metric.compute(
        predictions=model_test_translations, references=model_test_labels)
    test_results[model_name]['average_inf_time'] = inference_time_sum / \
        len(tokenized_datasets['test'])


# Save test results to json file.
for model_name in MODEL_NAMES:
    with open(f"../references/{model_name}.json", "w") as file:
        json.dump(test_results, file)
