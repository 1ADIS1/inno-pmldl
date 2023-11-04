from datasets import load_from_disk, load_metric
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import numpy as np


# Initialise models to train and compare.
MODEL_NAMES = [
    #     "google/umt5-small"
    "facebook/bart-base",
    "t5-base"
]


# Load dataset.
cropped_datasets = load_from_disk(
    "../data/interim/para-nmt-preprocessed-cropped")

# Load the BLUE metric
metric = load_metric("sacrebleu")

MAX_INPUT_LENGTH = 128
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 32


def preprocess_function(examples):
    # Tokenizer for inputs
    model_inputs = tokenizer(
        examples['reference'], max_length=MAX_INPUT_LENGTH, truncation=True)

    # Tokenizer for targets
    targets = tokenizer(examples['translation'],
                        max_length=MAX_TARGET_LENGTH, truncation=True)

    model_inputs["labels"] = targets["input_ids"]
    return model_inputs


def get_model_args(model_name: str):
    return Seq2SeqTrainingArguments(
        f"{model_name}-finetuned-tox-to-detox",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=3,
        num_train_epochs=10,
        predict_with_generate=True,
        fp16=True,
        report_to='tensorboard',
        logging_steps=100
    )


# simple postprocessing for text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


# compute metrics function to pass to trainer
def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(
        decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds,
                            references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(
        pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def get_model_trainer(model, args, data_collator, tokenizer, tokenized_datasets):
    return Seq2SeqTrainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )


# Loop through our models
for model_name in MODEL_NAMES:
    print(f"Training {model_name}...")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, model_max_length=MAX_INPUT_LENGTH)

    tokenized_datasets = cropped_datasets.map(
        preprocess_function, batched=True)
    args = get_model_args(model_name)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    trainer = get_model_trainer(
        model, args, data_collator, tokenizer, tokenized_datasets)
    trainer.train()
    trainer.save_model(f"../models/{model_name}_best")
