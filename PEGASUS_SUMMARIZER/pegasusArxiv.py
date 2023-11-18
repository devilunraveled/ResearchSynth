# from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
# from transformers import DataCollator, DataCollatorWithPadding
# import pickle 
# from datasets import load_dataset
# import torch
#
# def getTokenizer(modelName):
#     # Tokenize the dataset using the same tokenizer
#     tokenizer = PegasusTokenizer.from_pretrained(modelName)
#     print("Tokenizer loaded")
#     return tokenizer
#
# tokenizer = getTokenizer("google/pegasus-large")
#
# def preprocess_function(examples):
#     global tokenizer
#
#     if tokenizer is None:
#         getTokenizer("google/pegasus-large")
#
#     data = {}
#     data["summary"] = tokenizer(examples["abstract"], max_length=512, truncation=True, padding='max_length')
#     data["paper"] = tokenizer(examples["article"], max_length=512, truncation=True, padding='max_length')
#
#     return  {
#         "input_ids" : data["paper"]["input_ids"],
#         "attention_mask": data["paper"]["attention_mask"],  
#         "labels": data["summary"]["input_ids"],
#     }
#
# class MyDataCollator(DataCollatorWithPadding):
#     def collate_batch(self, features):
#         input_ids = torch.stack([torch.tensor(f["input_ids"]) for f in features])
#         attention_mask = torch.stack([torch.tensor(f["attention_mask"]) for f in features])
#         labels = torch.stack([torch.tensor(f["labels"]) for f in features])
#         return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
#
# def loadModel(modelName):
#     # Load pre-trained Pegasus model and tokenizer
#     model = PegasusForConditionalGeneration.from_pretrained(modelName)
#
#     print("Model loaded")
#
#     if model is None:
#         print("Model not found")
#         exit(0)
#
#     return model
#
# def prepareDataset(datasetName):
#     # Using the dataset
#     datasetName = "ccdv/arxiv-summarization"
#     dataset = load_dataset(datasetName)
#
#     print("Dataset loaded")
#     
#     processedDataset = {}
#     # Tokenizing the dataset
#     for split in ["train", "validation"]:
#         processedDataset[split] = dataset[split].map(preprocess_function, batched=True, num_proc=10, remove_columns=["abstract", "article"])
#
#     # return dataset
#     print("Dataset tokenized")
#     return processedDataset
#
# def setTrainingArguments(num_cpus=8):
#     total_batch_size = 4 * num_cpus
#     per_device_train_batch_size = total_batch_size // num_cpus
#     gradient_accumulation_steps = 2  # Adjust as needed
#
#     return TrainingArguments(
#         output_dir="./output",
#         overwrite_output_dir=True,
#         num_train_epochs=3,
#         per_device_train_batch_size=per_device_train_batch_size,
#         save_steps=10000,
#         save_total_limit=3,
#         evaluation_strategy="steps",
#         eval_steps=500,
#         logging_steps=500,
#         learning_rate=2e-5,
#         gradient_accumulation_steps=gradient_accumulation_steps,
#         warmup_steps=500,
#         weight_decay=0.01,
#         logging_dir="./logs",
#     )
#
# def trainModel(model, tokenized_datasets, training_args):
#     # Fine-tune the model
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         data_collator=MyDataCollator(tokenizer),
#         train_dataset=tokenized_datasets["train"],
#         eval_dataset=tokenized_datasets["validation"],
#     )
#     print("Starting training")
#
#     # Run the code to train the model
#     trainer.train()
#     
#     model.save_pretrained("pre-trained-model")
#     fileObject = open("model.pkl", "wb")
#     pickle.dump(model, fileObject)
#     fileObject.close()
#
# if __name__ == "__main__":
#     pegasusBase = loadModel("google/pegasus-large")
#     pegasusTokenizer = tokenizer
#     tokenizedDataset = prepareDataset("ccdv/arxiv-summarization")
#     trainingArgs = setTrainingArguments()
#     trainModel(pegasusBase, tokenizedDataset, trainingArgs)

from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments
import pickle 
from datasets import load_dataset
import torch

def getTokenizer(modelName):
    global tokenizer
    # Tokenize the dataset using the same tokenizer
    tokenizer = PegasusTokenizer.from_pretrained(modelName)
    print("Tokenizer loaded")
    return tokenizer

tokenizer = getTokenizer("google/pegasus-large")

def preprocess_function(examples):
    global tokenizer

    if tokenizer is None:
        getTokenizer("google/pegasus-large")

    data = {}
    data["summary"] = tokenizer(examples["abstract"], max_length=512, truncation=True, padding='max_length')
    data["paper"] = tokenizer(examples["article"], max_length=512, truncation=True, padding='max_length')

    return  {
        "input_ids" : data["paper"]["input_ids"],
        "attention_mask": data["paper"]["attention_mask"],  
        "labels": data["summary"]["input_ids"],
    }


def loadModel(modelName, device):
    # Load pre-trained Pegasus model and tokenizer
    # model_name = "google/pegasus-large"
    model = PegasusForConditionalGeneration.from_pretrained(modelName).to(device)

    print("Model loaded")

    if ( model is None ):
        print("Model not found")
        exit(0)

    return model


def prepareDataset(datasetName, train_subset_size=None, validation_subset_size=None):
    # Using the dataset
    datasetName = "ccdv/arxiv-summarization"
    dataset = load_dataset(datasetName)

    print("Dataset loaded")
    
    processedDataset = {}
    
    # Tokenizing the dataset
    for split in ["train", "validation"]:
        if split == "train" and train_subset_size is not None:
            processedDataset[split] = dataset[split].shuffle(seed=42).select(list(range(train_subset_size)))
        elif split == "validation" and validation_subset_size is not None:
            processedDataset[split] = dataset[split].shuffle(seed=42).select(list(range(validation_subset_size)))
        processedDataset[split] = processedDataset[split].map(preprocess_function, batched=True, num_proc=10, remove_columns=["abstract", "article"])

    # return dataset
    print("Dataset tokenized")
    return processedDataset
# def prepareDataset(datasetName):
#     # Using the dataset
#     datasetName = "ccdv/arxiv-summarization"
#     dataset = load_dataset(datasetName)
#
#     print("Dataset loaded")
#     
#     processedDataset = {}
#     #Tokenizing the dataset
#     for split in ["train", "validation"]:
#         processedDataset[split] = dataset[split].map(preprocess_function, batched=True, num_proc=10, remove_columns=["abstract", "article"])
#
#     # return dataset
#     print("Dataset tokenized")
#     return processedDataset


def setTrainingArguments():
    # Define training arguments
    return TrainingArguments(
        output_dir="./output",                 # Output directory for model checkpoints and predictions
        overwrite_output_dir=True,             # Overwrite the content of the output directory
        num_train_epochs=3,                    # Number of training epochs
        per_device_train_batch_size=5,         # Batch size per GPU
        save_steps=10000,                     # Save checkpoint every X steps
        save_total_limit=3,                    # Limit the total amount of checkpoints to save
        evaluation_strategy="steps",           # Evaluate and save checkpoint every eval_steps
        eval_steps=500,                        # Number of update steps between two evaluations
        logging_steps=500,                     # Log training information every X steps
        learning_rate=2e-5,                    # Learning rate
        gradient_accumulation_steps=4,         # Number of steps to accumulate gradients
        warmup_steps=500,                      # Number of steps for linear warmup
        weight_decay=0.01,                     # Weight decay for regularization
        logging_dir="./logs",                  # Directory for Tensorboard logs
    )


def trainModel(model, tokenized_datasets, training_args):
    # Fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    print("Starting training")

    # Run the code to train the model
    trainer.train()

    model.save_pretrained("outputModel")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    pegasusBase = loadModel("google/pegasus-large", device)
    pegasusTokenizer = tokenizer
    tokenizedDataset = prepareDataset("ccdv/arxiv-summarization", train_subset_size=10000, validation_subset_size=1000)
    trainingArgs = setTrainingArguments()
    trainModel(pegasusBase, tokenizedDataset, trainingArgs)
