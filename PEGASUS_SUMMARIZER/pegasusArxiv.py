from transformers import PegasusForConditionalGeneration, PegasusTokenizer, Trainer, TrainingArguments

from datasets import load_dataset

tokenizer = None

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

def retrieveDocs(document_embeddings, query_embedding):
    top_k = 15
    similarity_scores = []
    for doc_embedding in document_embeddings:
        similarity = torch.cosine_similarity(query_embedding, doc_embedding)
        similarity_scores.append(similarity.item())

    sorted_indices = sorted(range(len(similarity_scores)), key=lambda k: similarity_scores[k], reverse=True)
    top_documents = [(i, similarity_scores[i]) for i in sorted_indices[:top_k]]

    return top_documents

def getTokenizer(modelName):
    global tokenizer
    # Tokenize the dataset using the same tokenizer
    tokenizer = PegasusTokenizer.from_pretrained(modelName)
    print("Tokenizer loaded")
    return tokenizer


def loadModel(modelName):
    # Load pre-trained Pegasus model and tokenizer
    # model_name = "google/pegasus-large"
    model = PegasusForConditionalGeneration.from_pretrained(modelName)

    print("Model loaded")

    if ( model is None ):
        print("Model not found")
        exit(0)

    return model

def prepareDataset(datasetName):
    # Using the dataset
    datasetName = "ccdv/arxiv-summarization"
    dataset = load_dataset(datasetName)

    print("Dataset loaded")
    
    print(dataset)
    processedDataset = {}
    #Tokenizing the dataset
    for split in ["train", "validation"]:
        processedDataset[split] = dataset[split].map(preprocess_function, batched=True, num_proc=10, remove_columns=["abstract", "article"])

    # return dataset
    print("Dataset tokenized")
    return processedDataset


def setTrainingArguments():
    # Define training arguments
    return TrainingArguments(
        output_dir="./output",                 # Output directory for model checkpoints and predictions
        overwrite_output_dir=True,             # Overwrite the content of the output directory
        num_train_epochs=3,                    # Number of training epochs
        per_device_train_batch_size=1,         # Batch size per GPU
        save_steps=10_000,                     # Save checkpoint every X steps
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

if __name__ == "__main__":
    pegasusBase = loadModel("google/pegasus-large")
    pegasusTokenizer = getTokenizer("google/pegasus-large")
    tokenizedDataset = prepareDataset("ccdv/arxiv-summarization")
    trainingArgs = setTrainingArguments()
    trainModel(pegasusBase, tokenizedDataset, trainingArgs)
