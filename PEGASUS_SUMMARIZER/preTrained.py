from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load the saved model and tokenizer
model = PegasusForConditionalGeneration.from_pretrained("../preTrainedModel/")
print("Model loaded")
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")
print("tokenizer loaded")

# Example input text
input_text = "Your input text goes here."

# Tokenize and generate summary
inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
summary_ids = model.generate(inputs["input_ids"], max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Print the generated summary
print("Generated Summary:", summary)
