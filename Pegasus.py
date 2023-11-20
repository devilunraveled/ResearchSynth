import sys
sys.path.append('..')

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model_name = 'google/pegasus-large'
preTrainedModel = '../preTrainedModel/'


tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(preTrainedModel)

def get_response(input_text):
    global model
    batch = tokenizer([input_text],truncation=True,padding='longest',max_length=1024, return_tensors="pt")
    gen_out = model.generate(**batch,max_length=128,num_beams=5, num_return_sequences=1, do_sample = True, temperature=1.5)
    output_text = tokenizer.batch_decode(gen_out, skip_special_tokens=True)
    return output_text

def GetSummary(researchPaper):
    return get_response(researchPaper)[0]
