device = "cuda"
import torch
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('t5-small')

# Load model.bin
print("Please enter the input text:")
input_text = input()

model = torch.load("D:\DL5-SmartSense\output-model\output\final\pytorch_model.bin")

with torch.no_grad():
    tokenized_text = tokenizer(input_text, truncation=True, padding=True, return_tensors='pt')

    source_ids = tokenized_text['input_ids'].to(device, dtype = torch.long)
    source_mask = tokenized_text['attention_mask'].to(device, dtype = torch.long)

    generated_ids = model.generate(
        input_ids = source_ids,
        attention_mask = source_mask, 
        max_length=512,
        num_beams=5,
        repetition_penalty=1, 
        length_penalty=1, 
        early_stopping=True,
        no_repeat_ngram_size=2
    )

    pred = tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print("\noutput:\n" + pred)