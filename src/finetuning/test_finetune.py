from transformers import pipeline


model_path = "C:/Users/jaspi/OneDrive/Desktop/RAG_BASED_PROJECT/finetuned-model"

pipe = pipeline(
    "text-generation",
    model=model_path
)

prompt = """
### Instruction:
What is the company's mission?

### Response:
"""

result = pipe(prompt, max_length=200)

print(result[0]["generated_text"])