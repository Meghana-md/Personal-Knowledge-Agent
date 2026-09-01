from groq import Groq

client = Groq(api_key="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")

models = client.models.list()

for model in models.data:
    print(model.id)