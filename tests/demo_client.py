from bentoml.client import Client

client = Client.from_url("http://localhost:3000")

data ="""<s>system
Ты юридический помощник. Ты всегда разговариваешь с людьми, помогаешь им, предоставляя консультации по юридическим вопросам.</s>
<s>user
Привет! Как дела?</s>
<s>bot"""

print(client.prompt(data))
