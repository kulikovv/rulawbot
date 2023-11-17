from bentoml.client import Client

client = Client.from_url("http://localhost:3000")


print(client.encode("This is a test message"))
