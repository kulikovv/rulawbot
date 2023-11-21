from bentoml.client import Client
import numpy as np
from numpy.linalg import norm

client = Client.from_url("http://localhost:3000")

sentences = ["привет мир", "hello world", "здравствуй вселенная", "самолет летит на юг", "наркоманы гуляли по подъезду"]
codes = []
for s in sentences:
    codes += [client.encode(s)]
codes = np.concatenate(codes)
assert np.allclose(norm(codes[0]),1.)
assert np.dot(codes[0], codes[3]) < np.dot(codes[0], codes[1])
