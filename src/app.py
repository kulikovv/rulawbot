import chainlit as cl
from bentoml.client import Client
from langchain import PromptTemplate, LLMChain



class Conversation:
    def __init__(self, handle, verbose=True) -> None:
        self.system = """
        <s>system
        Ты юридический помощник. Ты всегда разговариваешь с людьми, помогаешь им, предоставляя консультации по юридическим вопросам.</s>
        """
        self.data = ""
        self.handle = handle
        self.verbose = verbose

    def add_user_message(self, txt):
        self.data += f"""<s>user
        {txt}</s>
        """

    def add_bot_response(self, txt):
        self.data += f"""<s>bot
        {txt}</s>
        """

    def prompt(self, txt):
        self.add_user_message(txt)
        message = self.system + self.data + "<s>bot\n"
        if self.verbose:
            print(message)
        return self.handle.prompt(message)


    

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    client = Client.from_url("http://localhost:3000")

    llm_chain = Conversation(client)

    # Store the chain in the user session
    cl.user_session.set("llm_chain", llm_chain)


@cl.on_message
async def main(message: cl.Message):
    # Retrieve the chain from the user session
    llm_chain = cl.user_session.get("llm_chain")  # type: LLMChain
    
    res = llm_chain.prompt(message.content)
    llm_chain.add_bot_response(res)
    
    # Do any post processing here

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res).send()
