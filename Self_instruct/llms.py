import os
import copy
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage
)

from typing import List
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from lmdeploy.serve.openai.api_client import APIClient


class MetaAgent:
    def __init__(
        self,
        model_name: str,
        top_k:int=40, # 최대 몇개의 samples를 만들 것인가?
        top_p:float=0.75, # 얼마나 신뢰도 높은 것들을 허용할 것인가?
        max_tokens=5,
        use_openai:bool = False,
        openai_key: str = "EMPTY",
        url: str = "http://localhost:23333/v1",
        system_prompt: str = None
    ):
        self.key = openai_key
        self.open_api_base = url
        self.model_name = model_name
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.use_openai = use_openai
        self.temperature = 0.7
        self.llm = self._get_llm()
        self.prompt = self._init_prompt()
        self.prompt_args = dict()
        
    def _get_llm(self):
        if self.use_openai:
            self.open_api_base="https://api.openai.com/v1"

        api_client = APIClient('http://192.168.0.80:23333')
        self.model_name = api_client.available_models[0]
        print(self.model_name)
        llm = ChatOpenAI(model=self.model_name,
                        temperature = self.temperature,
                        openai_api_base="http://172.17.0.1:23333/v1",
                        openai_api_key=os.getenv("OPENAI_API_KEY"),
                        n = self.top_k,
                        top_p = self.top_p,
                        max_tokens = self.max_tokens,
                        verbose=True,
                        callbacks = []
                        )
        return llm
    
    def _init_prompt(self):
        if not self.system_prompt:
            return []

        if isinstance(self.llm, ChatOpenAI):
            system_prompt_template = SystemMessage(content=self.system_prompt)
            return [system_prompt_template]
        else:
            system_prompt_template = "system: " + self.system_prompt
            return [system_prompt_template]
    

    def gen(
        self,
        human_prompt_template,
        human_prompt_args,
        temperature=0.2,
        stop=None,
        update_prompt=False,
        reset_prompt=False
    ):

        _old_prompt = copy.deepcopy(self.prompt)
        _old_prompt_args = copy.deepcopy(self.prompt_args)
        self.prompt_args.update(human_prompt_args)

        if self.temperature != temperature:
            self.temperature = temperature
            self.llm = self._get_llm()


        if isinstance(self.llm, ChatOpenAI):
            human_prompt_template = HumanMessage(content=human_prompt_template)
            self.prompt.append(human_prompt_template)
            prompt_template = ChatPromptTemplate.from_messages(self.prompt)
            prompt = prompt_template.format_messages(**self.prompt_args)
        else:
            self.prompt.append("human: " + human_prompt_template)
            prompt_template = PromptTemplate.from_template("\n\n".join(self.prompt))
            prompt = prompt_template.format_prompt(**self.prompt_args)
        print("prompt:",prompt)
        response = self.llm.invoke(prompt,frequency_penalty=0.2)
        # response = self.llm.generate(prompt)
        
        output = response.content if isinstance(response, AIMessage) else response
        print("**output**:",response.content)
        
        
        # Use the generate method to get multiple responses
        # test = self.llm.generate([prompt])
        # print("**LONG:", test.generations)
        # for i, g in enumerate(test.generations[0],1):
        #     print(f"**Samples {i}**:",g.message.content)
        # exit()

        if update_prompt:
            if isinstance(self.llm, ChatOpenAI):
                ai_prompt_template = AIMessage.from_template(output)
                self.prompt.append(ai_prompt_template)
            else:
                self.prompt.append(output)
        else:
            self.prompt = _old_prompt
            self.prompt_args = _old_prompt_args
            
        if reset_prompt:
            self._init_prompt()

        return output