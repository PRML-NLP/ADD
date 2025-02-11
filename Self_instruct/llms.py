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


class MetaAgent:
    def __init__(
        self,
        model_name: str,
        use_openai:bool = False,
        openai_key: str = "EMPTY",
        url: str = "http://localhost:23333/v1",
        system_prompt: str = None,
    ):
        self.key = openai_key
        self.open_api_base = url
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.use_openai = use_openai
        self.llm = self._get_llm()
        self.prompt = self._init_prompt()
        self.prompt_args = dict()
        # self.temperature = 
        
    def _get_llm(self):
        if self.use_openai:
            self.open_api_base="https://api.openai.com/v1"

        llm = ChatOpenAI(model=self.model_name,
                         openai_api_base=self.open_api_base)
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
        
    def generate(
        self,
        human_prompt_template,
        human_prompt_args,
        temprature=0.2,
        top_k=40,
        top_p=0.75,
        max_tokens=5,
        stop=None,
        update_prompt=False,
        reset_prompt=False
    ):
        _old_prompt = copy.deepcopy(self.prompt)
        _old_prompt_args = copy.deepcopy(self.prompt_args)
        self.prompt_args.update(human_prompt_args)
        
        if isinstance(self.llm, ChatOpenAI):
            human_prompt_template = HumanMessage(content=human_prompt_template)
            self.prompt.append(human_prompt_template)
            prompt_template = ChatPromptTemplate.from_messages(self.prompt)
            prompt = prompt_template.format_messages(**self.prompt_args)
        else:
            self.prompt.append("human: " + human_prompt_template)
            prompt_template = PromptTemplate.from_template("\n\n".join(self.prompt))
            prompt = prompt_template.format_prompt(**self.prompt_args)
        
        response = self.llm.invoke(prompt)
        
        output = response.content if isinstance(response, AIMessage) else response
        
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