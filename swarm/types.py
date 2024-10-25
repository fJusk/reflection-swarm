from openai.types.chat import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from typing import List, Callable, Union, Optional
from typing_extensions import Self

# Third-party imports
from pydantic import BaseModel, model_validator

AgentFunction = Callable[[], Union[str, "Agent", "Result", dict]]


class Agent(BaseModel):
    name: str = "Agent"
    model: str = "gpt-4o"
    instructions: Union[str, Callable[[], str]] = "You are a helpful agent."
    functions: List[AgentFunction] = []
    tool_choice: str = None
    parallel_tool_calls: bool = True
    has_reflection: bool = False

    @model_validator(mode='after')
    def validate_reflection(self) -> Self:
        if self.has_reflection and len(self.functions) == 0:
            raise ValueError(
                "Reflection capability is enabled (has_reflection=True), but no tools/functions are provided. "
                "An agent must have at least one function to either complete its task, or delegate control to "
                "another agent when reflection is active."
            )

        return self


class Response(BaseModel):
    messages: List = []
    agent: Optional[Agent] = None
    context_variables: dict = {}
    stop_reflection: bool = False


class Result(BaseModel):
    """
    Encapsulates the possible return values for an agent function.

    Attributes:
        value (str): The result value as a string.
        agent (Agent): The agent instance, if applicable.
        context_variables (dict): A dictionary of context variables.
    """

    value: str = ""
    agent: Optional[Agent] = None
    context_variables: dict = {}
    stop_reflection: bool = False
