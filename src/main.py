from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from tools import WeatherTool
from tools import CustomCalculatorTool

llm = OpenAI(openai_api_key="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", temperature=0)

tools = [WeatherTool(), CustomCalculatorTool()]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

agent.run("Query the weather of this week, And How old will I be in ten years? This year I am 28")
