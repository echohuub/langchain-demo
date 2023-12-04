from langchain.tools import BaseTool

class WeatherTool(BaseTool):
    name = "Weather"
    description = "useful for When you want to know about the weather"

    def _run(self, query: str) -> str:
        return "Sunny^_^"

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")

class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math."

    def _run(self, query: str) -> str:
        return "3"

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("BingSearchRun does not support async")