from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
import datetime

### Define Custom Tool ###
@tool
def get_current_datetime(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current date and time, formatted according to the provided Python strftime format string.
    Use this tool whenever the user asks for the current date, time, or both.
    Example format strings: '%Y-%m-%d' for date, '%H:%M:%S' for time.
    If no format is specified, defaults to '%Y-%m-%d %H:%M:%S'.
    """
    try:
        return datetime.datetime.now().strftime(format)
    except Exception as e:
        return f"Error formatting date/time: {e}"

tools = [get_current_datetime]
print("Custom tool defined.")


### Set Up the LLM ###
def get_agent_llm(model_name="qwen3:0.6b", temperature=0):
    """Initializes the ChatOllama model for the agent."""
    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        extra_body={"think": False},  # Disable Qwen3 thinking tokens
    )
    print(f"Initialized ChatOllama with model: {model_name}")
    return llm


### Build the Agent (LangGraph) ###
def build_agent(llm, tools):
    agent = create_agent(  # updated function name
        model=llm,
        tools=tools,
    )
    print("LangGraph agent created.")
    return agent


### Run the Agent ###
def run_agent(agent, user_input):
    """Invokes the LangGraph agent and prints the final response."""
    print(f"\nInvoking agent with input: {user_input}")
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        # Final answer is in the last message
        print("\nAgent Response:")
        print(response["messages"][-1].content)
    except Exception as e:
        print(f"\nError during agent execution: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    agent_llm = get_agent_llm(model_name="qwen3:0.6b")
    agent = build_agent(agent_llm, tools)

    run_agent(agent, "What is the current date?")
    run_agent(agent, "What time is it right now? Use HH:MM format.")
    run_agent(agent, "Tell me a joke.")  # Should not use the tool