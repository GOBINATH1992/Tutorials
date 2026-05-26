import nest_asyncio
from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser, create_async_playwright_browser
from langchain_openai import OpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent


# Apply nest_asyncio for Jupyter Notebooks or similar environments
#nest_asyncio.apply()
def test_playwright_agent():
    sync_browser = create_sync_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    tools = toolkit.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}
    navigate_tool = tools_by_name["navigate_browser"]
    get_elements_tool = tools_by_name["get_elements"]
    #navigate_tool.run({"url": "https://web.archive.org/web/20230428133211/"})
    # The browser is shared across tools, so the agent can interact in a stateful manner
    #get_elements_tool.run( {"selector": ".container__headline", "attributes": ["innerText"]} 3)
    # If the agent wants to remember the current webpage, it can use the `current_webpage` tool
    #tools_by_name["current_webpage"].run({})
    base_url = "http://127.0.0.1:1234/v1" 
    model_name ="google/gemma-3-4b"
    llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=None, timeout=None, max_retries=2,api_key="llm-studio", base_url=base_url)
    agent_chain = create_react_agent(model=llm, tools=tools)
    result = agent_chain.invoke({"messages": [("user", "Navigate to https://en.wikipedia.org/wiki/English_Wikipedia and tell me the main heading.")]} )
    print(result)
    

async def test_aplaywright_agent():
    async_browser = create_async_playwright_browser()
    toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=async_browser)
    tools = toolkit.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}
    navigate_tool = tools_by_name["navigate_browser"]
    get_elements_tool = tools_by_name["get_elements"]
    
    #result = await navigate_tool.arun({"url": "https://en.wikipedia.org/wiki/English_Wikipedia"})
    #print(result)
    # The browser is shared across tools, so the agent can interact in a stateful manner
    #get_elements_tool.run( {"selector": ".container__headline", "attributes": ["innerText"]} 3)
    # If the agent wants to remember the current webpage, it can use the `current_webpage` tool
    #tools_by_name["current_webpage"].run({})
    base_url = "http://127.0.0.1:1234/v1" 
    model_name ="qwen/qwen3-1.7b"
    llm = ChatOpenAI(model=model_name, temperature=0, max_tokens=None, timeout=None, max_retries=2,api_key="llm-studio", base_url=base_url)
    agent_chain = create_react_agent(model=llm, tools=tools)
    
    query = "what is content of post https://www.newsweek.com/elon-musk-x-latest-posts-tweets-update-2047984"
    query = "Navigate to https://en.wikipedia.org/wiki/English_Wikipedia and tell me the main heading."
    #query = "What are the headers on langchain.com?"
    #result = agent_chain.invoke({"messages": [("user", "What are the headers on langchain.com?")]} )
    result = await agent_chain.ainvoke({"messages": [("user", query)]} )
    print(result['messages'][-1].content)



async def test_aplaywright_loader():
    from langchain_community.document_loaders import PlaywrightURLLoader
    urls = [ "https://en.wikipedia.org/wiki/English_Wikipedia"]
    loader = PlaywrightURLLoader(urls=urls, remove_selectors=["header", "footer"])
    data = await loader.aload()
    print(data[0])
    



# To run this example, you would typically call it within an async context:
if __name__ == '__main__':
    import asyncio
    nest_asyncio.apply()
    asyncio.run(test_aplaywright_agent())
    #test_playwright_agent()
    