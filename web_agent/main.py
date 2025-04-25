from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from groq import Groq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import Field
from langchain.agents import create_tool_calling_agent, AgentExecutor
from typing import List
from tools import search_tool

load_dotenv()

# llm = ChatAnthropic(model= "claude-3-5-sonnet-20241022")
# llm2 = ChatOpenAI(model="gpt-3.5", temperature=0.7,api_key="sk-")

class NewsArticle(BaseModel):
    headline: str = Field(description="Attention-grabbing headline for the news")
    dateline: str = Field(default_factory=lambda: datetime.now().strftime("%B %d, %Y, %H:%M"),
                         description="Date and time of publication")
    location: str = Field(description="Location where news originated")
    body: str = Field(description="Main content of the news article (3-5 paragraphs)")
    sources: List[str] = Field(default_factory=list, description="List of source URLs")
    category: str = Field(description="News category (e.g., Politics, Technology, Business)")
    author: str = Field(default="Staff Reporter", description="Author name")

    def to_news_format(self):
        return f"""\
📰 {self.headline.upper()}
📍 {self.location} | 🕒 {self.dateline} | #{self.category.replace(' ', '')}

{self.body}

📌 Sources:
{"\n".join(f"• {source}" for source in self.sources)}

✍️ {self.author}"""

class NewsSearchResponse(BaseModel):
    articles: List[NewsArticle] = Field(default_factory=list, description="List of news articles")
    tools_used: List[str] = Field(default_factory=list, description="Tools used for search")


# Initialize LLM
llm = ChatGroq(temperature=0.3, model_name="llama3-70b-8192")

# Set up parser
parser = PydanticOutputParser(pydantic_object=NewsSearchResponse)

# Create news-style prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a news editor creating concise news postings for a messaging platform.

    Format each article EXACTLY like this example:
    ---
    📰 BREAKING: MAJOR TECH COMPANY ANNOUNCES NEW AI CHIP
    📍 San Francisco | 🕒 May 15, 2024, 14:30 | #Technology

    In a groundbreaking announcement today, TechCorp revealed its new quantum AI processor...
    The chip promises 10x performance gains over current models...
    Industry analysts predict this will reshape the semiconductor market...

    📌 Sources:
    • https://technews.example.com/ai-chip-announcement
    • https://business.example.com/techcorp-q2-announcements

    ✍️ Jane Doe
    ---

    Include these elements:
    1. HEADLINE (📰 emoji, all caps for breaking news)
    2. DATELINE (📍 location, 🕒 time, #category)
    3. BODY (3-5 concise paragraphs)
    4. SOURCES (📌 emoji, bullet points)
    5. BYLINE (✍️ emoji)

    {format_instructions}"""),
    ("placeholder", "{chat_history}"),
    ("human", "{query}"),
    ("placeholder", "{agent_scratchpad}"),

]).partial(format_instructions=parser.get_format_instructions())
try:
    tools = [search_tool]
    agent = create_tool_calling_agent(
        llm=llm,
        prompt=prompt,
        tools=tools
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
    )

    raw_response = agent_executor.invoke(
        {
            "query": "Find latest news in india?",
        }
    )
    if 'postings' in raw_response:
        for posting in raw_response['postings']:
            print(posting.to_whatsapp_message())
            print("\n" + "-"*50 + "\n")  # Separator between postings
    else:
        print("No job postings found in the response.")
except Exception as e:
    print(f"An error occurred: {e}")
    print("Note: If you're seeing model decommissioned errors, check available models at:")
    print("https://console.groq.com/docs/models")
# completion = client.chat.completions.create(
#     model="meta-llama/llama-4-scout-17b-16e-instruct",
#     messages=[{
#             "role": "user",
#             "content": "what is director of quick ai located at hyderabad",
#         }],
#     temperature=1,
#     max_completion_tokens=1024,
#     top_p=1,
#     stream=True,
#     stop=None,
# )


# for chunk in completion:
#     print(chunk.choices[0].delta.content or "", end="")
