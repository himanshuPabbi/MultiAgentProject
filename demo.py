import streamlit as st
import os
from dotenv import load_dotenv
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_anthropic import ChatAnthropic
import requests

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Access API keys from the environment
os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


# Initialize Wikipedia API Wrapper
wikipedia = WikipediaAPIWrapper()

# Initialize the Anthropic chat model
llm = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)

# 🧱 Tool 1: Search Wikipedia
@tool
def search_wikipedia(topic: str) -> str:
    """Search Wikipedia and return a detailed summary of the given topic."""
    return wikipedia.run(topic)

# 📰 Tool 2: Fetch News Articles
@tool
def fetch_news(topic: str) -> str:
    """Fetch recent news articles related to the topic using NewsAPI."""
    url = f"https://newsapi.org/v2/everything?q={topic}&apiKey={NEWS_API_KEY}&pageSize=3"
    response = requests.get(url)
    articles = response.json().get("articles", [])
    news = "\n".join(
        [f"- {a['title']} ({a['source']['name']}): {a['url']}" for a in articles]
    )
    return news or "No news articles found."

# ✍️ Tool 3: Summarize Content
@tool
def summarize_text(text: str) -> str:
    """Summarize the given content into a concise paragraph."""
    prompt = f"Summarize this content concisely:\n\n{text}"
    return llm.invoke(prompt).content

# 💡 Tool 4: Generate Insights (Pros, Cons, Risks)
@tool
def generate_insights(text: str) -> str:
    """Generate structured insights (Pros, Cons, Risks) based on the content."""
    prompt = f"Generate structured insights (Pros, Cons, Risks) from this text:\n\n{text}"
    return llm.invoke(prompt).content

# Initialize agents
agent1 = initialize_agent(
    tools=[search_wikipedia],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent2 = initialize_agent(
    tools=[fetch_news],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent3 = initialize_agent(
    tools=[summarize_text],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

agent4 = initialize_agent(
    tools=[generate_insights],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 🚀 Entry Point Function for Streamlit
def research_assistant(topic: str):
    st.info(f"🔎 Step 1: Fetching Wikipedia Content for '{topic}'...")
    with st.spinner('Searching Wikipedia...'):
        wiki_content = agent1.run(f"Get detailed Wikipedia information about {topic}")
    st.success("✅ Wikipedia content fetched.")
    st.markdown("---")

    st.info(f"📰 Step 2: Fetching News Articles for '{topic}'...")
    with st.spinner('Fetching news...'):
        news_content = agent2.run(f"Fetch recent news articles about {topic}")
    st.success("✅ News fetched.")
    st.markdown("---")

    combined_content = wiki_content + "\n\nRecent News:\n" + news_content

    st.info("✍️ Step 3: Summarizing combined content...")
    with st.spinner('Summarizing...'):
        summary = agent3.run(combined_content)
    st.success("✅ Summary generated.")
    st.markdown("---")

    st.info("💡 Step 4: Generating structured insights...")
    with st.spinner('Generating insights...'):
        insights = agent4.run(summary)
    st.success("✅ Insights generated.")
    st.markdown("---")

    return wiki_content, news_content, summary, insights

# --- Streamlit UI ---
st.title("📚 AI Research Assistant")
st.write("Enter a topic and let the AI assistant perform a comprehensive research and generate a report.")

topic_input = st.text_input("Enter a topic here:", "LangChain")

if st.button("Start Research"):
    if not topic_input:
        st.warning("Please enter a topic.")
    else:
        st.header("--- Final Research Report ---")
        
        wiki_content, news_content, summary, insights = research_assistant(topic_input)

        st.subheader(f"Topic: {topic_input}")
        
        st.subheader("Wikipedia Content")
        st.markdown(wiki_content)

        st.subheader("Recent News")
        st.markdown(news_content)

        st.subheader("Summary")
        st.markdown(summary)

        st.subheader("Insights")
        st.markdown(insights)