# Sourcing Financial News

The world of finance is inextricably linked with the world of news, each influencing the other in a never-ending dance. Many ‘financial market engineers’ strive to decode this relationship, but the first step is collecting the right financial news data. How do you do that? Where do you start? Here, we’ve laid out a quick, cost-free setup to get you started on gathering and analyzing financial news. Perfect for those looking to dive into the intricate web of market trends and news without spending a dime.

[![Open in GitHub](https://img.shields.io/badge/%7C%20-Open%20in%20GitHub-blue?logo=github)](https://github.com/DavideGCosta/sciqnt-pages/tree/main/static/notebooks/docs/strategies/01-financial-news/01-sourcing-news.ipynb)
[![Open in Colab](https://img.shields.io/badge/%7C-Open%20in%20Colab-orange?style=flat&logo=googlecolab)](https://colab.research.google.com/github/DavideGCosta/sciqnt-pages/blob/tree/main/static/notebooks/docs/strategies/01-financial-news/01-sourcing-news.ipynb)
[![Open in Codespaces](https://img.shields.io/badge/%7C-Open%20in%20Codespaces-darkblue?style=flat&logo=git-for-windows&logoColor=white)](https://github1s.com/DavideGCosta/sciqnt-pages/tree/main/static/notebooks/docs/strategies/01-financial-news/01-sourcing-news.ipynb)

## Background

In the fast-paced world of financial news and behavioral finance, each piece of information contains both fundamental and non-fundamental elements. The challenge lies in sifting through the overwhelming noise and massive influx of data that surpass human processing capabilities. But what if we could identify patterns within this data? Better yet, what if we could use computers to do it for us? Today, we are fortunate to have access to open-source models that make text comprehensible to machines. This process, known as ‘Sentence Embeddings,’ has revolutionized AI and deep learning. By converting textual data into meaningful numerical representations, sentence embeddings empower us to discern market sentiment, spot emerging trends, and predict financial movements with unprecedented efficiency and accuracy.

## Introduction
The first step in building a strategy that leans on financial news to analyze patterns and market sentiment is sourcing and ingesting those news items. Just like any data-driven analysis, the quality of your input data determines the quality of your output. Remember, garbage in, garbage out; no state-of-the-art model can compensate for poor data.

So, what makes a good dataset of financial news?

- **Comprehensive Coverage:** It should pull from a wide range of sources, including major financial news outlets and industry-specific publications. This ensures a broad perspective on market events and sentiments.

- **Timeliness:** Financial markets move quickly, so the dataset should be updated frequently to capture the latest news and developments. Real-time or near real-time data is ideal.

- **Relevance:** The news articles should be pertinent to the financial markets, covering topics like stock market updates, economic indicators, corporate earnings, mergers and acquisitions, regulatory changes, and geopolitical events.

- **Diversity of Perspectives:** Including news from various geographical regions and economic sectors can provide a more holistic view of the market. This helps in understanding global market dynamics and sector-specific trends.

- **Quality and Credibility:** The sources should be reputable and known for their accuracy and reliability. This reduces the risk of misinformation affecting analysis and investment decisions.

- **Metadata:** Each news item should come with useful metadata, such as publication date, source, author, geographical region, and topic tags. This aids in filtering, sorting, and analyzing the data more effectively.

- **Sentiment Analysis Ready:** The dataset should be in a format conducive to natural language processing and sentiment analysis. Clean, well-structured text data with minimal noise is ideal.

- **Historical Data:** Access to historical news data allows for trend analysis over time, which can be crucial for identifying long-term patterns and correlations.

By ensuring these characteristics, a financial news dataset becomes a robust tool for analyzing market conditions and identifying investment opportunities.

## Ingesting Financial News

### Financial News Sources 

For instance, a solid and relevant pool of Financial News Providers might look something like this. We’ve got the usual suspects with broad, comprehensive coverage, but we also delve into niche territory with specialized providers for Technology, Commodity and Energy Markets, and Healthcare. This mix ensures we’re not missing out on the big picture while keeping an eye on the nitty-gritty details in specific sectors.


```python
news_sources = {
    "General": {
        "Reuters": "reuters.com",
        "Financial Times": "ft.com",
        "CNBC": "cnbc.com",
        "MarketWatch": "marketwatch.com",
        "The Economist": "economist.com",
        "Yahoo Finance": "finance.yahoo.com",
        "Benzinga": "benzinga.com",
        "Investing.com": "investing.com",
    },
    "Technology Trends": {
        "Wired": "wired.com",
        "Ars Technica": "arstechnica.com",
    },
    "Commodity and Energy Markets": {
        "OilPrice": "oilprice.com",
        "Rigzone": "rigzone.com",
    },
    "Healthcare Developments": {
        "Modern Healthcare": "modernhealthcare.com",
        "BioSpace": "biospace.com",
    }
}
```

### Financial News Semantic Search

Alternatively, or as a complement to the previous sources, you can dive into the world of semantic search by keywords in financial news. This method casts a wide net, scooping up news from virtually any source, offering an impressively comprehensive search. However, there’s a catch—it also dredges up a fair share of irrelevant news and content from potentially dubious providers. So, before you dive into analysis, you’ll need a solid spam filter to sift out the noise and ensure you’re not swimming in a sea of non-credible stuff.


```python
news_topics = {
    "Market Dynamics": {
        "Title": "Market Dynamics",
        "Description": "Exploring the various factors that influence financial markets, including interest rates, inflation, employment, economic growth, consumer spending, and the housing market.",
        "Subtopics": {
            "Interest Rates": {
                "Title": "Interest Rates & Monetary Policy",
                "Description": "An analysis of how interest rates and central bank policies influence financial markets.",
                "Keywords": "\"Interest Rate\" OR \"Federal Reserve\" OR \"FED\" OR \"Central Bank\""
            } # add more ...
        }
    },
    "Geopolitical Events": {
        "Title": "Geopolitical Events",
        "Description": "Insights into key global political events, including elections, trade dynamics, international relations, military conflicts, EU matters, and global summits.",
        "Subtopics": {
            "Elections": {
                "Title": "Global Elections",
                "Description": "Coverage of major elections and political campaigns worldwide.",
                "Keywords": "\"Election\" OR \"Political Campaign\" OR \"Political Development\""
            } # add more ...
        }
    },
    "Regulatory Changes": {
        "Title": "Regulatory Changes",
        "Description": "Tracking significant shifts in tax, monetary and fiscal policies, antitrust, environmental, and financial regulations.",
        "Subtopics": {
            "Tax Policies": {
                "Title": "Tax Policy Changes",
                "Description": "Updates on tax reforms, legislation, and changes in tax policies.",
                "Keywords": "\"Tax Reform\" OR \"Tax Legislation\" OR \"Tax Policy\" OR \"Tax Cut\" OR \"Tax Hike\""
            }# add more ...
        }
    },
    "Technology Trends": {
        "Title": "Technology Trends",
        "Description": "Insights into the latest advancements in AI, blockchain, cybersecurity, telecommunication, automotive, and renewable energy technologies.",
        "Subtopics": {
            "AI and Machine Learning": {
                "Title": "AI and Machine Learning Innovations",
                "Description": "Exploring advancements in artificial intelligence and machine learning technologies.",
                "Keywords": "\"Artificial Intelligence\" OR \"Machine Learning\" OR \"AI ML\""
            } # add more ...
        }
    },
    "Corporate Actions": {
        "Title": "Corporate Actions",
        "Description": "A focus on financial reports, mergers, public offerings, stock activities, and dividend policies of corporations.",
        "Subtopics": {
            "Financial Reports": {
                "Title": "Corporate Financial Reporting",
                "Description": "Analysis of earnings reports, financial statements, and annual and quarterly reports.",
                "Keywords": "\"Earnings\" OR \"Earnings Report\" OR \"Financial Statement\" OR \"Quarterly Report\" OR \"Annual Report\" OR \"10-K\" OR \"10-Q\""
            }# add more ...
        }
    },
    "Commodity and Energy Markets": {
        "Title": "Commodity and Energy Markets",
        "Description": "In-depth analysis of commodity markets including oil, precious metals, agricultural commodities, and renewable energy sources.",
        "Subtopics": {
            "Oil and Petroleum": {
                "Title": "Oil and Petroleum Dynamics",
                "Description": "Exploration of the oil and petroleum markets, focusing on trends and economic impacts.",
                "Keywords": "\"Oil\" OR \"Crude\" OR \"Petroleum\" OR \"Energy\""
            }# add more ...
        }
    },
    "Healthcare Developments": {
        "Title": "Healthcare Developments",
        "Description": "Exploration of key factors in healthcare including pandemics, medical innovations, health policies, and pharmaceutical advances.",
        "Subtopics": {
            "Pandemics": {
                "Title": "Pandemic Outbreaks",
                "Description": "Focus on the impact and management of global health crises and pandemics.",
                "Keywords": "\"Pandemic Outbreak\" OR \"Global Health Crisis\""
            } # add more ...
        }
    },
    "Environmental and Social Issues": {
        "Title": "Environmental and Social Issues",
        "Description": "Focus on critical environmental and social challenges like climate change, sustainable investing, corporate governance, and social movements.",
        "Subtopics": {
            "Climate Change": {
                "Title": "Climate Change Impacts",
                "Description": "Analysis of climate change effects and the response to global warming.",
                "Keywords": "\"Climate Change\" OR \"Global Warming\" OR \"Carbon Emission\" OR \"Carbon Tax\""
            }# add more ...
        }
    },
    "Market Sentiment Indicators": {
        "Title": "Market Sentiment Indicators",
        "Description": "A comprehensive look at the indicators that reflect market sentiment, including volatility, options market, short selling, and market surveys.",
        "Subtopics": {
            "Market Volatility": {
                "Title": "Market Volatility Analysis",
                "Description": "Analysis of market volatility and its implications for investors and traders.",
                "Keywords": "\"Volatility Index\" OR \"VIX\" OR \"Market Volatility\""
            }# add more ...
        }
    },
    "Economic Indicators": {
        "Title": "Economic Indicators",
        "Description": "Analysis of key economic indicators including PMI, consumer prices, housing market, and industrial production.",
        "Subtopics": {
            "Purchasing Managers Index": {
                "Title": "Purchasing Managers Index Insights",
                "Description": "In-depth analysis of the PMI and its implications for the economy.",
                "Keywords": "\"Purchasing Managers' Index\" OR \"PMI\""
            },
            "Consumer Prices": {
                "Title": "Consumer Price Trends",
                "Description": "Examination of consumer price indices and inflation.",
                "Keywords": "\"Consumer Price Index\" OR \"Inflation\" OR \"CPI\""
            }# add more ...
        }
    }
}

```

### Collecting the Data

Now that we’ve identified our sources and keywords, it’s time to pull the news data. There are several methods to do this, ranging from free to premium, but let’s focus on the budget-friendly options:

- **Financial News APIs:** Free tiers of services like Alpha Vantage, NewsAPI, and Finnhub provide real-time and historical financial news data. This method offers the simplest access but can be limited in scope and data availability.

- **Web Scraping:** Build your own web scrapers using tools like Beautiful Soup or Scrapy in Python to extract data from financial news websites. However, this requires adherence to each site’s scraping policies, and handling captchas and logins can make it quite challenging.

- **RSS Feeds:** Some sources offer free RSS feeds you can subscribe to for pulling information. The downside is you’ll need to handle each source individually, and not all sources provide RSS feeds.

- **News Aggregators:** Platforms like Google News and Feedly consolidate news from multiple sources in one place, standardizing the information. While they typically offer only news titles, summaries, and metadata, this is often sufficient for initial analysis.

Google News also provides an RSS feed, simplifying the process of pulling financial news. For this reason, we’ll use it as our example.

By running the snippet below you can leverage the Google News Aggregator to pull the financial news for your `news_sources` and `news_topics` defined above. 

The snippet can be split as follows:
- `make_DTawareUCT`: Imagine you’ve got a bunch of datetime strings lounging around with no clue what timezone they belong to. That’s where this function steps in, takes these naive datetime strings, and whisks them into the UTC timezone.

- `clean_description`: You know how news descriptions come plastered with HTML tags and clutter? Using BeautifulSoup, it scrapes off all those pesky tags and unwanted spaces, leaving behind just the clean, pure text. It’s like giving your news description a nice shower, ready for the spotlight in your analysis.

- `get_googleNewsRss`: Now, here’s the main event. This function crafts a search URL with all your secret ingredients—topics, subtopics, keywords, and timing. It then sends this URL out into the web and waits for the XML data to come back. Once it’s got the goods, it parses through the XML, picking out relevant bits like titles, links, descriptions, and sources. It even checks the publication date’s timezone and cleans up the description’s HTML mess. Each piece of news gets its own little data packet, and before you know it, you’ve got a DataFrame full of fresh, neatly organized news articles.


```python
%%capture
# Install the necessary packages
!pip install requests pandas beautifulsoup4 lxml pytz tabulate

import warnings, os
# Suppress all warnings - remove it when testing the notebook to see the warnings
warnings.simplefilter("ignore")
```


```python
# Importing the necessary dependencies
import requests
import xml.etree.ElementTree as ET
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
import random

# Convert naive datetimes to aware datetimes in UTC
def make_DTawareUCT(naive_datetime_str, dtformat):
    naive_datetime = datetime.strptime(naive_datetime_str, dtformat)
    aware_datetime = pytz.utc.localize(naive_datetime)
    return aware_datetime

# Clean HTML content in news descriptions 
def clean_description(html_description):
    if not html_description:
        return ""
    soup = BeautifulSoup(html_description, 'html.parser')
    # Find all text in the description and concatenate it
    return ' '.join(soup.stripped_strings)

# Fetch Google News RSS feeds
def get_googleNewsRss(topic='', subtopic='', keyword=False, inUrl=False, when=False, logger=False):
    # Construct the search query URL
    keyword_str = f"{keyword}" if keyword is not False else ""
    inUrl_str = f"inurl:{inUrl}" if inUrl is not False else ""
    when_str = f"when:{when}" if when is not False else ""
    url = f"https://news.google.com/rss/search?q={keyword_str}+{inUrl_str}+{when_str}&hl=en-US&gl=US&ceid=US:en"
    # Make the request to Google News RSS
    xml_data = None
    response = requests.get(url)
    if response.status_code == 200:
        xml_data = response.text
    else:
        raise Exception(f"Failed to get RSS feed from {url}")
    if xml_data == None:
        return pd.DataFrame()
    else:
        # Parse the XML data
        root = ET.fromstring(xml_data)
        items_list = []
        for item in root.findall('.//item'):
            # Extract necessary information from each item
            title = item.find('title').text if item.find('title') is not None else None
            link = item.find('link').text if item.find('link') is not None else None
            description = item.find('description').text if item.find('description') is not None else None
            source = item.find('source').text if item.find('source') is not None else None
            pubDate = item.find('pubDate').text if item.find('pubDate') is not None else None
            pubDate = make_DTawareUCT(pubDate, '%a, %d %b %Y %H:%M:%S GMT')
            # Clean the description HTML content
            clean_desc = clean_description(description)
            item_data = {
                'topic': topic,
                'subtopic': subtopic,
                'title': title,
                'link': link,
                'description': clean_desc,
                'source': source,
                'pubDate' : pubDate,
                'tags' : keyword_str.replace("\"", "").split(" OR ") if keyword_str !="" else [],
                'Headline Event': ''
            }
            items_list.append(item_data)
        newsDf = pd.DataFrame(items_list)
    return newsDf

# Example usage: 
when = "1d" # gets news from the past day -- see google news search syntax for more options
list_news_bySources = []
for topic in news_sources.keys():
    for _, sourceUrl in news_sources[topic].items():
        # Append news data from each source
        list_news_bySources.append(get_googleNewsRss(topic, subtopic='', inUrl=sourceUrl, when=when))

list_news_byTopics = []
topics = list(news_topics.keys()) 
random.shuffle(topics)
for topic in topics:
    topic_data = news_topics[topic]
    # Convert subtopics to a list and shuffle
    subtopics = list(topic_data['Subtopics'].keys())
    random.shuffle(subtopics)
    for subtopic in subtopics:
        subtopic_data = topic_data['Subtopics'][subtopic]
        # Append news data from each randomly selected subtopic
        list_news_byTopics.append(get_googleNewsRss(topic, subtopic, keyword=subtopic_data['Keywords'], when=when))

list_AllNews = list_news_byTopics + list_news_byTopics
df_AllNews = pd.concat(list_AllNews, ignore_index=True)
```

### Financial News Stats

Now that we’ve meticulously gathered our financial news articles and compiled them into the df_AllNews DataFrame, it’s time to dive into the data and extract some meaningful insights. With a rich dataset at our disposal, we can move beyond merely collecting news to analyzing it for patterns, trends, and actionable intelligence. To achieve this, we’ll utilize the show_interesting_stats function, which will provide a comprehensive overview of our dataset. This function will dissect the DataFrame, revealing the number of articles, the diversity of sources, the time span of the news, and the most frequently mentioned keywords and topics. By examining these statistics, we can gain a clearer understanding of the current financial news landscape and uncover the dominant narratives shaping the market.


```python
from collections import Counter
import re

def extract_keywords_from_title(title):
    # Simple keyword extraction by splitting words and removing common stopwords
    stopwords = set(['and', 'or', 'the', 'in', 'on', 'a', 'an', 'for', 'with', 'of', 'to', 'by'])
    words = re.findall(r'\w+', title.lower())
    keywords = [word for word in words if word not in stopwords and len(word) > 2]
    return keywords

def show_interesting_stats(df):
    if df.empty:
        print("# The DataFrame is empty.")
        return

    # Number of articles
    num_articles = len(df)
    
    # Number of unique sources
    num_sources = df['source'].nunique()
    
    # Time range covered by the articles
    min_date = df['pubDate'].min()
    max_date = df['pubDate'].max()
    
    # Most common sources
    sources_counts = df['source'].value_counts().head(5)
    
    # Extract keywords from titles
    all_keywords = df['title'].apply(lambda title: extract_keywords_from_title(title))
    keyword_counts = Counter([keyword for keywords in all_keywords for keyword in keywords]).most_common(5)
    
    # Most common topics
    topic_counts = df['topic'].value_counts().head(5)
    
    print(f"### Interesting Stats")
    print(f"- **Number of articles:** {num_articles}")
    print(f"- **Number of unique sources:** {num_sources}")
    print(f"- **Time range covered:** {min_date} to {max_date}")
    
    print("\n#### Most Common Keywords in Titles")
    for keyword, count in keyword_counts:
        print(f"- **{keyword}:** {count}")
    
    print("\n#### Most Common Sources")
    for source, count in sources_counts.items():
        print(f"- **{source}:** {count}")
    
    print("\n#### Most Common Topics")
    for topic, count in topic_counts.items():
        print(f"- **{topic}:** {count}")

# Example usage
show_interesting_stats(df_AllNews)
```

    ### Interesting Stats
    - **Number of articles:** 1512
    - **Number of unique sources:** 368
    - **Time range covered:** 2024-07-10 19:42:15+00:00 to 2024-07-11 19:34:30+00:00
    
    #### Most Common Keywords in Titles
    - **inflation:** 244
    - **news:** 194
    - **climate:** 186
    - **earnings:** 172
    - **change:** 162
    
    #### Most Common Sources
    - **Yahoo Finance:** 136
    - **Bloomberg:** 58
    - **Reuters:** 58
    - **MarketWatch:** 42
    - **ABC News:** 36
    
    #### Most Common Topics
    - **Economic Indicators:** 222
    - **Environmental and Social Issues:** 194
    - **Technology Trends:** 194
    - **Market Dynamics:** 182
    - **Corporate Actions:** 182
    

**That's all. The really cool stuff comes from employing machine learning models to the dataset - take a look at our next articles for an understanding of how to do it.**
