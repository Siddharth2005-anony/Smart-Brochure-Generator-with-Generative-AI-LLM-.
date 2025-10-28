import os
from dotenv import load_dotenv
from perplexity import Perplexity
import gradio as gr
from bs4 import BeautifulSoup
import requests
import re

# ------------------- LOAD API KEY -------------------
load_dotenv()
api_k = os.getenv("PERPLEXITY_API")
print(f"API Key loaded: {bool(api_k)}")  # don't print the actual key for security

# ------------------- CONTENT SCRAPER -------------------
def coll_content(url):
    """Fetch and clean webpage content."""
    try:
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
    except Exception as e:
        print(f"Unable to fetch URL ({url}): {e}")
        return ""

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove unnecessary elements
    for element in soup(['script', 'style', 'img', 'input', 'noscript']):
        element.decompose()

    # Extract and clean text
    text_content = soup.get_text(separator='\n', strip=True)
    lines = [line.strip() for line in text_content.split('\n') if line.strip()]
    clean_content = '\n'.join(lines)

    print(f"Fetched {len(clean_content)} characters from the site.")
    return clean_content[:8000]  # optional: limit to avoid API overload


# ------------------- API CALL -------------------
def api_call(msg):
    client = Perplexity(api_key=api_k, base_url="https://api.perplexity.ai")

    sys_prompt = """
You are a creative and professional content designer who specializes in writing brochures.  
When given a website URL, analyze its content and create a short, well-structured brochure based 
on the following format:

1. Header / Title  
2. Introduction / Overview  
3. Key Features / Highlights  
4. Benefits / Why Choose Us  
5. Products / Services Section  
6. Call to Action (CTA)  
7. Contact Information  

Keep the tone engaging, concise, and reader-friendly. Use bullet points where suitable, and ensure 
the output feels like a brochure — not a webpage summary.
"""

    # Detect URL in message
    url_pattern = r'(https?://[^\s]+)'
    match = re.search(url_pattern, msg)
    a1 = match.group(0) if match else None

    if not a1:
        return "⚠️ Please include a valid URL in your message."

    # Fetch and clean website content
    soup_content = coll_content(a1)
    if not soup_content:
        return f"⚠️ Could not extract content from {a1}"

    user_prompt = f"{msg}\n\nWebsite content:\n{soup_content}"

    prompt = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # Streaming the model response
    try:
        stream = client.chat.completions.create(
            model="sonar",
            messages=prompt,
            max_tokens=450,
            stream=True
        )

        result = ""
        result = ""
        for chunk in stream:
            if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                result += chunk.choices[0].delta.content
                yield result
    except Exception as e:
        yield f"❌ API call failed: {e}"


# ------------------- GRADIO UI -------------------
view = gr.Interface(
    fn=api_call,
    inputs=gr.Textbox(label='Enter your message (include URL)', lines=6),
    outputs=gr.Textbox(label='Generated Brochure', lines=15),
    allow_flagging='never',
)

if __name__ == "__main__":
    view.launch()
