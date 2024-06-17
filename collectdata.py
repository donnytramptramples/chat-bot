import json
import random
import time
import os
import signal
from bs4 import BeautifulSoup
import requests

try:
    from googlesearch import search
except ImportError:
    print("google module not found, please install using: pip install google")
    exit()

# Load existing intents or initialize empty
try:
    with open('Intents.json', 'r') as f:
        intents = json.load(f)
except FileNotFoundError:
    intents = {"intents": []}

data_collected = 0

def signal_handler(sig, frame):
    print(f'\nData collection stopped. Total data collected: {data_collected / (1024 * 1024):.2f} MB')
    with open('intents.json', 'w') as f:
        json.dump(intents, f, indent=4)
    exit(0)

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

def generate_random_query():
    # List of random topics or queries of interest
    topics = [
        "interesting facts about animals",
        "latest technology trends",
        "health benefits of meditation",
        "historical events in ancient civilizations",
        "delicious recipes for dinner",
        "top travel destinations in 2024",
        "famous quotes from literature",
        "science discoveries in the last decade",
        "popular myths debunked",
        "tips for productivity and time management"
    ]
    return random.choice(topics)

def update_intents_with_search_results(query):
    global data_collected
    search_results = []
    try:
        for url in search(query, num_results=3, stop=3):  # Limit to 3 results per query
            try:
                response = requests.get(url, timeout=5)
                data_collected += len(response.content)
                soup = BeautifulSoup(response.text, 'html.parser')
                content = soup.get_text(separator=' ', strip=True)
                search_results.append(content[:200])  # Get first 200 characters of the result
            except requests.RequestException as e:
                print(f"Error fetching {url}: {e}")
            time.sleep(1)  # Sleep to avoid getting blocked
    except Exception as e:
        print(f"Error searching for {query}: {e}")

    # Create new intents for each search result
    new_intents = []
    for i, result in enumerate(search_results):
        new_intent = {
            "tag": f"{query.replace(' ', '_')}_search_result_{i+1}",
            "patterns": [query],
            "responses": [result]
        }
        new_intents.append(new_intent)

    intents['intents'].extend(new_intents)

    return intents

print("Collecting data... Press Ctrl+C to stop.")

while True:
    query = generate_random_query()
    print(f"Collecting data for query: {query}")
    
    intents = update_intents_with_search_results(query)

    # Periodically save intents to file
    with open('intents.json', 'w') as f:
        json.dump(intents, f, indent=4)

    print(f"Data collected: {data_collected / (1024 * 1024):.2f} MB")
    time.sleep(10)  # Wait for a while before collecting the next random query

print(f"Data collection complete. Total data collected: {data_collected / (1024 * 1024):.2f} MB")
