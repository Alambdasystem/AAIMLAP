import datetime
import os
from dotenv import load_dotenv
import discord
import pandas as pd
from discord.ext import commands
import openai
import csv
import asyncio
import re
from collections import defaultdict
from pathlib import Path

# Load the environment variables from the .env file
load_dotenv()
print("Loaded environment variables.")

# Getting the Discord Bot Token from environment variables
TOKEN = os.environ.get('DISCORD_BOT_TOKEN')

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

def create_empty_chat_history():
    return pd.DataFrame(columns=['timestamp', 'author', 'content'])

chat_histories = defaultdict(create_empty_chat_history)

# Loading the chat history from the CSV file into the DataFrame
chat_history = pd.read_csv('chat_history.csv', names=['timestamp', 'author', 'content'], skiprows=1)
chat_history['timestamp'] = pd.to_datetime(chat_history['timestamp'], format="%Y-%m-%d %H:%M:%S.%f%z")
chat_history = chat_history.dropna()  # remove blank lines
print("Loaded chat history from CSV.")

recall_role_instructions = "You are a helpful assistant that finds information in a conversation and answers user's questions about what has occurred or been said in this chat"
summarize_role_instructions = "You are a helpful assistant that summarizes a conversation."

def load_chat_history(channel_id):
    csv_file = f'chat_history_{channel_id}.csv'
    if Path(csv_file).exists():  # check if the file exists
        # Load the chat history from the CSV file into the DataFrame
        chat_history = pd.read_csv(csv_file, names=['timestamp', 'author', 'content'], skiprows=1, encoding='ISO-8859-1')
        chat_history['timestamp'] = pd.to_datetime(chat_history['timestamp'], format="%Y-%m-%d %H:%M:%S.%f%z")
        chat_history = chat_history.dropna()  # remove blank lines
        print(f"Loaded chat history for channel {channel_id} from CSV.")
    else:
        chat_history = create_empty_chat_history()
        print(f"No existing chat history for channel {channel_id}. Created new chat history.")
    return chat_history

async def answer_question(query, text):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    def sync_request():
        print("Generating OpenAI Request...")
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": recall_role_instructions},
                {"role": "user",
                    "content": f"Please answer the question or tell me what we discussed regarding ({query}) within this conversation:\n\n{text}"}
            ],
            max_tokens=150,
            n=1,
            temperature=0.7,
        )

    loop = asyncio.get_event_loop()
    print("Executing OpenAI Request...")
    response = await loop.run_in_executor(None, sync_request)
    answer = response.choices[0].message['content'].strip()
    print(f"Answer: {answer}")
    return answer

async def generate_summary(text, query=None):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    if query:
        user_text = f"Please provide a summary of the following conversation, focusing on {query}:\n\n{text}"
    else:
        user_text = f"Please provide a summary of the following conversation:\n\n{text}"

    def sync_requestApologies for the incomplete response. Here's the rest of the code with the modified sections included:

```python
    def sync_request():
        print("Generating OpenAI Request for Summary...")
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": summarize_role_instructions},
                {"role": "user", "content": user_text}
            ],
            max_tokens=150,
            n=1,
            temperature=0.7,
        )

    loop = asyncio.get_event_loop()
    print("Executing OpenAI Request for Summary...")
    response = await loop.run_in_executor(None, sync_request)
    summary = response.choices[0].message['content'].strip()
    print(f"Summary: {summary}")
    return summary

from flask import Flask
app = Flask(__name__)

@app.route('/')
def dockerconfirm():
    return 'Woohoo! Docker container is successfully running on this instance.'

@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')
    print(f"Current Chat History: {chat_history}")
    for guild in bot.guilds:  # iterate over all servers the bot is connected to
            for channel in guild.channels:  # iterate over all channels in the server
                if isinstance(channel, discord.TextChannel):  # make sure it's a text channel
                    # Load the chat history for the channel
                    chat_histories[channel.id] = load_chat_history(channel.id)
    print("Loaded all chat histories.")

@bot.event
async def on_message(message):
    # ... existing on_message event handler code ...

    # Execute all the command coroutines concurrently
    await asyncio.gather(*command_coroutines)

    await bot.process_commands(message)

@bot.command(name='recall')
async def recall(ctx, *args):
    # ... existing recall command code ...

    # Append the recall command coroutine to the list
    command_coroutines.append(recall(ctx, *args))

@bot.command(name='summarize')
async def summarize(ctx, *args):
    # ... existing summarize command code ...

    # Append the summarize command coroutine to the list
    command_coroutines.append(summarize(ctx, *args))

# ... Add more command functions as needed ...

bot.run(TOKEN)
