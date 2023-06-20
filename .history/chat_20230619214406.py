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
from transformers import AlbertForQuestionAnswering, AlbertTokenizer
import torch


# Load the environment variables from the .env file
load_dotenv()
print("Loaded environment variables.")

# Getting the Discord Bot Token from environment variables
TOKEN = os.environ.get('DISCORD_BOT_TOKEN')

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True

bot = commands.Bot(command_prefix='!', intents=intents)

model_name = 'albert-base-v2'  # Replace with the desired ALBERT model
model = AlbertForQuestionAnswering.from_pretrained(model_name)
tokenizer = AlbertTokenizer.from_pretrained(model_name)

def create_empty_chat_history():
    return pd.DataFrame(columns=['timestamp', 'author', 'content'])

#chat_histories = defaultdict(create_empty_chat_history)
chat_histories = {}

# Loading the chat history from the CSV file into the DataFrame
chat_history = pd.read_csv('chat_history.csv', names=[
                           'timestamp', 'author', 'content'], skiprows=1)
chat_history['timestamp'] = pd.to_datetime(
    chat_history['timestamp'], format="%Y-%m-%d %H:%M:%S.%f%z")
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
    inputs = tokenizer.encode_plus(query, text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model(**inputs)
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits

    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    answer = tokenizer.decode(input_ids[answer_start:answer_end], skip_special_tokens=True)

    return answer

async def generate_summary(text, query=None):
    if query:
        user_text = f"Please provide a summary of the following conversation, focusing on {query}:\n\n{text}"
    else:
        user_text = f"Please provide a summary of the following conversation:\n\n{text}"

    inputs = tokenizer.encode_plus(user_text, add_special_tokens=True, return_tensors="pt")
    input_ids = inputs["input_ids"].tolist()[0]

    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

async def answer_question(query, text):
    openai.api_key = os.environ.get("OPENAI_API_KEY")

    def sync_request():
        print("Generating OpenAI Request...")
        return openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": recall_role_instructions},
                {"role": "user",
                    "content": f"Please answer the question or tell me what we discussed regarding ( {query} ) within this conversation:\n\n{text}"}
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

    if not os.path.isfile('chat_history.csv'):
        with open('chat_history.csv', 'w', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(['timestamp', 'author', 'content'])

    for guild in bot.guilds:  # iterate over all servers the bot is connected to
        for channel in guild.channels:  # iterate over all channels in the server
            if isinstance(channel, discord.TextChannel):  # make sure it's a text channel
                # Load the chat history for the channel
                chat_histories[channel.id] = load_chat_history(channel.id)
    print("Loaded all chat histories.")
    
@bot.event
async def on_message(message):
    global chat_histories

    if message.author == bot.user:
        return

    content = message.content

    # Get the chat history for the current channel
    chat_history = chat_histories[message.channel.id]

    # If the message is not a command
    if not content.startswith('!'):
        # Append the new message to the chat history DataFrame with the timestamp
        chat_history.loc[len(chat_history)] = {
            'timestamp': message.created_at, 'author': message.author.name, 'content': content}

        print(f"Added message to chat history of channel {message.channel.id}")

        # Append the new message to the CSV file
        with open(f'chat_history_{message.channel.id}.csv', 'a', newline='', encoding='utf-8') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(
                [message.created_at, message.author.name, content])

        print(f"Added message to CSV for channel {message.channel.id}")

        # Update the chat_histories dictionary
        chat_histories[message.channel.id] = chat_history

    await bot.process_commands(message)

@bot.command(name='recall')
async def recall(ctx, *args):
    query = ' '.join(args)
    max_tokens = 3000

    chat_history = chat_histories[ctx.channel.id]

    relevant_history = chat_history[
        ~(chat_history['author'] == bot.user.name) & 
        ~(chat_history['content'].str.startswith('!')) & 
        (chat_history['content'].str.contains('|'.join(args), case=False, na=False))
    ].tail(10)
    print(f"relevant_history message{relevant_history}")

    # Estimate token count
    estimated_tokens = relevant_history['content'].apply(lambda x: len(x.split()))

    # Split messages into two parts if estimated token count exceeds the limit
    if estimated_tokens.sum() > max_tokens:
        # Find the index where the cumulative sum of tokens exceeds the limit
        split_index = estimated_tokens.cumsum().searchsorted(max_tokens)[0]

        # Split the messages
        part1 = "\n".join(
            f"{row.timestamp} - {row.author}: {row.content}" for _, row in relevant_history.iloc[:split_index].iterrows()
        )
        part2 = "\n".join(
            f"{row.timestamp} - {row.author}: {row.content}" for _, row in relevant_history.iloc[split_index:].iterrows()
        )

        # Send two recall requests
        summary1 = await answer_question(query, part1)
        summary2 = await answer_question(query, part2)
        summary = f"Part 1: {summary1}\nPart 2: {summary2}"
    else:
        conversation_text = "\n".join(
            f"{row.timestamp} - {row.author}: {row.content}" for _, row in relevant_history.iterrows()
        )

        summary = await answer_question(query, conversation_text)

    await ctx.send(summary)


@bot.command(name='summarize')
async def summarize(ctx, *args):
    query = ' '.join(args) if args else None

    chat_history = chat_histories[ctx.channel.id]

    relevant_history = chat_history[~(chat_history['author'] == bot.user.name) & ~(
        chat_history['content'].str.startswith('!'))]

    conversation_text = "\n".join(
        f"{row.timestamp} - {row.author}: {row.content}" for _, row in relevant_history.iterrows())

    summary = await generate_summary(conversation_text, query)
    await ctx.send(summary)
    
# @bot.command(name='createproject')
# async def create_project(ctx, project_name):
#     # Check if the project name is provided
#     if not project_name:
#         await ctx.send("Please provide a project name.")
#         return

#     # Check if the project already exists
#     if project_name in projects:
#         await ctx.send("A project with that name already exists.")
#         return

#     # Create a new project with the specified name
#     project = {
#         'name': project_name,
#         'tasks': [],
#         'members': [],
#         'status': 'Active'
#     }

#     # Add the project to the projects list or database
#     projects[project_name] = project

#     await ctx.send(f"New project '{project_name}' created successfully.")


# @bot.command(name='test')
# async def test_command(ctx, *args):
#     # Process the test command arguments
#     if len(args) == 0:
#         await ctx.send("Please provide test arguments.")
#         return

#     # Perform the desired test actions
#     # ...

#     # Send the test results or output
#     # ...

#     await ctx.send("Test command executed.")


#@bot.command(name='addtask')
#async def add_task(ctx, project_name, task_description, assignee, deadline):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task description is provided
    #if not task_description:
        #await ctx.send("Please provide a task description.")
        #return

    # Check if the assignee is provided
    #if not assignee:
        #await ctx.send("Please provide an assignee for the task.")
        #return

    # Check if the deadline is provided
    #if not deadline:
        #await ctx.send("Please provide a deadline for the task.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Get the project from the projects list or database
    #project = projects[project_name]

    # Create a new task with the provided details
    #task = {
        #'description': task_description,
        #'assignee': assignee,
        #'deadline': deadline,
        #'status': 'In Progress'
    #}

    # Add the task to the project's task list
    #project['tasks'].append(task)

    #await ctx.send(f"Task added successfully to project '{project_name}'.")
    
    
#@bot.command(name='viewtask')
#async def view_task(ctx, project_name, task_id):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Get the project from the projects list or database
    #project = projects[project_name]

    # Check if the task ID is valid
    #if task_id < 0 or task_id >= len(project['tasks']):
        #await ctx.send("Invalid task ID.")
        #return

    # Retrieve the task with the given ID from the project
    #task = project['tasks'][task_id]

    # Prepare the detailed information for display
    # ...

    # Send the task information as a response
    # ...

    #await ctx.send("Task information displayed successfully.")
    
    
#@bot.command(name='updatetask')
#async def update_task(ctx, project_name, task_id, new_description):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the new description is provided
    #if not new_description:
        #await ctx.send("Please provide a new description for the task.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Get the project from the projects list or database
    #project = projects[project_name]

    # Check if the task ID is valid
    #if task_id < 0 or task_id >= len(project['tasks']):
        #await ctx.send("Invalid task ID.")
        #return

    # Update the description of the task with the new description
    # ...

    #await ctx.send("Task description updated successfully.")
    
#@bot.command(name='archiveproject')
#async def archive_project(ctx, project_name):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Archive the project
    # ...

    #await ctx.send(f"Project '{project_name}' archived successfully.")
    
#@bot.command(name='removetask')
#async def remove_task(ctx, project_name, task_id):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Remove the task from the project
    # ...

    #await ctx.send(f"Task {task_id} removed successfully from project '{project_name}'.")
    
#@bot.command(name='completetask')
#async def complete_task(ctx, project_name, task_id):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Mark the task as completed
    # ...

    #await ctx.send(f"Task {task_id} marked as completed in project '{project_name}'.")
    
#@bot.command(name='assigntask')
#async def assign_task(ctx, project_name, task_id, new_assignee):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the new assignee is provided
    #if not new_assignee:
        #await ctx.send("Please provide a new assignee for the task.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Change the assignee of the task
    # ...

    #await ctx.send(f"Assignee of task {task_id} updated in project '{project_name}'.")


#@bot.command(name='changetaskdeadline')
#async def change_task_deadline(ctx, project_name, task_id, new_deadline):
    # Check if the project name is provided
    #if not project_name:
        #await ctx.send("Please provide a project name.")
        #return

    # Check if the task ID is provided
    #if not task_id:
        #await ctx.send("Please provide a task ID.")
        #return

    # Check if the new deadline is provided
    #if not new_deadline:
        #await ctx.send("Please provide a new deadline for the task.")
        #return

    # Check if the project exists
    #if project_name not in projects:
        #await ctx.send("Project not found.")
        #return

    # Change the deadline of the task
    # ...

    #await ctx.send(f"Deadline of task {task_id} updated in project '{project_name}'.")
bot.run(TOKEN)
