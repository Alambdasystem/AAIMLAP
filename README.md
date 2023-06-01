# Contextual Chat Facilitator

Contextual Chat Facilitator is a Discord bot that helps manage conversations by answering user questions, recalling specific topics, and summarizing conversations. It utilizes OpenAI's GPT-3.5 Turbo for generating responses and summaries.

## Prerequisites

- An API key from OpenAI (sign up for one at https://beta.openai.com/signup/).
- A Discord bot (follow the instructions at https://discordpy.readthedocs.io/en/stable/discord.html to create one and invite it to your server).

## Features

- Recall past conversations based on a user query.
- Summarize past conversations to give an overview of the discussion.
- Separate chat history for individual channels.

## Installation

1. Clone the repository to your local machine using the command `git clone https://github.com/yourusername/ContextualChatFacilitator.git ~/Code/ContextualChatFacilitator`.

2. Set up a virtual environment by navigating to the `~/Code/ContextualChatFacilitator` directory and running `python -m venv env`.

3. Activate the virtual environment using `source env/bin/activate` (macOS/Linux) or `env\Scripts\activate` (Windows).

4. Install the required dependencies using `pip install -r requirements.txt`.

5. Create a `.env` file in the project's root directory with the following content:

```
    DISCORD_BOT_TOKEN=your_discord_bot_token
    OPENAI_API_KEY=your_openai_api_key
```

Replace `your_discord_bot_token` with your actual Discord bot token, and `your_openai_api_key` with your OpenAI API key.

6. Run the Discord bot using `python chat.py`.

## Usage

- To recall information about a specific topic, type `!recall your_query` in a Discord channel where the bot is active.
- To summarize past conversations, type `!summarize your_query` in a Discord channel where the bot is active.

Replace `your_query` with your desired query to narrow down the context of the summary. For example, `!summarize lunch plans` will summarize all the chat about lunch plans, and `!summarize lunch plans today` will do the same but limit it to just messages regarding today's lunch plans.