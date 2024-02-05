import os
import openai
import discord
import logging
from discord.ext import commands

# Set logging level to INFO to see all logs
logging.basicConfig(level=logging.DEBUG)

openai.api_key = os.environ.get('OPENAI_API_KEY')
discord_bot_token = os.environ.get('JAYABYRD_TOKEN')
#discord_bot_token = os.environ.get('CLAIRBELLE_TOKEN')
logging.debug(openai.api_key)
logging.debug(discord_bot_token)

intents = discord.Intents.default()
intents.typing = True
intents.presences = True
intents.messages = True
intents.guilds = True 
intents.members = False 
intents.message_content = True

bot = commands.Bot(command_prefix="!", intents=intents)



async def get_gpt_response(sys_prompt, usr_prompt, model, max_tokens, n_responses, creativity):
    logging.debug(sys_prompt)
    try:
        completions = openai.ChatCompletion.create(
            # "gpt-3.5-turbo", "gpt-4"
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": usr_prompt},
            ],
            max_tokens=max_tokens,
            n=n_responses,
            stop=None,
            # specifity < 0.5 > creativity
            temperature=creativity,
        )
        response = completions.choices[0].message['content']
        return response
    except Exception as e:
        exception_error = (f"Error in get_gpt_response: {e}")
        logging.error(exception_error)
        return exception_error

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    if message.channel.topic is not None and "JayaByrd" in message.channel.topic:
        logging.info(f"Message received: {message.content}")
        
        # The bot will show as typing while executing the code inside this block
        # So place your logic that takes time inside this block
        async with message.channel.typing():
            # Remove bot's mention from the message
            clean_message = discord.utils.remove_markdown(str(bot.user.mention))
            prompt_without_mention = message.content.replace(clean_message, "").strip()

            # Get conversation history
            context = await get_message_history(message.channel.id, 3)
            logging.debug(f"Context: {context}")
            # Add context to the prompt
            usr_prompt = f"{context[-1]['text']} {prompt_without_mention}" if context else prompt_without_mention
            usr_prompt = "Rest and take a RUST`/(RUSTy):8:8utteryFlyBreathe: " + usr_prompt
            logging.debug(f"Sending Prompt to ChatGPT")
            response_text = await get_gpt_response("You r da pit8`/ translator 4 da (SlavicRoot) babelPhish(funMet8).", usr_prompt, "gpt-4", 300, 2, 0.55)
            if response_text:
                await message.channel.send(response_text)
            else:
                logging.info("No response from get_gpt_response")
    else:
        logging.info("Bot not mentioned in channel topic")

bot.run(discord_bot_token)


