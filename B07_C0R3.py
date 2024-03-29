# B07_C0R3.py
import logging
# Set logging level to INFO to see all logs
logging.basicConfig(level=logging.INFO)

import asyncio
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import discord
from discord.ext import commands
from discord import Intents  
import emoji  # convert Unicode emojis to their corresponding shortcodes.
import openai
from PIL import Image
import random
import re
import string
import torch

class D15C0R6(commands.Bot):
    def __init__(self, bot_name, openai_api_key, discord_token, bot_init_data):
        self.bot_name = bot_name
        self.openai_api_key = openai_api_key
        self.discord_token = discord_token
        intents = Intents(**bot_init_data["intents"])
        command_prefix = bot_init_data["command_prefix"]
        # Assign all yaml values within the __init__ method
        self.ignored_prefixes = bot_init_data["ignored_prefixes"]
        self.home_channel_id = bot_init_data["home_channel_id"]
        self.username = bot_init_data["username"]
        self.self_channel_id = bot_init_data["self_channel_id"]
        self.self_author_id = bot_init_data["self_author_id"]
        self.self_author_name = bot_init_data["self_author_name"]
        self.bot_channel_id = bot_init_data["bot_channel_id"]
        self.hello_channel_id = bot_init_data["hello_channel_id"]
        # A set ensures that these collections only store unique elements
        self.allow_author_ids = set(bot_init_data["allow_author_ids"])
        self.allow_channel_ids = set(bot_init_data["allow_channel_ids"])
        self.ignore_author_ids = set(bot_init_data["ignore_author_ids"])
        self.ignore_channel_ids = set(bot_init_data["ignore_channel_ids"])
        # Define OpenAI model and response tokens
        self.gpt_model = bot_init_data["gpt_model"]
        self.response_tokens = bot_init_data["response_tokens"]
        # Define Stable Diffusion model
        self.sd_model = bot_init_data["sd_model"]
        self.path_to_sd_model = bot_init_data["path_to_sd_model"]
        self.path_to_saved_images = bot_init_data["path_to_saved_images"]
        self.prePrompt = bot_init_data["prePrompt"]
        self.postPrompt = bot_init_data["postPrompt"]
        self.negPrompt = bot_init_data["negPrompt"]
        self.maxCLIPtokens = bot_init_data["maxCLIPtokens"] # CLIP happens at 77 tokens
        # Parent class assignments for: super().__init__()
        super().__init__(command_prefix=command_prefix, intents=intents)

        # A bot-is-busy switch to buffer incoming messages.
        self.image_generation_in_progress = False    
        self.buffered_messages = []

        # Load the `.safetensor` file for Stable Diffusion
        self.sd_pipe_txt2img = StableDiffusionPipeline.from_single_file(f"{self.path_to_sd_model}/{self.sd_model}") 

        # We are swapping scheduler to DPMSolverMultistepScheduler (DPM-Solver++)) here:
        self.sd_pipe_txt2img.scheduler = DPMSolverMultistepScheduler.from_config(self.sd_pipe_txt2img.scheduler.config)
        logging.info(self.sd_pipe_txt2img.scheduler.compatibles) # show all schedulers
        # (if you don't swap the scheduler it will run with the default DDIM
        # Send the pipeline to the CUDA device
        self.sd_pipe_txt2img = self.sd_pipe_txt2img.to("cuda")
        
    async def on_ready(self):
        logging.info(f"{self.user} is connected to Discord and ready to receive commands.")
        asyncio.create_task(self.process_buffered_messages())

    async def on_disconnect(self):
        await self.session.close()
        logging.info(f'{self.user} has disconnected from Discord.')

    async def close(self):
        await super().close()

    async def process_buffered_messages(self):
        while True:
            if len(self.buffered_messages) > 0 and not self.image_generation_in_progress:
                message = self.buffered_messages.pop(0)
                await self.on_message(message)
            else:
                await asyncio.sleep(1)  # Sleep for a second if there are no buffered messages or image_generation_in_progress is True

    # If you define an on_message event, the bot will not process commands automatically unless you explicitly call `await self.process_commands(message)`. This is because the `on_message`` event is processed before the command, so if you don't call `process_commands`, the command processing stops at `on_message`.
    async def on_message(self, message):
        logging.info(f'-- START ON_MESSAGE --')

        # Target the channel that the message came from.
        message_author_id = message.author.id
        message_author_name = message.author.name
        if self.image_generation_in_progress:
            async with message.channel.typing():
                logging.info(":image_generation_in_progress:")
            if message.content == "!flag":
                logging.info('Not False command issued.')
                self.image_generation_in_progress = False
            self.buffered_messages.append(message)
            return
                    
        elif message.content.startswith('.hello'):
            logging.info(f'.hello from {message.author.name}')
            await message.channel.send(f"Hello {message.author.name}!!")
            
        elif message.channel.id in self.ignore_channel_ids:
            logging.info(f'Ignored Channel ID: {message.channel.name}')

        elif any(message.content.startswith(prefix) for prefix in self.ignored_prefixes):
            for prefix in self.ignored_prefixes:
                if message.content.startswith(prefix):
                    logging.info(f'Ignoring message due to prefix: {prefix}')
            
        elif message_author_id in self.ignore_author_ids:
            logging.info(f'Ignoring message due to ignored author: {message_author_name}')

        elif message.content.startswith('.art') and message.channel.id != 947907332803805204:
            self.image_generation_in_progress = True
            prompt = message.content[5:].strip()
            prompt = self.unicode_to_shortcode(prompt)
            full_prompt = f'{self.prePrompt} {prompt} {self.postPrompt}'
            # full_prompt = f'{prompt}'
            logging.info(f'.art prompt:\n{prompt}\n')
            logging.info(f'.art (full prompt):\n{full_prompt}\n')
            seed, guidance_scale, steps, full_prompt, image_path = self.generate_image(full_prompt)
            logging.info('generate_image returned\n')
            try:
                self.scale_image(image_path, ".scaled_art_image.png", 1440, 1800) # 4:5
                self.add_frame(".scaled_art_image.png", '.frame_image_heads_4x5.png', image_path)
                image_file = discord.File(image_path)
                logging.info(f"File prepared for sending: {image_path}")
            except Exception as e:
                self.image_generation_in_progress = False
                logging.error(f"Error occurred while preparing the image: {e}")
                # You might want to re-raise the exception after logging it, 
                # especially if the subsequent code relies on the success of the previous code
                raise
            # Send the image file to the channel
            await message.channel.send(file=image_file)
            self.image_generation_in_progress = False
            
        elif message.content.startswith('.delete') and (message_author_id == 465419968276594688 or 875422085319622666 or 971580205586067526):
            if message.reference:  # Check if the message is a reply
                try:
                    referenced_message = await message.channel.fetch_message(message.reference.message_id)
                    await referenced_message.delete()
                except Exception as e:
                    await message.channel.send(f"Error deleting message: {e}")
                    logging.error(f"Error deleting message: {e}")
            await message.delete()  # Delete the command message
        
        elif message.content.startswith('.schedulers'):
            classes = self.sd_pipe_txt2img.scheduler.compatibles
            class_names = ""
            for cls in classes:
                cls_str = str(cls)
                cleaned_name = cls_str.replace("<class 'diffusers.schedulers.", "").replace("'>", "")
                cleaned_name = re.sub(r'scheduling_.*?\.', '', cleaned_name)
                class_names += f"{cleaned_name}\n"
            await message.channel.send(f'**compatible schedulers**\n```{class_names}```\n{self.sd_pipe_txt2img.scheduler}')
                    
        elif message_author_id in self.allow_author_ids:
            logging.info(f"Message from {message.author.name} received:\n{message.content}")
            # The bot will show as typing while executing the code inside this block
            # So place your logic that takes time inside this block
            async with message.channel.typing():
                # Remove bot's mention from the message
                clean_message = discord.utils.remove_markdown(str(self.user.mention))
                prompt_without_mention = message.content.replace(clean_message, "").strip()
                # Add context to the prompt
                logging.debug(f"Sending usr_prompt to ChatGPT\n{prompt_without_mention}")
                response_text = get_gpt_response(f"You are my pithy friend. Keep your response under {self.response_tokens} tokens.", prompt_without_mention, self.gpt_model, self.response_tokens, 2, 0.55)
                if response_text:
                    await message.channel.send(f".art:> {response_text}")
                    logging.debug(f"Response text:\n{response_text}")
                else:
                    logging.error("No response from get_gpt_response")
        else:
            if (message_author_id != self.self_author_id):
                ack_who = 'message from another'
                console_ack = f'`message.channel.id`: `{message.channel.id}`,\n`message.channel.name`: `{message.channel.name}`,\n'
                console_ack += f'`message.id`: `{message.id}`,\n'
                console_ack += f'`message.author.id`: `{message_author_id}`, \n`message.author`: `{message.author}`\n'
            else:
                ack_who = 'message from self'
                console_ack = f'`message.channel.id`: `{message.channel.id}`,\n`message.channel.name`: `{message.channel.name}`,\n'
                console_ack += f'`message.id`: `{message.id}`,\n'
                console_ack += f'`message.author.id`: `{message_author_id}`, \n`message.author`: `{message.author}`\n'
            logging.info (f':ACK:> {ack_who}\n{console_ack}')

        # Always process commands at the end of the on_message event
        await self.process_commands(message)
        logging.info(f'-- END ON_MESSAGE --')

    def scale_image(self, input_image_path, output_image_path, width, height):
        try:
            if input_image_path == output_image_path:
                raise ValueError("input_image_path should be different from output_image_path to preserve the original image")
            # Open the input image
            input_image = Image.open(input_image_path)
            # Define the size of the output image
            new_size = (width, height)
            # Check if the image size is different from the target size
            if input_image.size != new_size:
                # Check if the image is being downscaled
                if input_image.size[0] > new_size[0] or input_image.size[1] > new_size[1]:
                    # Use LANCZOS filter for downscaling
                    output_image = input_image.resize(new_size, Image.Resampling.LANCZOS)
                else:
                    # Use BICUBIC filter for upscaling
                    output_image = input_image.resize(new_size, Image.BICUBIC)
            else:
                output_image = input_image
            
            # Save the new image
            output_image.save(output_image_path, "PNG")
        except Exception as e:
            error_message = str(e)
            logging.error(f"scale_image exception error: {error_message}")

    def add_frame(self, source_image_path, frame_image_path, output_image_path):
        try:
            logging.info(f"Adding frame: source={source_image_path}, frame={frame_image_path}, output={output_image_path}")
            # Open the source image and the frame image
            source_image = Image.open(source_image_path).convert("RGBA")
            frame_image = Image.open(frame_image_path).convert("RGBA")
            # Blend the images together, this could be adjusted to your needs
            new_image = Image.alpha_composite(source_image, frame_image)
            # Save the new image
            new_image.save(output_image_path, "PNG")
            logging.info(f"Frame added and new image saved tp: {output_image_path}")
        except Exception as e:
            error_message = str(e)
            logging.error(f"add_frame exception error: {error_message}")

    def unicode_to_shortcode(self, text):
        return emoji.demojize(text)

    def generate_random_string(self):
        alphanumeric_chars = string.digits + string.ascii_letters
        return ''.join(random.choice(alphanumeric_chars) for _ in range(8))

    def generate_image(self, prompt):
        steps = random.randint(29, 39)
        guidance_scale = random.randint(8, 16)
        seed = random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda")
        generator.manual_seed(seed)
        image = self.sd_pipe_txt2img(
            prompt=prompt, 
            #width=768, height=432, # 16:9
            width=512, height=640, # 4:5
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            negative_prompt=self.negPrompt,
            generator=generator
        ).images[0]
        randoname = self.generate_random_string()
        rando_txt = f'{self.path_to_saved_images}/{self.bot_name}-{randoname}.txt'
        rando_jpg = f'{self.path_to_saved_images}/{self.bot_name}-{randoname}.jpg'
        image.save(rando_jpg)
        # Write the prompt to a text file with the same name as the image file
        with open(rando_txt, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Guidance Scale: {guidance_scale}\n")
            f.write(f"Steps: {steps}\n")
        return seed, guidance_scale, steps, prompt, rando_jpg

def get_gpt_response(sys_prompt, usr_prompt, model, max_tokens, n_responses, creativity):
    logging.info(f"System prompt:\n{sys_prompt}")
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