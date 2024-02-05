# Y4ML807.py
# Set logging level to INFO to see all logs
logging.basicConfig(level=logging.INFO)

# import libraries
import logging
import random
import re
import asyncio
import discord                 # pip install discord
from discord.ext import commands
from discord import Intents  
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from transformers import GPT2Tokenizer
# Define da tokenizer module
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

class D15B17(commands.Bot):
    def __init__(self, openai_api_key, discord_token, bot_init_data):
        self.openai_api_key = openai_api_key
        self.discord_token = discord_token
        # Parent class assignments for: super().__init__()
        intents = Intents(**bot_init_data["intents"])
        command_prefix = bot_init_data["command_prefix"]
        
        # Assign all yaml values within the __init__ method
        self.ignored_prefixes = bot_init_data["ignored_prefixes"]
        self.home_channel_id = bot_init_data["home_channel_id"]
        self.username = bot_init_data["username"]

        self.self_channel_id = bot_init_data["self_channel_id"]
        self.self_author_id = bot_init_data["self_author_id"]
        self.self_author_name = bot_init_data["self_author_name"]
        self.other_author_id = bot_init_data["other_author_id"]
        self.other_channel_id = bot_init_data["other_channel_id"]
        self.bot_channel_id = bot_init_data["bot_channel_id"]
        self.hello_channel_id = bot_init_data["hello_channel_id"]

        # Define Stable Diffusion model
        self.sd_model_id = bot_init_data["sd_model_id"]
        self.prePrompt = bot_init_data["prePrompt"]
        self.postPrompt = bot_init_data["postPrompt"]
        self.negPrompt = bot_init_data["negPrompt"]

        self.prePrompt_tokens = tokenizer.tokenize(self.prePrompt)
        self.postPrompt_tokens = tokenizer.tokenize(self.postPrompt)
        self.negPrompt_tokens = tokenizer.tokenize(self.negPrompt)

        self.maxCLIPtokens = bot_init_data["maxCLIPtokens"]

        # A set ensures that these collections only store unique elements
        self.allow_author_ids = set(bot_init_data["allow_author_ids"])
        self.allow_channel_ids = set(bot_init_data["allow_channel_ids"])
        self.ignore_author_ids = set(bot_init_data["ignore_author_ids"])
        self.ignore_channel_ids = set(bot_init_data["ignore_channel_ids"])
        
        super().__init__(command_prefix=command_prefix, intents=intents)

        self.busy = False
        # a bot-is-busy switch to buffer incoming messages.
        # would like to start with True and set False after on_ready() finishes.
        # however, on_ready is using .art command, which Jaya also responds too.
        self.image_generation_in_progress = False    
        self.buffered_messages = []

        self.should_continue = True

        # There are two important arguments to know for loading variants:
        # `torch_dtype` defines the floating point precision of the loaded checkpoints. For example, if you want to save bandwidth by loading a fp16 variant, you should specify torch_dtype=torch.float16 to convert the weights to fp16. Otherwise, the fp16 weights are converted to the default fp32 precision. You can also load the original checkpoint without defining the variant argument, and convert it to fp16 with torch_dtype=torch.float16. In this case, the default fp32 weights are downloaded first, and then they’re converted to fp16 after loading.
        # `variant` defines which files should be loaded from the repository. For example, if you want to load a non_ema variant from the diffusers/stable-diffusion-variants repository, you should specify variant="non_ema" to download the non_ema files.
        #  variant="fp16"; variant="non_ema"
        # Non-exponential mean averaged (EMA) weights which shouldn’t be used for inference. You should use these to continue finetuning a model (img2img)

        # Enable NSFW (if not `safety_checker=None`` in StableDiffusionPipeline.from_pretrained() )
        #self.sd_pipe_txt2img.safety_checker = lambda images, clip_input: (images, False) 
        # Load the Stable Diffusion model - Move to YAML init file.
        self.sd_pipe_txt2img = StableDiffusionPipeline.from_single_file("/home/cisnez/Downloads/pirsusEpicRealism_pirsusEpicRealismV40.safetensors") 
        #self.sd_model_id = "prompthero/openjourney"          # works 512x640 # Include 'mdjrny-v4 style' in prompt.
        #self.sd_pipe_txt2img = StableDiffusionPipeline.from_pretrained(self.sd_model_id, safety_checker=None)
        self.sd_pipe_components = self.sd_pipe_txt2img.components
        #self.sd_pipe_img2img = StableDiffusionImg2ImgPipeline(**self.sd_pipe_components)

        # (if you don't swap the scheduler it will run with the default DDIM
        logging.debug(self.sd_pipe_txt2img.scheduler.compatibles)
        # We are swapping it to DPMSolverMultistepScheduler (DPM-Solver++)) here:
        self.sd_pipe_txt2img.scheduler = DPMSolverMultistepScheduler.from_config(self.sd_pipe_txt2img.scheduler.config)
        # Send the pip to CUDA
        self.sd_pipe_txt2img = self.sd_pipe_txt2img.to("cuda")
        
        # Define wrapper_tokens and its token count
        self.wrapper_string = f"{self.prePrompt}§{self.postPrompt}"
        self.wrapper_char_count = len(self.wrapper_string)
        #self.wrapper_tokens = word_tokenize(self.wrapper_string)
        self.wrapper_tokens = tokenizer.tokenize(self.wrapper_string)
        self.wrapper_token_count = len(self.wrapper_tokens)

        # print the init
        logging.info(f'\nself.wrapper_string:\n{self.wrapper_string}\n')
        logging.info(f'\nself.wrapper_tokens:\n{self.wrapper_tokens}\n')
        logging.info(f'self.wrapper_char_count: {self.wrapper_char_count}')
        logging.info(f'self.wrapper_token_count: {self.wrapper_token_count}')
        logging.info(f'self.q4_tokens: {self.q4_tokens}')
        logging.info(f'self.q4_char_count: {self.q4_char_count}\n')

    async def on_ready(self):
        logging.info(f"{self.user} is connected to Discord and ready to receive commands.")
        # Get the home channel by its ID
        # channel = self.get_channel(self.home_channel_id)

        # Send a message to the channel
        # if channel is not None:
        #     #await channel.send(f".art {self.prePrompt} :Honey! I'm home!: {self.postPrompt}")
        #     await channel.send(f"Honey! I'm home!")
        # # Start the process_buffered_messages loop
        # asyncio.create_task(self.process_buffered_messages())

    async def on_disconnect(self):
        await self.session.close()
        logging.info(f'{self.user} has disconnected from Discord.')

    async def run_until_disconnected(self):
        while self.should_continue:
            try:
                await self.start(self.discord_token)
            except Exception as e:
                logging.info(f"Error: {e}")

            if self.should_continue:
                await asyncio.sleep(5)
            else:
                await self.wait_for("close")  # Wait for the close event to complete

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
        randoname = generate_random_string()
        rando_txt = f'_img_/clair-{randoname}.txt'
        rando_jpg = f'_img_/clair-{randoname}.jpg'
        image.save(rando_jpg)
        # Write the prompt to a text file with the same name as the image file
        with open(rando_txt, 'w', encoding='utf-8') as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Guidance Scale: {guidance_scale}\n")
            f.write(f"Steps: {steps}\n")
        return seed, guidance_scale, steps, prompt, rando_jpg

def generate_random_string():
    alphanumeric_chars = string.digits + string.ascii_letters
    return ''.join(random.choice(alphanumeric_chars) for _ in range(8))

## NOTES ##
#
# {message.author.mention} pings the author of the message... (mention == ping)
