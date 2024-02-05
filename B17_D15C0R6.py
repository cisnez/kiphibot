#B17_D15C0RD.py
import discord
from discord.ext import commands
import logging

class D15C0R6(commands.Bot):
    def __init__(self, data, bot_init_data):
        self.bot_init_data = bot_init_data
        self.flash_data = bot.flash_data
        self.discord_token = bot.discord_token

        intents = discord.Intents(**bot.bot_init_data["intents"])
        command_prefix = bot.bot_init_data["command_prefix"]

        super().__init__(command_prefix=command_prefix, intents=intents)

    async def start(self):
        await super().start(self.discord_token)
        await self.message_queue.put(self.Message(ZIP, "game"))
        self.data.flash_set(ZIP)  # Use the set method

    async def close(self):
        await super().close()

    async def on_ready(self):
        home_channel = self.get_channel(self.bot.home_channel_id)
        if home_channel is not None:
            await home_channel.send("Honey! I'm home!")
        else:
            self.data.flash_set(f"Error: Could not find a channel with ID {self.bot.home_channel_id}")  # Use the set method
            print(f"Error: Could not find a channel with ID {self.bot.home_channel_id}")
            quit()

        # If you define an on_message event, the bot will not process commands automatically unless you explicitly call `await self.process_commands(message)`. This is because the `on_message`` event is processed before the command, so if you don't call `process_commands`, the command processing stops at `on_message`. Any return statement will break process_commands.
        async def on_message(self, message):
            logging.debug(f'\n-- START ON_MESSAGE --\n')
            # The bot will show as typing while executing the code inside this block
            # So place your logic that takes time inside this block
            async with message.channel.typing():
                # Remove bot's mention from the message
                clean_message = discord.utils.remove_markdown(str(discord.user.mention))
                message_content = message.content.replace(clean_message, "").strip()

                # Target the channel that the message came from.
                source_channel = message.channel
                message.author.id = message.author.id
                message.author.name = message.author.name
                if self.image_generation_in_progress:
                    logging.info(":image_generation_in_progress:")
                    if message_content == "!flag":
                        logging.info('\n!!! False Flag !!!\n')
                        self.image_generation_in_progress = False
                    self.buffered_messages.append(message)

                elif message.author.id in self.ignore_author_ids:
                    logging.debug(f'Ignored Author: {message.author.name} [ID:{message.author.id}]')
                    

                elif message.channel.id in self.ignore_channel_ids:
                    logging.info(f'Ignored Channel ID: {message.channel.name}\n')
                    

                elif message_content.startswith('.art') and message.channel.id != 947907332803805204:
                    self.image_generation_in_progress = True
                    prompt = message_content[5:].strip()
                    prompt = self.unicode_to_shortcode(prompt)
                    full_prompt = f'{self.prePrompt} {prompt} {self.postPrompt}'
                    # full_prompt = f'{prompt}'
                    logging.info(f'.art prompt:\n{prompt}\n')
                    logging.info(f'.art (full prompt):\n{full_prompt}\n')
                    seed, guidance_scale, steps, full_prompt, image_path = self.generate_image(full_prompt)
                    logging.info('generate_image returned\n')
                    # Create the output file name by appending "stamp-" to the base file name
                    output_file = postfix_filename(image_path, "-coin")  # Generate new filename with '-framed' postfix
                    try:
                        # content = f'ClairBelle: {prompt}\nseed: {seed} | guidance scale: {guidance_scale} | steps: {steps}'
                        content = ""
                        #content = f'{self.username}: {prompt}'
                        # Modify image_path and save it back
                        #self.scale_image(image_path, ".scaled_art_image.png", 2048, 1152) # 16:9
                        self.scale_image(image_path, ".scaled_art_image.png", 1440, 1800) # 4:5
                        #self.add_frame(".scaled_art_image.png", '.frame_image_heads_16x9.png', output_file)
                        self.add_frame(".scaled_art_image.png", '.frame_image_heads_4x5.png', output_file)

                        # Save the framed image back to output_file
                        #self.add_text_to_image(image_path, content, output_file)  # Add text to the image, save to output_file
                        # Create the discord.File object using output_file
                        image_file = discord.File(output_file)
                        logging.info(f"File prepared for sending: {output_file}")

                    except Exception as e:
                        self.image_generation_in_progress = False
                        logging.error(f"Error occurred while preparing the image: {e}")
                        # You might want to re-raise the exception after logging it, 
                        # especially if the subsequent code relies on the success of the previous code
                        raise
                    # Create the message content
                    # Send the image file to the channel
                    # content = f'``seed: {seed}\nscale: {guidance_scale} | steps: {steps}``\n``{prompt}``'
                    await message.channel.send(file=image_file)
                    self.image_generation_in_progress = False
                    # await message.channel.send(file=image_file, content=content)
                    

                elif message_content.startswith('.delete') and (message.author.id == 465419968276594688 or 875422085319622666 or 971580205586067526):
                    if message.reference:  # Check if the message is a reply
                        try:
                            referenced_message = await message.channel.fetch_message(message.reference.message_id)
                            await referenced_message.delete()
                        except Exception as e:
                            await message.channel.send(f"Error deleting message: {e}")
                            logging.error(f"Error deleting message: {e}")
                    await message.delete()  # Delete the command message
                    
                elif message_content.startswith('.hello'):
                    logging.info('.hello')
                    await message.channel.send("Hello Channel!")
                    
                elif message_content.startswith('.schedulers'):
                    classes = self.sd_pipe_txt2img.scheduler.compatibles
                    class_names = ""
                    for cls in classes:
                        cls_str = str(cls)
                        cleaned_name = cls_str.replace("<class 'diffusers.schedulers.", "").replace("'>", "")
                        cleaned_name = re.sub(r'scheduling_.*?\.', '', cleaned_name)
                        class_names += f"{cleaned_name}\n"
                    await message.channel.send(f'**compatible schedulers**\n```{class_names}```\n{self.sd_pipe_txt2img.scheduler}')

                elif any(message_content.startswith(prefix) for prefix in self.ignored_prefixes):
                    for prefix in self.ignored_prefixes:
                        if message_content.startswith(prefix):
                            logging.info(f'Ignoring message due to prefix: {prefix}\n')

                else:
                    if (message.author.id != self.self_author_id and message.author.id in self.allow_author_ids):
                        ack_who = f'-------\nSaw VIP message from {message.author.name}.'
                        console_ack = f'`message.channel.id`: `{message.channel.id}`, `message.channel.name`: `{message.channel.name}`,\n'
                        console_ack += f'`message.id`: `{message.id}`,\n'
                        console_ack += f'`message.author.id`: `{message.author.id}``message.author`: `{message.author}`\n-------'
                    else:
                        ack_who = '-------\nSaw message from self.'
                        console_ack = f'`message.channel.id`: `{message.channel.id}`, `message.channel.name`: `{message.channel.name}`,\n'
                        console_ack += f'`message.id`: `{message.id}`,\n'
                        console_ack += f'`message.author.id`: `{message.author.id}``message.author`: `{message.author}`\n-------'

                # Always process commands at the end of the on_message event
                await self.process_commands(message)
                logging.debug(f'\n-- END ON_MESSAGE --\n')
