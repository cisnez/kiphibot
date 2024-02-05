from B17_D15C0R6 import D15C0R6 as discord


# Retrieve bot configuration and token
config_files = ["_init__global.yaml", f"_init_{bot_name}.yaml"]
merged_config = self.yaml.merge_yaml_files(config_files)
bot_init_data = merged_config

discord_token = None
aws_secret_access_key = None
aws_access_key_id = None
openai_key = None
telegram_api_id = None
telegram_api_hash = None

# Retrieve keys
try:
    discord_token = self.tokens[f"{bot_name}_discord_token"]
    aws_secret_access_key = self.keys[f"{bot_name}_aws_secret_access_key"]
    aws_access_key_id = bot_init_data.get("aws_access_key_id")
    openai_key = self.keys["openai_api_key"]
    telegram_api_id = bot_init_data.get("telegram_api_id")
    telegram_api_hash = self.keys[f"{bot_name}_telegram_api_hash"]

    self.data.set_flash('debug', f"Keys Dictionary: {self.keys}")
    self.data.set_flash('debug', f"aws_secret_access_key for {bot_name}: {aws_secret_access_key}")
    self.data.set_flash('debug', f"aws_access_key_id for {bot_name}: {aws_access_key_id}")
except AttributeError as e:
    error_message = f"An attribute is missing for {bot_name}. Please check the bot's configuration, and the keys and tokens yaml files. {str(e)}"
    self.data.set_flash('error', error_message)
    return error_message
except KeyError as e:
    error_message = f"A key property is missing for {bot_name}. Please check the bot's configuration, and the keys and tokens yaml files. {str(e)}"
    self.data.set_flash('error', error_message)
    return error_message

if discord_token is None:
    warning_message = f"The discord_token is set to None for {bot_name}. Please check the ___tokens__.yaml file."
    self.data.set_flash('warning', warning_message)
else:
    self.data.set_flash('debug', f"Retrieved discord_token for {bot_name}.")

if aws_secret_access_key is None or aws_access_key_id is None:
    warning_message = f"The aws_secret_access_key or aws_access_key_id is set to None for {bot_name}. Please check the ___keys__.yaml file."
    self.data.set_flash('warning', warning_message)
else:
    self.data.set_flash('debug', f"Retrieved AWS keys for {bot_name}.")

if openai_key is None:
    warning_message = f"The openai_api_key is set to None for {bot_name}. Please check the ___keys__.yaml file."
    self.data.set_flash('warning', warning_message)
else:
    self.data.set_flash('debug', f"Retrieved openai_api_key for {bot_name}.")

if telegram_api_id is None or telegram_api_hash is None:
    warning_message = f"The Telegram api_id or api_hash is set to None for {bot_name}. Please check the _init_{bot_name}.yaml and the ___keys__.yaml files."
    self.data.set_flash('warning', warning_message)
else:
    self.data.set_flash('debug', f"Retrieved Telegram api_id and api_hash for  {bot_name}.")

# Create and store the bot instance
try:
    self.data.set_flash('debug', f"{type(bot_name)}: bot_name")
    self.data.set_flash('debug', f"{type(self.bot_tasks)}: self.bot_tasks")
    self.data.set_flash('debug', f"{type(self.secrets)}: self.secrets")
    self.data.set_flash('debug', f"{type(self.data)}: self.data")
    self.data.set_flash('debug', f"{type(bot_init_data)}: bot_init_data")
    self.data.set_flash('debug', f"{type(openai_key)}: openai_key")
    self.data.set_flash('debug', f"{type(discord_token)}: iscord_token")
    # self.data.set_flash('debug', f"{type(telegram_api_id)}: telegram_api_id")
    # self.data.set_flash('debug', f"{type(telegram_api_hash)}: telegram_api_hash")
    # self.data.set_flash('debug', f"{type(aws_secret_access_key)}: aws_secret_access_key")
    # self.data.set_flash('debug', f"{type(aws_access_key_id)}: aws_access_key_id")

    # Instantiate the bot
    bot = B07(asyncio.get_event_loop(), self.secrets, self.data, bot_name, bot_init_data, openai_key, discord_token, telegram_api_id, telegram_api_hash, aws_secret_access_key, aws_access_key_id)

    # Create the task from the coroutine
    task = asyncio.create_task(bot.start_bit_manager(), name=bot_name)

    # Add the bot task to the bot_tasks dict
    self.bot_tasks[bot_name] = task
    task.add_done_callback(self.handle_bot_task_done)

    # Add the bot to the dictionary after it starts successfully
    self.bots[bot_name] = bot

    # Check if the bot_name exists in the dictionary
    if bot_name not in self.bots:
        raise BotDoesNotExistInDictionaryError(f"Bot {bot_name} does not exist in dictionary!")
