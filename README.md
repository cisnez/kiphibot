# DiscordStableDiffusion

## Description
A Discord bot using Discord `API` and Stable Diffusion `.safetensor` models; generates images from its own Discord posts that start with `.art:> `.

## Files
main.py
, B07_C0R3.py
, _init_botname.yaml
, _init__global.yaml

## Requirements to run Stable Diffusion using Python and Nvidia CUDA.
1. CUDA drivers
https://developer.nvidia.com/cuda-downloads
2. PyTorch with CUDA
https://pytorch.org/get-started/locally/
Both websites allow you to pick your OS and provide easy install instructions.
Note: with PyTorch choose CUDA 12.1. No worries that the latest CUDA is 12.3 (as of 09 feb 2024).

## Usage
0. Install all required python libraries (discord)
1. Clone the repository to your local machine.
2. Configure your bot by modifying the example `_init_botname.yaml` template.
3. Create a system environment variable for your bot's DISCORD_TOKEN
4. Run the bot using `python main.py botname`.
