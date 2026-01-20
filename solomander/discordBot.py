
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import threading
from typing import final
import discord
from discord.ext import commands
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import io
import asyncio
from tabulate import tabulate 
from openpyxl import Workbook

try: 
    from .logger import log, stamp, pront
    
    from .baseStrategy import Strategy
    from .utils import random_color
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront
    from baseStrategy import Strategy
    from utils import random_color
  

class DiscordBot:
    def __init__(self, mt5_live = None, main_channel_id = 1426314357788246036):
        
        
        self.main_channel_id = main_channel_id  # default main channel
        self.live_bot = mt5_live
   
        self.am_ready = threading.Event()
        self.am_ready.clear()
        self._live_bot_attached = threading.Event()
        self._live_bot_attached.clear()

        # link to live bot if provided
        if self.live_bot:
            self.live_bot._discord_attached.set()  
            self.live_bot.discord_bot = self
            self._live_bot_attached.set()
        
        
        load_dotenv()
        self.token = str(os.getenv('DISCORD_BOT_TOKEN'))
        self.intents = discord.Intents.default()
        self.intents.message_content = True  # required to read messages
        
        # sets command prefix to "!"
        self.bot = commands.Bot(command_prefix="!", intents=self.intents)

        # Register events and commands
        self._register_events()
        self._register_commands()

        # Discord bot in its own thread

        self.___discord_listener = threading.Thread(target=self._run_bot, daemon=True)
        self.___discord_listener.start()
        

    def _run_bot(self):
        self.bot.run(self.token)

    def _register_events(self):
        @self.bot.event
        async def on_ready():

            self.am_ready.set()
            stamp.success(f"[Discord] ✅ Logged in as {self.bot.user}")

    def _register_commands(self):
        
        @self.bot.command(name="say")
        async def say(ctx, *, message: str):
            """Repeats whatever you type after the command"""
            stamp.show(f"[Discord] {ctx.author}: {message}")
            await ctx.send(message)

        @self.bot.command(name="stop")
        
        async def stop(ctx):
            """Stops both Discord bot and MT5 event loop"""

            if self._live_bot_attached.is_set():
                await ctx.send("🛑 Shutdown command received. Stopping systems...")
                # Stop MT5 and Discord loop if event is linked
                
                if self.live_bot._running is not None:
                    self.live_bot._running.clear()
                    stamp.success("[Discord] 🧠 MT5 bot and Discord bot stopped.")
                    stamp.success("[Discord] 💤 Discord bot closing.")
                else:
                    log.warning("[Discord] issue closing MT5 bot, event not linked.")
            else:
                await ctx.send("😬 Live runner not attached sorry...")
                stamp.warning("[Discord] 😬 Live runner not attached sorry...")

        # === visualisation ===

        @self.bot.command(name="stats")
        async def stats(ctx):
            """Shows currents stats"""

            if self._live_bot_attached.is_set():
                
                bot = self.live_bot.s.s

                embed,files = self._embed(bot.discord_stats())
                
                embed.set_footer(text=f"For {ctx.author.display_name}", icon_url=ctx.author.display_avatar.url)

                await ctx.send(embed=embed, files=files)
                stamp.show(f"[Discord] {ctx.author} requested stats in {ctx.channel}.")
            else:
                await ctx.send("😬 eh Live runner not attached sorry...")
                stamp.warning("[Discord] 😬 Live runner not attached sorry...")

        @self.bot.command(name="market")
        async def market(ctx):

            """Shows market info"""

            if self._live_bot_attached.is_set():

                embed, files = self._embed(self.live_bot.s.s.discord_market())
                embed.set_footer(text=f"For {ctx.author.display_name}", icon_url=ctx.author.display_avatar.url)

                await ctx.send(embed=embed, files=files)
                stamp.show(f"[Discord] {ctx.author} requested market data in {ctx.channel}.")
            else:
                await ctx.send("😬 eh Live runner not attached sorry...")
                stamp.warning("[Discord] 😬 Live runner not attached sorry...")

        @self.bot.command(name="settings")
        async def settings(ctx):

            '''Shows input parameters'''

            if self._live_bot_attached.is_set():
                
                embed, files = self._embed(self.live_bot.s.discord_settings())
                embed.set_footer(text=f"For {ctx.author.display_name}",icon_url=ctx.author.display_avatar.url)

                await ctx.send(embed=embed, files=files)
                stamp.show(f"[Discord] {ctx.author} requested market data in {ctx.channel}.")
            else:
                await ctx.send("😬 eh Live runner not attached sorry...")
                stamp.warning("[Discord] 😬 Live runner not attached sorry...")
            
        @self.bot.command(name="inputs")
        async def inputs(ctx):

            '''Shows input parameters'''

            if self._live_bot_attached.is_set():
                
                embed,files = self._embed(self.live_bot.s.discord_inputs())
                embed.set_footer(text=f"For {ctx.author.display_name}",icon_url=ctx.author.display_avatar.url)

                await ctx.send(embed=embed, files=files)
                stamp.show(f"[Discord] {ctx.author} requested market data in {ctx.channel}.")
            else:
                await ctx.send("😬 eh Live runner not attached sorry...")
                stamp.warning("[Discord] 😬 Live runner not attached sorry...")
        
        @self.bot.command(name="livesettings")
        async def livesettings(ctx):
        
            if self._live_bot_attached.is_set():
                    
                column2 = (
                f"_bar_needs_closed: `{self.live_bot._bar_needs_closed}`\n"
                f"_last_candle_time: `{self.live_bot._last_candle_time}`\n"
                f"_discord_attached: `{self.live_bot._discord_attached}`\n"
                f"_running: `{self.live_bot._running}`\n"
                f"_password_verified: `{self.live_bot._password_verified}`\n"
                )

                title = "🎮 Live Settings"
                color= "#267b1c"
                embed = [{'value': column2, 'type': "text", 'inline': True, "title": ""}]


                embed,files = self._embed([embed, color, title])
                embed.set_footer(text=f"For {ctx.author.display_name}",icon_url=ctx.author.display_avatar.url)

                await ctx.send(embed=embed, files=files)
                stamp.show(f"[Discord] {ctx.author} requested live settings in {ctx.channel}.")
            else:
                await ctx.send("😬 eh Live runner not attached sorry...")
                stamp.warning("[Discord] 😬 Live runner not attached sorry...")
                


    def post_fig(self, buf: io.BytesIO, message: str, channel_id = None, color="#237ce0"):

        if channel_id is None:
            channel_id = self.main_channel_id

        stamp.show("[Discord] 📈 Sending figure to Discord...")
        if not self.bot.is_ready():
            stamp.warning("[Discord] Bot not ready, cannot send yet.")
            return

        channel = self.bot.get_channel(channel_id)
        color_value = int(color.replace("#", ""), 16)
        file = discord.File(buf, filename="chart.png")
        
        embed = discord.Embed(
            description=message,
            color=color_value
        )

        embed.set_image(url="attachment://chart.png")

        future = asyncio.run_coroutine_threadsafe(
        channel.send(embed=embed, file=file),
        self.bot.loop
        )

        try:
            future.result(timeout=5)
            stamp.success("[Discord] 📈 Figure pushed to Discord.")
        except Exception as e:
            stamp.error(f"[Discord] ❌ Failed to send figure: {e}")

    
        return
    
    def post_message(self, message: str, channel_id = None):
        
        if channel_id is None:
            channel_id = self.main_channel_id

        channel = self.bot.get_channel(channel_id)
        if channel is None:
            stamp.error("[Discord] ❌ Channel not found.")
            return

        future = asyncio.run_coroutine_threadsafe(
        channel.send(content=message),
        self.bot.loop
        )
        try:
            future.result()  # wait for send to complete (optional)
            stamp.success("[Discord] ✅ Message sent successfully!")
        except Exception as e:
            stamp.error(f"[Discord] ❌ Failed to send message: {e}")

    def post_embed(self, to_embed, channel_id = None ):
        """ Line for multiple embed files, and single image """
        
        if channel_id is None:
            channel_id = self.main_channel_id

        channel = self.bot.get_channel(channel_id)

        embed,files = self._embed(to_embed)

        future = asyncio.run_coroutine_threadsafe(
        channel.send(embed=embed,files=files),
        self.bot.loop
        )
        try:
            future.result()
            stamp.success("[Discord] ✅ Embed sent successfully!")
        except Exception as e:
            stamp.error(f"[Discord] ❌ Failed to send embed: {e}")

        return
       
    def _embed(self, to_embed):
        
        embed_data = to_embed[0]
        color = int(to_embed[1].replace("#", ""), 16)
        title = to_embed[2]
        embed = discord.Embed(title=title,color=color)
        files = []

        for i, section in enumerate(embed_data):
            typ = section.get("type")
            val = section.get("value")
            inline = section.get("inline", True)
            name = section.get("title", "")

            if typ == "text":
                embed.add_field(name=name or " ", value=val, inline=inline)

            elif typ == "file" and val:
                
                file = discord.File(val, filename=f"figure_{i}.png")
                files.append(file)
                embed.set_image(url=f"attachment://figure_{i}.png")  
    

        return embed, files

    def post_data(self, df: pd.DataFrame, channel_id=None):

        if channel_id is None:
            channel_id = self.main_channel_id
        channel = self.bot.get_channel(channel_id)
        
        data = df
        
        # Convert to CSV text
        csv_buffer = io.StringIO()
        data.to_csv(csv_buffer, index=True)
        csv_buffer.seek(0)
        
        # Wrap in a Discord file object
        file = discord.File(fp=io.StringIO(csv_buffer.getvalue()), filename="data.csv")

        future = asyncio.run_coroutine_threadsafe(channel.send(file=file),self.bot.loop)
        try:
            future.result()
            stamp.success("[Discord] ✅ Embed sent successfully!")
        except Exception as e:
            stamp.error(f"[Discord] ❌ Failed to send embed: {e}")

        return

    def post_data_table(self, df: pd.DataFrame, channel_id=None):

        if channel_id is None:
            channel_id = self.main_channel_id
        channel = self.bot.get_channel(channel_id)
        
        data = df
        table = tabulate(data, headers='keys', tablefmt='github', showindex=False)
        message = f"```{table}```"

        future = asyncio.run_coroutine_threadsafe(channel.send(message),self.bot.loop)
        try:
            future.result()
            stamp.success("[Discord] ✅ Embed sent successfully!")
        except Exception as e:
            stamp.error(f"[Discord] ❌ Failed to send embed: {e}")

        return

if __name__ == "__main__":
    

    pass

