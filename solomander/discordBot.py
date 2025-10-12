
import pandas as pd
import os
from dotenv import load_dotenv
import threading
from typing import final
import discord
from discord.ext import commands
from datetime import datetime, timezone
import io

import matplotlib
matplotlib.use("Agg")  # non-GUI backend (for servers / threads)
import matplotlib.pyplot as plt
import mplcyberpunk as cyberpunk

try: 
    from .logger import log, stamp, pront
    from .mt5 import MT5_live
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront
    from mt5 import MT5_live
  

def basic_discord_graph(x,y,color="#2ecc71",xlabel="X-axis",ylabel="Y-axis"):

    # generate your plot
    plt.style.use("cyberpunk")
    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.plot(x, y, color=color, marker='o')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cyberpunk.add_glow_effects()

    # remove background
    fig.patch.set_alpha(0.0)       # transparent figure background
    ax.set_facecolor("none")       # transparent plotting area
    ax.grid(True, alpha=0.2, color="#ffffff")
    ax.grid(False, axis="x")


    # save to a BytesIO buffer instead of disk
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)

    # send it to Discord
    return discord.File(buf, filename="chart.png")



class DiscordBot:
    def __init__(self, mt5_live: MT5_live):
        
        self.live_bot = mt5_live
        self.live_bot._discord_attached.set()  #tell mt5 bot that discord is attached
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
            self.live_bot._discord_ready.set()
            stamp.success(f"[Discord] ✅ Logged in as {self.bot.user}")

    def _register_commands(self):
        
        @self.bot.command(name="say")
        async def say(ctx, *, message: str):
            """Repeats whatever you type after the command"""
            stamp.show(f"[Discord] {ctx.author}: {message}")
            await ctx.send(message)

        @self.bot.command(name="shutdown")
        async def shutdown(ctx):
            """Stops both Discord bot and MT5 event loop"""
            await ctx.send("🛑 Shutdown command received. Stopping systems...")

            # Stop MT5 and Discord loop if event is linked
            if self.live_bot._running is not None:
                self.live_bot._running.clear()
                stamp.success("[Discord] 🧠 MT5 bot and Discord bot stopped.")
                stamp.success("[Discord] 💤 Discord bot closing.")
            else:
                log.warning("[Discord] issue closing MT5 bot, event not linked.")


        @self.bot.command(name="stats")
        async def stats(ctx):
            """Shows currents stats"""

            embed = discord.Embed(title="Current Stats", color=0x0ffff, timestamp=datetime.now(timezone.utc))

            embed.add_field(name="", value=f"💰 Balance: `{self.live_bot.ACTIVE_MARGIN}`", inline=False)
            embed.add_field(name="", value=f"📈 Open Trades: `{self.live_bot.OPEN_TRADES}`", inline=False)
            embed.add_field(name="", value=f"📊 Total Trades: `{self.live_bot.TOTAL_TRADES}`", inline=False)
            embed.add_field(name="", value=f"🏆 Win Rate: `{self.live_bot.WIN_RATE}%`", inline=False)
            
            file = basic_discord_graph([0,1,2,3,4,5], [0,1,2,3,4,5], xlabel="Trades", ylabel="Margin", color="#00ffff")
            embed.set_image(url="attachment://chart.png")
            
            embed.set_footer(text=f"For {ctx.author.display_name}", icon_url=ctx.author.display_avatar.url)

            stamp.show(f"[Discord] {ctx.author} requested stats in {ctx.channel}.")
            await ctx.send(embed=embed, file=file)
            

 
       


if __name__ == "__main__":
    

    pass

