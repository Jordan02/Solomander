
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

import matplotlib
matplotlib.use("Agg")  # non-GUI backend (for servers / threads)
import matplotlib.pyplot as plt
import mplcyberpunk as cyberpunk

try: 
    from .logger import log, stamp, pront
    from .mt5 import MT5_live
    from .baseStrategy import Strategy
except ImportError: 
    #for running as main script
    from logger import log, stamp, pront
    from mt5 import MT5_live
    from baseStrategy import Strategy
  

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
    def __init__(self, mt5_live: MT5_live = None):
        
        self.live_bot = mt5_live
   
        self.am_ready = threading.Event()
        self.am_ready.clear()
        self._live_bot_attached = threading.Event()
        self._live_bot_attached.clear()

        # link to live bot if provided
        if self.live_bot:
            self.live_bot._discord_attached.set()  
            self._live_bot_attached.set()

            self.live_bot.discord_bot = self
            self.live_bot._discord_attached.set()
        
        
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

        @self.bot.command(name="shutdown")
        
        async def shutdown(ctx):
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


        @self.bot.command(name="stats")
        async def stats(ctx):
            """Shows currents stats"""

            if self._live_bot_attached.is_set():
                bot = self.live_bot.wrapper.s
                color = "#E5FF00"
                discord_color= int(color.replace("#", ""), 16)
                start_time = self.live_bot.TIME_START
                current_time = datetime.now(ZoneInfo("Europe/London"))
                time_elapsed = current_time - start_time

                start_time_str = start_time.strftime("%d/%m/%y %H:%M %Z")
                time_elapsed_str = str(time_elapsed).split('.')[0]  # remove microseconds for cleaner display
                
                embed = discord.Embed(
                    title="📊 Current Stats Dashboard",
                    color=discord_color,
                    timestamp=datetime.now(timezone.utc)
                )

                column1 = (
                    
                    f"Total PnL: `{bot.TOTAL_PNL:.2f} {bot.TICK_CURRENCY}`\n"
                    f"Margin: `{bot.TOTAL_MARGIN:.2f} {bot.TICK_CURRENCY}`\n"
                    f"Total Return: `{bot.TOTAL_RETURN*100:.2f}%`\n"
                    f"Profit Factor: `{bot.TOTAL_PROFIT_FACTOR:.2f}`\n"
                    f"Max Drawdown: `{bot.TOTAL_MAX_DRAWDOWN:.2f} {bot.TICK_CURRENCY}`\n"
                    f"Payoff Ratio: `{bot.TOTAL_PAYOFF_RATIO:.2f}`\n"
                    f"Sharpe (Annual): `{bot.TOTAL_SHARPE_RATIO_ANNUAL:.2f}`\n"
                    f"Sharpe (Daily): `{bot.TOTAL_SHARPE_RATIO_DAILY:.2f}`\n"
                    f"Sortino (Annual): `{bot.TOTAL_SORTINO_RATIO_ANNUAL:.2f}`\n"
                    f"Sortino (Daily): `{bot.TOTAL_SORTINO_RATIO_DAILY:.2f}`\n"
                    f"PnL/MDD Ratio: `{bot.TOTAL_PNL_MDD_RATIO:.2f}`\n\n"
                )
        
                column2 = (
                    f"Start time `{start_time_str}`\n"
                    f"Elapsed time `{time_elapsed_str}`\n"
                    f"Total Trades: `{bot.TOTAL_TRADES}`\n"
                    f"Open Trades: `{bot.OPEN_TRADES}`\n"
                    f"Total Longs: `{bot.TOTAL_LONGS}`\n"
                    f"Total Shorts: `{bot.TOTAL_SHORTS}`\n"
                    f"Win Rate: `{bot.TOTAL_WIN_RATE*100:.2f}%`\n"
                    f"Win Rate Long: `{bot.TOTAL_WIN_RATE_LONG*100:.2f}%`\n"
                    f"Win Rate Short: `{bot.TOTAL_WIN_RATE_SHORT*100:.2f}%`\n"
                    f"Average Profit: `{bot.AVERAGE_PROFIT:.2f} {bot.TICK_CURRENCY}`\n"
                    f"Average Loss: `{bot.AVERAGE_LOSS:.2f} {bot.TICK_CURRENCY}`\n"
                    f"Average Return: `{bot.AVERAGE_RETURN*100:.2f}%`\n"
     
                )

                # ---- ADD COLUMNS ----
                embed.add_field(name="", value=column1, inline=True)
                embed.add_field(name="", value=column2, inline=True)
            

                # ---- ADD PNL CHART ----
                file = basic_discord_graph(
                    np.arange(len(bot.l_CUMSUM_PNL)),
                    bot.l_CUMSUM_PNL,
                    xlabel="Trades",
                    ylabel="Pnl",
                    color=color
                )
                embed.set_image(url="attachment://chart.png")

                # ---- FOOTER ----
                embed.set_footer(
                    text=f"For {ctx.author.display_name}",
                    icon_url=ctx.author.display_avatar.url
                )

                stamp.show(f"[Discord] {ctx.author} requested stats in {ctx.channel}.")
                await ctx.send(embed=embed, file=file)
            else:
                await ctx.send("😬 eh Live runner not attached sorry...")
                stamp.warning("[Discord] 😬 Live runner not attached sorry...")


    def post_fig(self, buf: io.BytesIO, message: str, channel_id = 1426314357788246036, color="#237ce0"):

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
 
       


if __name__ == "__main__":
    

    pass

