import os
import json
import requests
import discord
from discord.ext import commands
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from dotenv import load_dotenv

print("🔨 Starting SourceBot.py")
# ————————————
# 1. Configuration
# ————————————
#load_dotenv()
TOKEN       = "MTQxMDQxMjU0NTA4ODY4NDA4NQ.Gs_sFR.3eu2fjivPahSxR7W_AqUS2VodErcJJgW-JgNsM"
STORE_FILE  = "sources.json"

print(f"🔑 TOKEN loaded? {bool(TOKEN)}")
if TOKEN:
    print(f"🔐 Ready to connect with token prefix: {TOKEN[:8]}…")
else:
    print("❌ No token found. Make sure .env defines DISCORD_TOKEN")
    exit(1)
# ————————————
# 2. Persistent Storage
# ————————————
if os.path.exists(STORE_FILE):
    with open(STORE_FILE, "r") as f:
        sources = json.load(f)
else:
    sources = {}  # url -> category

# Inside your Discord command handler


# ————————————
# 4. Bot & Intents
# ————————————
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)



# ————————————
# 5. On Ready
# ————————————
@bot.event
async def on_ready():
    print(f"✅ Bot is online as {bot.user}")

# ————————————
# 6. Add Source
# ——————————

from sourceClassifier import classify_source  # import from wherever you save it

@bot.command()
async def add_source(ctx, url: str):
    res = classify_source(url, explain=True)
    cat = res["category"]
    conf = res["confidence"]
    await ctx.send(
        f"Added: {url}\nCategory: {cat} ({conf:.2%} confident)\n"
        f"Reason: {res['explanation'][:1800]}..."
    )

# ————————————
# 7. Remove Source (Owner Only)
# ————————————
@bot.command(name="remove_source")

async def remove_source(ctx, url: str):
    if url not in sources:
        return await ctx.send("❌ URL not found.")
    del sources[url]
    save_sources()
    await ctx.send(f"🗑️ Removed <{url}>")

# ————————————
# 8. Edit Source Category (Owner Only)
# ————————————
@bot.command(name="edit_source")

async def edit_source(ctx, url: str, new_category: str):
    new_cat = new_category.capitalize()
    valid = ("Primary", "Secondary", "Tertiary", "Other/Very Biased")
    if url not in sources:
        return await ctx.send("❌ URL not found.")
    if new_cat not in valid:
        return await ctx.send(f"❌ Invalid category. Choose one of {valid}.")
    sources[url] = new_cat
    save_sources()
    await ctx.send(f"✏️ Updated <{url}> → **{new_cat}**")

# ————————————
# 9. List All Sources
# ————————————
@bot.command(name="list_sources")
async def list_sources(ctx):
    if not sources:
        return await ctx.send("No sources recorded yet.")
    lines = [f"<{u}> → **{c}**" for u, c in sources.items()]
    await ctx.send("\n".join(lines))

# ————————————
# 10. Summary & Pie Chart
# ————————————
@bot.command(name="summary")
async def summary(ctx):
    if not sources:
        return await ctx.send("No sources to summarize.")
    
    # Count categories
    counts = { "Primary":0, "Secondary":0, "Tertiary":0, "Other/Very Biased":0 }
    for cat in sources.values():
        counts[cat] = counts.get(cat, 0) + 1

    # Compute total valid sources
    total_valid = counts["Primary"] + counts["Secondary"] + counts["Tertiary"]

    # Generate pie chart
    labels = list(counts.keys())
    sizes  = [counts[lbl] for lbl in labels]
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    chart_path = "source_pie.png"
    fig.savefig(chart_path)
    plt.close(fig)

    # Build summary message
    msg_lines = []
    total = sum(sizes) or 1
    for lbl in labels:
        pct = counts[lbl] / total * 100
        msg_lines.append(f"{lbl}: {counts[lbl]} ({pct:.1f}%)")
    summary_text = "📊 **Source Breakdown**\n" + "\n".join(msg_lines)
    summary_text += f"\n\n🔢 Total Valid Sources: {total_valid}"

    # Send
    await ctx.send(summary_text)
    await ctx.send(file=discord.File(chart_path))
bot.run(TOKEN)