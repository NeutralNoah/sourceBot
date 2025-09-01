import os
import json
import requests
import discord
import requests
from discord.ext import commands
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from dotenv import load_dotenv

print("🔨 Starting SourceBot.py V3.6")
# ————————————
# 1. Configuration
# ————————————
from flask import Flask
import threading
import os

app = Flask('')

@app.route('/')
def home():
    return "Bot is running!"

def run():
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)

# Start the web server in a separate thread
threading.Thread(target=run).start()


load_dotenv()
TOKEN = os.getenv("DISCORD_TOKEN")
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
if os.path.exists(STORE_FILE) and os.path.getsize(STORE_FILE) > 0:
    with open(STORE_FILE, "r") as f:
        sources = json.load(f)
else:
    sources = {}  # url -> category
    with open(STORE_FILE, "w") as f:
        json.dump(sources, f, indent=2)


# Inside your Discord command handler
def save_sources():
    with open(STORE_FILE, "w") as f:
        json.dump(sources, f, indent=2)


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
# ———————————

import requests

def is_valid_url(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/115.0 Safari/537.36"
        }
        # stream=True avoids downloading the full body
        response = requests.get(url, headers=headers, timeout=5, stream=True, allow_redirects=True)
        return response.status_code < 400
    except requests.RequestException:
        return False




from sourceClassifier import classify_source  # import from wherever you save it

@bot.command()
async def add_source(ctx, url: str):
    if not is_valid_url(url):
        return await ctx.send("❌ That doesn't appear to be a valid or reachable URL.")

    try:
        res = classify_source(url, explain=True)
        cat = res["category"]
        conf = res["confidence"]
        if conf == 0.25:
            cat="Other"
            conf=0
        if cat == "Other":
            cat="Other/Very Biased/Unrelated"
        sources[url] = cat
        save_sources()
        await ctx.send(
            f"✅ Added: {url}\nCategory: {cat} ({conf:.2%} confident)\n"
        )
    except Exception as e:
        await ctx.send(f"❌ Classification failed: {e}")


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
    valid = ("Primary", "Secondary", "Tertiary", "Other/Very Biased/Unrelated")
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
    print("list sources triggered")
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
    counts = { "Primary":0, "Secondary":0, "Tertiary":0, "Other/Very Biased/Unrelated":0 }
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
    os.remove(chart_path)
bot.run(TOKEN)
