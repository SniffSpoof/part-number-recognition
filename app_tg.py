import asyncio
import os
import signal
import sys
import subprocess
from multiprocessing import Process
import time
import json
import nest_asyncio
import asyncio

from get_links import get_links

from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, ContextTypes
)


N = 1 # Number of processes to start

CHAT_ID = None

process_dict = {}  # Dictionary to store process references


import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for running the Extra")

    parser.add_argument('--model', type=str, required=True, help="The name of the model to use, e.g., 'gemini'")
    parser.add_argument('--api-keys', nargs='+', required=True, help="List of API keys to use")
    parser.add_argument('--gemini-api-model', type=str, default='gemini-1.5-pro', required=False, help="Gemini model you're going to use")
    parser.add_argument('--prompt', type=str, default=None, required=False, help="Path to a text file containing the prompt")
    parser.add_argument('--first-page-link', type=str, default=None, required=False, help="First page link (optional)")
    parser.add_argument('--save-file-name', type=str, default='recognized_data', required=False, help="Name of the file to save recognized data")
    parser.add_argument('--ignore-error', action='store_true', help="Ignore errors and continue processing")
    parser.add_argument('--max-steps', type=int, default=3, required=False, help="Maximum steps to collect links")
    parser.add_argument('--max-links', type=int, default=90, required=False, help="Maximum number of links to collect")
    parser.add_argument('--page-offset', type=int, default=1, required=False, help="Number of threads to use (default is 1)")
    parser.add_argument('--links', nargs='+', default=None, required=False, help="List of pre-generated links to work with")
    parser.add_argument('--telegram-token', type=str, required=True, help="Your Telegram API token")
    parser.add_argument('--chat-id', type=int, required=True, help="Your chat ID with bot. Use get_chat_id.py to define it")
    parser.add_argument('--car-brand', type=str, required=True, help="Car brand to use for prompts. Supported brands: audi, toyota, nissan, suzuki, honda, daihatsu, subaru, mazda, bmw, lexus, volkswagen, volvo, mini, fiat, citroen, renault, ford, isuzu, opel, mitsubishi, mercedes, jaguar, peugeot, porsche, alfa_romeo, chevrolet")

    args = parser.parse_args()

    if args.page_offset < 1:
        raise ValueError("The --page-offset value must be 1 or greater, representing the number of threads.")

    return args


def get_part(arr, k, i):
    part_size = len(arr) // k
    remainder = len(arr) % k

    start_index = i * part_size + min(i, remainder)
    end_index = start_index + part_size + (1 if i < remainder else 0)

    return arr[start_index:end_index]

def show_last_log_lines(offset, lines=10):
  file_name = f"/content/part-number-recognition/process_log{offset}.log"
  try:
      with open(file_name, "r") as file:
          log_lines = file.readlines()[-lines:]
      return "".join(log_lines)
  except FileNotFoundError:
      return f"Log file for process {offset} not found."


# Function to start a script and save the reference to the process
def run_script(args, process_dict, links):
    link_args = get_part(links, N, int(args["page_offset"]))

    command = [
        "python", "main.py",
        "--model", args["model"],
        "--api-keys", *args["api_keys"],
        "--save-file-name", args["save_file_name"],
        "--gemini-api-model", args["gemini_api_model"],
        "--prompt", args["prompt"],
        "--car-brand", args["car_brand"],
        "--page-offset", args["page_offset"],
        "--links", *link_args
    ]

    process = subprocess.Popen(
      command,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL
    )
    process_dict[args["page_offset"]] = {"process": process, "paused": False}


# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Welcome to Process Manager Bot! Use /help to see available commands.")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    commands = (
        "/start - Start the bot\n"
        "/status - Show status of all processes\n"
        "/pause <page_offset> - Pause a process\n"
        "/resume <page_offset> - Resume a process\n"
        "/stop <page_offset> - Stop a process\n"
        "/logs <page_offset> <lines> - Show last N log lines\n"
        "/exit - Stop all processes and shutdown bot"
    )
    await update.message.reply_text(f"Available commands:\n{commands}")


async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    status_message = []
    for offset, proc_info in process_dict.items():
        proc = proc_info["process"]
        if proc.poll() is None:
            status = "paused" if proc_info["paused"] else "running"
        else:
            status = "completed"
        status_message.append(f"Process {offset}: {status}")

    if not status_message:
        status_message.append("No processes are running.")

    await update.message.reply_text("\n".join(status_message))


async def pause(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /pause <page_offset>")
        return

    offset = int(context.args[0])
    proc_info = process_dict.get(offset)
    if proc_info and proc_info["process"].poll() is None and not proc_info["paused"]:
        os.kill(proc_info["process"].pid, signal.SIGSTOP)
        proc_info["paused"] = True
        await update.message.reply_text(f"Process {offset} paused.")
    else:
        await update.message.reply_text(f"Process {offset} not found, already paused, or completed.")


async def resume(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /resume <page_offset>")
        return

    offset = int(context.args[0])
    proc_info = process_dict.get(offset)
    if proc_info and proc_info["process"].poll() is None and proc_info["paused"]:
        os.kill(proc_info["process"].pid, signal.SIGCONT)
        proc_info["paused"] = False
        await update.message.reply_text(f"Process {offset} resumed.")
    else:
        await update.message.reply_text(f"Process {offset} not found, already running, or completed.")


async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 1:
        await update.message.reply_text("Usage: /stop <page_offset>")
        return

    offset = int(context.args[0])
    proc_info = process_dict.get(offset)
    if proc_info and proc_info["process"].poll() is None:
        proc_info["process"].terminate()
        await update.message.reply_text(f"Process {offset} stopped.")
    else:
        await update.message.reply_text(f"Process {offset} not found or already completed.")


async def logs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if len(context.args) != 2:
        await update.message.reply_text("Usage: /logs <page_offset> <lines>")
        return

    offset = int(context.args[0])
    lines = int(context.args[1])
    log_output = show_last_log_lines(offset, lines)
    await update.message.reply_text(log_output)


async def exit_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    for proc_info in process_dict.values():
        if proc_info["process"].poll() is None:
            proc_info["process"].terminate()
    await update.message.reply_text("All processes have been terminated. Shutting down bot.")
    os._exit(0)


async def monitor_processes(context: ContextTypes.DEFAULT_TYPE):
    if all(proc_info["process"].poll() is not None for proc_info in process_dict.values()):
        print(context.job)
        await context.bot.send_message(chat_id=CHAT_ID, text="All processes completed.")
        context.job.schedule_removal() 


if __name__ == "__main__":

    args = parse_args()

    keys = args.api_keys

    N = args.page_offset
    
    CHAT_ID = args.chat_id

    links = args.links or get_links(args.car_brand, args.max_steps, args.max_links, 0)

    script_arguments = [
        {
            "model": args.model,
            "api_keys": [keys[i % len(keys)]], 
            "save_file_name": f"{args.save_file_name}_{i}",
            "gemini_api_model": args.gemini_api_model,
            "prompt": args.prompt,
            "car_brand": args.car_brand,
            "page_offset": str(i)
        }
        for i in range(N)  
    ]

    # Start processes
    for arg in script_arguments:
        run_script(arg, process_dict, links)

    nest_asyncio.apply()
    async def main():
        application = ApplicationBuilder().token(args.telegram_token).post_init(lambda app: app.job_queue).build()

        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("status", status))
        application.add_handler(CommandHandler("pause", pause))
        application.add_handler(CommandHandler("resume", resume))
        application.add_handler(CommandHandler("stop", stop))
        application.add_handler(CommandHandler("logs", logs))
        application.add_handler(CommandHandler("exit", exit_bot))

        #print(args.chat_id)
        job_queue = application.job_queue
        job_queue.run_repeating(monitor_processes, interval=10, first=10)

        await application.initialize()
        print("Bot started. Use it via Telegram.")
        await application.start()
        await application.updater.start_polling(drop_pending_updates=True)  
        await asyncio.Future()  # Keep the program running

    asyncio.get_event_loop().run_until_complete(main())
