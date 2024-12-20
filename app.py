import os
import signal
import subprocess
from multiprocessing import Process
import time

from get_links import get_links

import json

N = 1 # Number of processes to start

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
    parser.add_argument('--car-brand', type=str, required=True, help="Car brand to use for prompts. Supported brands: audi, toyota, nissan, suzuki, honda, daihatsu, subaru, mazda, bmw, lexus, volkswagen, volvo, mini, fiat, citroen, renault, ford, isuzu, opel, mitsubishi, mercedes, jaguar, peugeot, porsche, alfa_romeo, chevrolet")

    args = parser.parse_args()

    if args.page_offset < 1:
        raise ValueError("The --page-offset value must be 1 or greater, representing the number of threads.")
    
    return args

def get_part(arr, k, i):
    part_size = len(arr) // k

    start_index = i * part_size
    end_index = (i + 1) * part_size

    return arr[start_index:end_index]

def show_last_log_lines(page_offset, n=10):
    log_filename = f"process_log{page_offset}.log"

    if not os.path.exists(log_filename):
        print(f"Log file: {log_filename} - Not found.")
        return

    try:
        with open(log_filename, 'r') as log_file:
            lines = log_file.readlines()

            print(f"Last {n} lines from {log_filename}:")
            for line in lines[-n:]:
                print(line, end='')
    except Exception as e:
        print(f"Failed while reading {log_filename}: {e}")

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

    print(f"Starting command: {' '.join(command)}")
    process = subprocess.Popen(
    command,
      stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL
    )

    process_dict[int(args["page_offset"])] = {"process": process, "paused": False}
    print(f"Process {args['page_offset']} started with PID: {process.pid}")


def main():

    args = parse_args()

    keys = args.api_keys

    global N
    N = args.page_offset

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

    process_dict = {}  # Dictionary to store process references

    # Start processes
    for args in script_arguments:
        run_script(args, process_dict, links)

    # Simple command-line interface to manage processes
    try:
        while True:
            print("\nManagement commands:")
            print("1. pause <page_offset> - Pause a process")
            print("2. resume <page_offset> - Resume a process")
            print("3. stop <page_offset> - Stop a process")
            print("4. status - Display the status of all processes")
            print("5. logs <page_offset> <lines> - Show last log lines from process")
            print("6. exit - Stop all processes and exit")

            command = input("\nEnter a command: ").strip().split()

            if not command:
                continue

            cmd = command[0]
            if cmd == "pause" and len(command) > 1:
                offset = int(command[1])
                proc_info = process_dict.get(offset)
                if proc_info and proc_info["process"].poll() is None and not proc_info["paused"]:
                    os.kill(proc_info["process"].pid, signal.SIGSTOP)
                    proc_info["paused"] = True
                    print(f"Process {offset} paused.")
                else:
                    print(f"Process {offset} not found, already paused, or completed.")

            elif cmd == "resume" and len(command) > 1:
                offset = int(command[1])
                proc_info = process_dict.get(offset)
                if proc_info and proc_info["process"].poll() is None and proc_info["paused"]:
                    os.kill(proc_info["process"].pid, signal.SIGCONT)
                    proc_info["paused"] = False
                    print(f"Process {offset} resumed.")
                else:
                    print(f"Process {offset} not found, already running, or completed.")

            elif cmd == "stop" and len(command) > 1:
                offset = int(command[1])
                proc_info = process_dict.get(offset)
                if proc_info and proc_info["process"].poll() is None:
                    proc_info["process"].terminate()
                    print(f"Process {offset} stopped.")
                else:
                    print(f"Process {offset} not found or already completed.")

            elif cmd == "status":
                for offset, proc_info in process_dict.items():
                    proc = proc_info["process"]
                    if proc.poll() is None:
                        status = "paused" if proc_info["paused"] else "running"
                    else:
                        status = "completed"
                    print(f"Process {offset}: {status}")

                if all(proc_info["process"].poll() is not None for proc_info in process_dict.values()):
                    print("All processes have completed.")

            elif cmd == "logs" and len(command) > 2:
                show_last_log_lines(int(command[1]), int(command[2]))

            elif cmd == "exit":
                print("Stopping all processes...")
                for proc_info in process_dict.values():
                    if proc_info["process"].poll() is None:  # If the process is still running
                        proc_info["process"].terminate()
                break

            else:
                print("Unknown command. Please try again.")

    except KeyboardInterrupt:
        print("\nForce stopping all processes...")
        for proc_info in process_dict.values():
            if proc_info["process"].poll() is None:
                proc_info["process"].terminate()

    finally:
        print("All processes have been terminated.")

main()
