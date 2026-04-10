"""
vps_worker.py — Self-hosted VPS inference worker with multiprocessing queue (free-tier)

Runs on the California VPS at https://california.notmyvoice.ai
Exposes a FastAPI endpoint at POST /interface that:
  1. Accepts audio file + job parameters
  2. Enqueues the job into a multiprocessing.Queue
  3. Returns immediately with a "Queued" response
  4. A persistent worker process drains the queue serially, broadcasting queue
     position updates to all waiting jobs via their updateLink callback URLs

This is the cost-saving tier. Free users share this single GPU/CPU server.
If this server is unreachable, site/app.py falls back to Modal (see routing_and_callbacks.py).
"""

import gc
import uuid
import modal
import shutil
import boto3
from pydub import AudioSegment
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
import uvicorn, os
import requests

os.system("export LC_CTYPE=en_US.UTF-8")
os.environ["PYTHONIOENCODING"] = "utf-8"

app = FastAPI()

image = (
    modal.Image.debian_slim()
    .pip_install("setuptools<58", "uvicorn", "tqdm", "requests")
    .run_commands(["apt-get update", "apt-get install zip -y"])
)
volume = modal.SharedVolume().persist("packages")
volume2 = modal.SharedVolume().persist("models")
cacheVolume = modal.SharedVolume().persist('.cache')
stub = modal.Stub(name="rvc_testing")


# ---------------------------------------------------------------------------
# RVC — local inference class (runs on the VPS directly, no Modal GPU billing)
# ---------------------------------------------------------------------------

class RVC():
    def __init__(self, model, updateLink=None):
        import json
        self.updateLink = updateLink
        def update(update, desc):
            if self.updateLink is None: return
            if type(desc) == dict:
                desc = json.dumps(desc)
            requests.post(self.updateLink, json={"update":{"status": update, "description": desc}})

        self.update = update

        self.update("☀️ Getting model ready...", "Warming up {} model".format(model))
        print("Warming up {} ".format(model))
        import sys, os
        import barervc
        self.barervc = barervc
        modelPath = os.path.abspath("/home/u3z9aphve/notmyvoice/models/"+ model)

        # Download model from S3 if not cached locally
        if not os.path.exists(modelPath):
            import os
            access_key = os.environ['STORJ_ACCESS_KEY']
            secret_key = os.environ['STORJ_SECRET_KEY']

            bucket_name = 'voicemodels'

            s3 = boto3.client(
                's3',
                endpoint_url='https://gateway.storjshare.io',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
            self.update("🆕 Updating model", "We're updating this model to the newest version. This will only happen once per update.")
            try:
                s3.download_file(bucket_name, f"{model}.zip", f"{model}.zip")
            except:
                try:
                    s3.download_file(bucket_name, f"custom/{model}.zip", f"{model}.zip")
                except:
                    return ("Could not find model in S3")
            import zipfile
            with zipfile.ZipFile(f"{model}.zip", 'r') as zip_ref:
                zip_ref.extractall(f"{model}")
            os.remove(f"{model}.zip")
            shutil.move(f"{model}", f"/home/u3z9aphve/notmyvoice/models/{model}")
        gc.collect()

        print("Searching {} for model files...".format(modelPath))
        indexFile, weightFile = None, None
        for root, dirs, files in os.walk(modelPath):
            print("> {}".format(root))
            for file in files:
                print(">> {}".format(file))
                if file.startswith("added_") and file.endswith(".index"):
                    indexFile = os.path.join(root, file)
                elif file.endswith(".pth") and "D_" not in file and "G_" not in file:
                    weightFile = os.path.join(root, file)
                if indexFile and weightFile:
                    break
        else:
            print("Missing something...")
            if not indexFile:
                indexFile = ""
            else:
                if not weightFile:
                    assert("Seems like the model is missing a weight file. Please contact support.")

        self.indexFile = indexFile
        self.weightFile = weightFile
        print("Using the following files:")
        print("indexFile: {}".format(indexFile))
        print("weightFile: {}".format(weightFile))
        barervc.myinfer.coldBoot("cuda:0", True, self.weightFile)
        self.update("✔️ Initalized", "Warmed up {} model".format(model))
        print("Warmed up {}".format(model))

    def inference(self, audio, transpose=0):
        gc.collect()
        def uploadFile(file_opened_as_rb):
            url = os.environ['CDN_UPLOAD_URL']
            r = requests.post(url, files={"file": file_opened_as_rb})
            return r.text

        import sys, os, requests, time, subprocess

        barervc = self.barervc

        with open("abnormal.m4a", "wb") as f:
            f.write(audio)

        ffmpeg_cmd = [
            'ffmpeg',
            '-y',
            '-i', 'abnormal.m4a',
            '-af', 'pan=mono|c0=FL+FR',
            '-af', 'loudnorm=I=-23:LRA=7:TP=-2.5:print_format=summary',
            '-ar', '44100',
            'audio.mp3'
        ]
        subprocess.call(ffmpeg_cmd)

        audio = os.path.abspath("audio.mp3")

        print("Running inference")
        self.update("🚀 Our AI model is doing its work...", "We're inferencing your track. This part usually takes the longest.")
        output = barervc.myinfer.inferExternally(transpose, audio, self.indexFile, "harvest", 0.8)

        print(output)
        with open(output, "rb") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Audio helper functions (local equivalents of Modal remote functions)
# ---------------------------------------------------------------------------

def merge(audio1, audio2, inc_volume=1.5):
    import os, ffmpeg
    with open("audio1.m4a", "wb") as f:
        f.write(audio1)

    with open("audio2.m4a", "wb") as f:
        f.write(audio2)

    audio1 = os.path.abspath("audio1.m4a")
    audio2 = os.path.abspath("audio2.m4a")

    print("Merging...")
    vocals_stream = ffmpeg.input(audio1).audio.filter("volume", inc_volume)
    no_vocals_stream = ffmpeg.input(audio2)
    combined = ffmpeg.filter([no_vocals_stream, vocals_stream], 'amix', inputs=2)
    combined = ffmpeg.output(combined, "merged.mp3", ac=2, ar=44100)
    ffmpeg.run(combined, overwrite_output=True)

    print("Finished merging")
    with open("merged.mp3", "rb") as f:
        return f.read()


def fextract(audio):
    "RETURNS VOCALS (BYTES) AND NO_VOCALS (BYTES)"
    print("[EXTRACT FUNC] Extracting...")
    def uploadFile(file_opened_as_rb):
        url = "https://cdn.notmyvoice.ai/supersecretupload"
        r = requests.post(url, files={"file": file_opened_as_rb})
        return r.text
    import requests, sys, os, subprocess
    with open("audio.mp3", "wb") as f:
        f.write(audio)

    print("[EXTRACT FUNC] Imported file...")
    audio = os.path.abspath("audio.mp3")

    result = {"vocals": None, "no_vocals": None}

    print("[EXTRACT FUNC] Running demucs...")
    import subprocess, ffmpeg
    subprocess.call(["demucs", "--two-stems=vocals", "-n", "htdemucs", 'audio.mp3'])

    trackName = os.path.splitext(os.path.basename(audio))[0]
    print("Extraction finished")

    if not os.path.exists(f"separated/htdemucs/audio/vocals.wav"):
        raise Exception("Error extracting vocals. Can't find vocals.wav")

    result["no_vocals"] = "separated/htdemucs/audio/no_vocals.wav"
    result["vocals"] = "separated/htdemucs/audio/vocals.wav"

    print("[EXTRACT FUNC] Sending back...")
    return {
        "no_vocals": open(result['no_vocals'], "rb").read(),
        "vocals": open(result['vocals'], "rb").read()
    }


# ---------------------------------------------------------------------------
# interface_instance — runs a single job (called inside worker subprocess)
# ---------------------------------------------------------------------------

def interface_instance(modelName: str, audio, transpose=0, extract=False, updateLink=None):
    extract = True if str(extract).lower() == "true" else False
    def update(update, desc):
        if updateLink is None: return
        import requests, json
        if type(desc) == dict:
            desc = json.dumps(desc)
        requests.post(updateLink, json={"update":{"status": update, "description": desc}})

    try:
        import os, sys, subprocess, ffmpeg, threading, uuid
        unique = "/home/u3z9aphve/notmyvoice/sessions/"+str(uuid.uuid4())
        os.mkdir(f"{unique}")
        os.chdir(f"{unique}")

        def uploadFile(file_opened_as_rb):
            import requests
            url = os.environ['CDN_UPLOAD_URL']
            r = requests.post(url, files={"file": file_opened_as_rb})
            return r.text

        with open("imported_audio.mp3", "wb") as f:
            f.write(audio)

        audio = os.path.abspath("imported_audio.mp3")

        results = {
            "vocals": None,
            "no_vocals": None,
            "merged": None
        }

        if extract:
            update("🎤 Separating vocals...", "We're separating the vocals from your track.")
            results = fextract(open("imported_audio.mp3", "rb").read())

            with open("vocals.wav", "wb") as f:
                f.write(results['vocals'])

            dbfs_vocals = AudioSegment.from_file("vocals.wav").dBFS

            print("[MAIN] Received from extract")
            results['vocals'] = os.path.abspath("vocals.wav")
            audio = results['vocals']

        print("[MAIN] Starting conversion...")
        with open(audio, "rb") as f:
            converted_vocals = RVC(modelName, updateLink).inference(f.read(), transpose)
            with open("converted_vocals.wav", "wb") as f:
                f.write(converted_vocals)
        print("[MAIN] Conversion finished")

        threads = []
        threads.append(threading.Thread(target=lambda: results.__setitem__("vocals", uploadFile(open("converted_vocals.wav", "rb")))))
        threads[-1].start()

        threads.append(threading.Thread(target=lambda: results.__setitem__("no_vocals", uploadFile(results['no_vocals']))))
        threads[-1].start()

        if extract:
            update("🎼 Merging vocals...", "We're merging the vocals back into your track.")

            with open("no_vocals.wav", "wb") as f:
                f.write(results['no_vocals'])
            dbfs_instrumental = AudioSegment.from_file("no_vocals.wav").dBFS
            dbfs_vocals = AudioSegment.from_file("vocals.wav").dBFS
            difference = max(dbfs_instrumental+5, dbfs_vocals)
            print(f"[MAIN] Difference: {difference}")
            print("[MAIN] Increasing gain...")
            sound = AudioSegment.from_file("converted_vocals.wav")
            sound = sound + difference
            sound.export("converted_vocals.wav", format="wav")
            print("[MAIN] Gain increased")

            merged_bytes = merge(open("converted_vocals.wav","rb").read(), results['no_vocals'])
            print("[MAIN] Received from merge")
            with open("merged.mp3", "wb") as f:
                f.write(merged_bytes)
            results['merged'] = uploadFile(open("merged.mp3", "rb"))

        for thread in threads: thread.join()

        for key in list(results.keys()):
            if "http" not in str(results[key]):
                del results[key]

        print("[MAIN] All threads finished")

        update("Done", results)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        update("ERROR", str(e))
        return "ERROR"


# ---------------------------------------------------------------------------
# Queue management — session cleanup
# ---------------------------------------------------------------------------

import requests, subprocess, time

def clean_up_sessions():
    """Delete all completed session folders and stale queue- temp files (>1 hour old)."""
    for folder in os.listdir("/home/u3z9aphve/notmyvoice/sessions"):
        try:
            shutil.rmtree(os.path.join("/home/u3z9aphve/notmyvoice/sessions", folder))
        except Exception as e:
            print("Failed to delete folder:", str(e))

    for file in os.listdir("/home/u3z9aphve/notmyvoice/"):
        if file.startswith("queue-"):
            if time.time() - os.path.getctime(os.path.join("/home/u3z9aphve/notmyvoice/", file)) >= 3600:
                subprocess.call(["rm", "-rf", os.path.join("/home/u3z9aphve/notmyvoice/", file)])


# ---------------------------------------------------------------------------
# Worker — serial job processor that drains the multiprocessing.Queue
# ---------------------------------------------------------------------------

import multiprocessing
from multiprocessing import Queue

worker_queue = Queue()

def worker(queue, interface_instance, unique):
    """
    Runs in a separate Process. Blocks on queue.get(), processes one job at a
    time, then broadcasts updated queue positions to all remaining jobs.

    Timeout logic:
      - 300s if the model isn't cached locally (includes S3 download time)
      - 90s if the model is already on disk
    """
    while True:
        print("Waiting for job...")

        args = queue.get()
        try:
            p = multiprocessing.Process(target=interface_instance, args=args)
            p.start()
            if not os.path.exists("/home/u3z9aphve/notmyvoice/models/{}".format(args[0])):
                p.join(300)
            else:
                p.join(90)
            res = p.exitcode
            if res == 1 or p.is_alive():
                [p.kill() if p.is_alive() else None]
                res = "Timeout"
                requests.post(args[4], json={"update": {"status": "ERROR", "description": "Your request timed out."}})
        except Exception as e:
            res = str(e)
            print("Error!")
        print("Result: {}".format(res))

        # Broadcast updated queue positions to all remaining waiting jobs
        queue_size = queue.qsize()
        queue_args = []
        for i in range(queue_size):
            queue_args.append(queue.get())
        for arg in queue_args:
            queue.put(arg)
        for i, arg in enumerate(queue_args):
            try:
                print("Sending update to", arg[4])
                requests.post(arg[4], json={"update": {"status": "🚧 Queued", "description": "Free members share limited resources in a queue. You're #{} in line.".format(i+1)}})
            except Exception as e:
                print("Failed to send update:", str(e))

        clean_up_sessions()


global worker_thread
worker_thread = None


# ---------------------------------------------------------------------------
# FastAPI endpoint — receives jobs from site/app.py (free-tier routing branch)
# ---------------------------------------------------------------------------

@app.post("/interface")
async def process_interface(model_name, transpose, extract, update_link, background_tasks: BackgroundTasks, audio: UploadFile = File(...)):
    """
    Called by site/app.py when a user is on the free tier:
        requests.post("https://california.notmyvoice.ai/interface", params={...}, files={"audio": f})

    Returns immediately with "Queued". The job is processed asynchronously by
    the worker process, which POSTs status updates back to update_link.
    """
    print("New request")
    async def update(update_status, description):
        if update_link is None:
            return
        try:
            requests.post(update_link, json={"update": {"status": update_status, "description": description}})
        except Exception as e:
            print("Failed to send update:", str(e))

    unique = "queue-" + str(uuid.uuid4()).split("-")[0]

    with open(unique, "wb") as f:
        f.write(await audio.read())

    worker_queue.put([model_name, open(unique, "rb").read(), transpose, extract, update_link])

    await update("🚧 Queued", "Free members share limited resources in a queue. You're #{} in line.".format(worker_queue.qsize()))

    # Lazily start the worker process if it died or hasn't been started yet
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = multiprocessing.Process(target=worker, args=(worker_queue, interface_instance, unique))
        worker_thread.start()

    gc.collect()
    return "Queued"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)
