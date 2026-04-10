"""
modal_worker.py — Modal serverless GPU worker (paid-tier inference)

Deployed to Modal and invoked via modal.Function.lookup('rvc', "interface").spawn(...)
from site/app.py when a user has paid credits.

The `interface` function is the main entry point. It:
  1. Ensures the model is present (downloads from S3 if not)
  2. Optionally extracts vocals with Demucs
  3. Runs RVC voice conversion on a T4 GPU
  4. Uploads results to CDN
  5. POSTs status updates back to the Flask callback URL (updateLink)
"""

import io
import modal
import zipfile
import shutil
from fastapi import FastAPI, File, UploadFile

image = (
    modal.Image.debian_slim()
    .pip_install("setuptools<58")
    .pip_install("demucs", "boto3", "requests", "pydub", "click==8.1.3", "fairseq==0.12.2", "faiss_cpu==1.7.2", "ffmpeg==1.4", "ffmpeg_python==0.2.0", "librosa==0.9.2", "numpy==1.23.5", "praat-parselmouth>=0.4.3", "pyworld==0.3.3", "scipy==1.10.1", "termcolor==2.3.0", "torch==2.0.1", "torchcrepe")
    .run_commands("apt-get update", "apt-get install -y ffmpeg")
)
volume = modal.SharedVolume().persist("packages")
volume2 = modal.SharedVolume().persist("models")
cacheVolume = modal.SharedVolume().persist('.cache')
stub = modal.Stub(name="rvc")


# ---------------------------------------------------------------------------
# RVC class — warm GPU worker with persistent model state across inference calls
# ---------------------------------------------------------------------------

@stub.cls(shared_volumes={"/root/packages": volume, "/root/models": volume2, '/root/.cache': cacheVolume}, image=image, gpu="T4", allow_cross_region_volumes=True, timeout=600)
class RVC():
    def __init__(self, model, updateLink=None):
        import json
        self.updateLink = updateLink
        def update(update, desc):
            print(update, desc)
            if self.updateLink is None: return
            import requests
            if type(desc) == dict:
                desc = json.dumps(desc)
            requests.post(self.updateLink, json={"update":{"status": update, "description": desc}})
        self.update = update

        self.update("☀️ Getting model ready...", "Warming up {} model".format(model))
        print("Warming up {} ".format(model))
        import sys, os
        sys.path.append("/root/packages")
        import barervc
        self.barervc = barervc
        modelPath = os.path.abspath("/root/models/"+ model)
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
        barervc.myinfer.coldBoot("cuda:0", True, self.weightFile)
        self.update("✔️ Initalized", "Warmed up {} model".format(model))
        print("Warmed up {}".format(model))

    @modal.method()
    def inference(self, audio, transpose=0):
        def uploadFile(file_opened_as_rb):
            url = os.environ['CDN_UPLOAD_URL']
            r = requests.post(url, files={"file": file_opened_as_rb})
            return r.text

        import sys, os, requests, time, subprocess
        sys.path.append("/root/packages")

        barervc = self.barervc

        with open("abnormal.m4a", "wb") as f:
            f.write(audio)

        ffmpeg_cmd = [
            'ffmpeg',
            '-i', 'abnormal.m4a',
            '-af', 'pan=mono|c0=FL+FR',
            '-af', 'loudnorm=I=-23:LRA=7:TP=-2.5:print_format=summary',
            '-ar', '44100',
            'audio.m4a'
        ]
        subprocess.call(ffmpeg_cmd)

        audio = os.path.abspath("audio.m4a")

        print("Running inference")
        self.update("🚀 Our AI model is doing its work...", "We're inferencing your track. This part usually takes the longest.")

        output = barervc.myinfer.inferExternally(transpose, audio, self.indexFile, "harvest", 0.8)

        with open(output, "rb") as f:
            return f.read()


# ---------------------------------------------------------------------------
# Helper Modal functions
# ---------------------------------------------------------------------------

@stub.function(shared_volumes={"/root/packages": volume, "/root/models": volume2, '/root/.cache': cacheVolume}, image=image, allow_cross_region_volumes=True, timeout=600)
def merge(audio1, audio2):
    import os, ffmpeg
    with open("audio1.m4a", "wb") as f:
        f.write(audio1)

    with open("audio2.m4a", "wb") as f:
        f.write(audio2)

    audio1 = os.path.abspath("audio1.m4a")
    audio2 = os.path.abspath("audio2.m4a")

    print("Merging...")
    vocals_stream = ffmpeg.input(audio1).audio.filter("volume", 1.5)
    no_vocals_stream = ffmpeg.input(audio2)
    combined = ffmpeg.filter([no_vocals_stream, vocals_stream], 'amix', inputs=2)
    combined = ffmpeg.output(combined, "merged.mp3", ac=2, ar=44100)
    ffmpeg.run(combined, overwrite_output=True)

    print("Finished merging")
    with open("merged.mp3", "rb") as f:
        return f.read()


@stub.function(shared_volumes={"/root/packages": volume, "/root/models": volume2, '/root/.cache': cacheVolume}, image=image, allow_cross_region_volumes=True, timeout=600, gpu="T4")
def fextract(audio):
    "RETURNS VOCALS (BYTES) AND NO_VOCALS (LINK TO FILE)"
    print("[EXTRACT FUNC] Extracting...")
    def uploadFile(file_opened_as_rb):
        url = os.environ['CDN_UPLOAD_URL']
        r = requests.post(url, files={"file": file_opened_as_rb})
        return r.text
    import requests, sys, os, subprocess
    with open("audio.mp3", "wb") as f:
        f.write(audio)

    print("[EXTRACT FUNC] Imported file...")
    audio = os.path.abspath("audio.mp3")

    result = {
        "vocals": None,
        "no_vocals": None
    }

    print("[EXTRACT FUNC] Running demucs...")
    import subprocess, ffmpeg
    subprocess.call(["demucs", "--two-stems=vocals", "-n", "htdemucs", 'audio.mp3'])

    trackName = os.path.splitext(os.path.basename(audio))[0]
    print("Extraction finished")

    if not os.path.exists(f"/root/separated/htdemucs/audio/vocals.wav"):
        raise Exception("Error extracting vocals. Can't find vocals.wav")

    result["no_vocals"] = "/root/separated/htdemucs/audio/no_vocals.wav"
    result["vocals"] = "/root/separated/htdemucs/audio/vocals.wav"

    print("[EXTRACT FUNC] Sending back...")
    return {
        "no_vocals": open(result['no_vocals'], "rb").read(),
        "vocals": open(result['vocals'], "rb").read()
    }


@stub.function(shared_volumes={"/root/packages": volume, "/root/models": volume2, '/root/.cache': cacheVolume}, image=image, allow_cross_region_volumes=True, timeout=600)
def verify_model_exists(modelName: str, updateFunc) -> bool:
    import os
    if os.path.exists(f"/root/models/{modelName}"):
        return True
    else:
        import boto3, shutil
        access_key = os.environ['STORJ_ACCESS_KEY']
        secret_key = os.environ['STORJ_SECRET_KEY']

        bucket_name = 'voicemodels'

        s3 = boto3.client(
            's3',
            endpoint_url='https://gateway.storjshare.io',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
        )
        updateFunc("🆕 Updating model", "We're updating this model to the newest version. This will only happen once per update.")
        try:
            s3.download_file(bucket_name, f"{modelName}.zip", f"{modelName}.zip")
        except:
            try:
                s3.download_file(bucket_name, f"custom/{modelName}.zip", f"{modelName}.zip")
            except:
                updateFunc("❌ Model not found", "We couldn't find this model in our database. Please contact support ASAP, this shouldn't happen.")
                raise Exception("Model not found")

        import zipfile
        with zipfile.ZipFile(f"{modelName}.zip", 'r') as zip_ref:
            zip_ref.extractall(f"{modelName}")
        os.remove(f"{modelName}.zip")
        shutil.move(f"{modelName}", f"/root/models/{modelName}")
        return True


# ---------------------------------------------------------------------------
# interface — main entry point spawned by site/app.py for paid-tier jobs
# ---------------------------------------------------------------------------

@stub.function(shared_volumes={"/root/packages": volume, "/root/models": volume2, '/root/.cache': cacheVolume}, image=image, allow_cross_region_volumes=True, timeout=600)
def interface(modelName: str, audio, transpose=0, extract=False, updateLink=None):
    try:
        import os, sys, subprocess, ffmpeg, threading, boto3

        def update(update, desc):
            print(update, desc)
            if updateLink is None: return
            import requests, json
            if type(desc) == dict:
                desc = json.dumps(desc)
            requests.post(updateLink, json={"update":{"status": update, "description": desc}})

        verify_model_exists.call(modelName, update)

        def uploadFile(file_opened_as_rb):
            import requests, os
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
            results = fextract.call(open("imported_audio.mp3", "rb").read())

            with open("vocals.wav", "wb") as f:
                f.write(results['vocals'])
            print("[MAIN] Received from extract")
            results['vocals'] = os.path.abspath("vocals.wav")
            audio = results['vocals']

        print("[MAIN] Starting conversion...")
        with open(audio, "rb") as f:
            converted_vocals = RVC.remote(modelName, updateLink).inference.call(f.read(), transpose)
            with open("converted_vocals_no_silences.wav", "wb") as f:
                f.write(converted_vocals)
        print("[MAIN] Conversion finished")

        threads = []
        threads.append(threading.Thread(target=lambda: results.__setitem__("vocals", uploadFile(open("converted_vocals_no_silences.wav", "rb")))))
        threads[-1].start()

        if extract:
            with open("no_vocals.mp3", "wb") as f:
                f.write(results['no_vocals'])
            threads.append(threading.Thread(target=lambda: results.__setitem__("no_vocals", uploadFile(open("no_vocals.mp3", "rb")))))
            threads[-1].start()
            update("🎼 Merging vocals...", "We're merging the vocals back into your track.")
            merged_bytes = merge.call(open("converted_vocals_no_silences.wav","rb").read(), results['no_vocals'])
            print("[MAIN] Received from merge")
            with open("merged.mp3", "wb") as f:
                f.write(merged_bytes)
            results['merged'] = uploadFile(open("merged.mp3", "rb"))

        for thread in threads: thread.join()
        new = {}
        for key,val in results.items():
            if val is not None:
                new[key] = val
        results = new

        print(results)
        update("Done", results)
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(e)
        return "ERROR"
