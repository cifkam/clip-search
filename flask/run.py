import os
import secrets
from multiprocessing import Process
from multiprocessing.connection import Listener, Client
from pathlib import Path
from datetime import datetime
from settings import settings

local_passwd = secrets.token_bytes(16)

def process_main():
    from app import run_app, FlaskExitException

    address = ('localhost', settings.LOCALHOST_PORT)
    conn = Client(address, authkey=local_passwd)

    #filename = log_dir / ("log-" + datetime.now().strftime("%Y-%m-%d_%M-%H-%S") + ".txt")
    #with open(str(filename), "w", buffering=1) as f:
        #run_app(conn, f)
    run_app(conn)


def run_process():
    process = Process(target=process_main)
    process.start()
    return process


def stop_process(process, timeout=5):
    process.join(timeout)
    if process.is_alive():
        print("App is still running after {timeout} seconds. Force stopping.")
        process.terminate()


def main():
    log_dir = Path(os.path.realpath(__file__)).parent / "logs"
    log_dir.mkdir(exist_ok=True)

    process = None
    conn = None
    address = ('localhost', settings.LOCALHOST_PORT)
    listener = Listener(address, authkey=local_passwd)

    def start_app():
        nonlocal process, conn
        settings.load()
        process = run_process()
        print("Waiting for connection...")
        conn = listener.accept()
        print("Connection established!")

    def stop_app():
        conn.close()
        stop_process(process)
        print("App stopped!")
    

    start_app()
    while True:

        data = conn.recv()
        print(":", data)

        if data == "restart":
            print("Restarting...")
            stop_app()
            start_app()

        elif data == "shutdown":
            print("Shutting down...")
            stop_app()
            break
    

    print("Bye bye!")


if __name__ == "__main__":
    main()