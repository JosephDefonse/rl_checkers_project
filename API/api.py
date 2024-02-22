import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('API_KEY')

class Controller:

    def __init__(self):
        self.url = "http://172.20.10.2"
        self.headers = { 'Content-Type': 'application/json', 'X-Api-Key': API_KEY }
        self.printhead = self.url + '/api/printer/printhead'

        self.device_id = self.configure_switch()

    def move_xy(self, x, y):
        """ Moves the handler over the (X-Y Plane). Optimized for speed """

        command = {'command': 'jog',
                    'x': x,
                    'y': y,
                    'absolute': True }

        requests.post(self.printhead, headers=self.headers, json=command)

    def move_z(self, z):
        """ Moves the handler over the (Z Plane) """

        command = {'command': 'jog',
                    'z': z,
                    'absolute': True }

        requests.post(self.printhead, headers=self.headers, json=command)

    def await_position(self):
        """ Waits until the handler is in position """

        command = {'command': 'query_endstops'}
        r = requests.post(self.printhead, headers=self.headers, json=command)
        while r.json()['endstops']['x_min'] == False:
            r = requests.post(self.printhead, headers=self.headers, json=command)

    def configure_switch(self):
        headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer X",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive"
                }

        url = "https://api.smartthings.com/v1/devices"

        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()['items'][0]['deviceId']
        else:
            print("error")

    def pick_up(self, value):
        headers = {
                    "Content-Type": "application/json",
                    "Authorization": "Bearer X",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive"
                }

        body = {
            "commands": [
                {
                "component": "main",
                "capability": "switch",
                "command": value
                }
            ]
        }

        device = self.device_id
        url = "https://api.smartthings.com/v1/devices/"+str(device)+"/commands"

        response = requests.post(url, json=body, headers=headers)
        if response.status_code != 200:
            print("error")


if __name__ == "__main__":
    cb = Controller()
    # cb.move_xy(200,200)
    # cb.move_z(200)
    cb.pick_up("on")