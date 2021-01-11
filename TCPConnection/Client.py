import socket
from enum import Enum


class Code(Enum):
    # Sending from server (this end)
    PAUSE_ON_ROAD = "0xAF"
    STOP_AND_SHUTDOWN = "0xAE"
    REQUEST_LOCATION = "0xAC"
    NONE_OBJECT_DETECTED = "0xAB"
    SERVER_SHUTDOWN = "0xAA"
    # Recieve from ROBOT
    TARGET_POS_REQUETS = "0xFF"
    TARGET_ORIENTATION_REQUEST = "0xFE"
    TARGET_DELIVERED = "0xFD"
    STANDY_MOVED = "0xFC"
    MOVING = "0xFB"
    CLIENT_ERROR = "0xFA"
    OUTSIDE_OF_REACH = "OxAD"
    # Default
    NONE = ""
    # Client Code:
    CONNECTED = "Connected"


class Client:
    BUFFER_SIZE = 4096
    CONNECTED = False

    def __init__(self, IP_server, Port_server):
        self.connectIP = IP_server
        self.connectPort = Port_server
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def requestStatus(self):
        try:
            self.socket.connect((self.connectIP, self.connectPort))
            return (Code.CONNECTED, self.socket.recv(self.BUFFER_SIZE))
        except Exception:
            return Exception
    # def SendSignal(self):


if __name__ == '__main__':
    newClient = Client(IP_server='192.168.125.1', Port_server=1025)
    str = newClient.requestStatus()
    print(str)
