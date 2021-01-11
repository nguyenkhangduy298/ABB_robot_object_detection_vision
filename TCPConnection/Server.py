# Create a server python
import random
import socket
from enum import Enum

BUFFER_SIZE = 1024

hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)

PORT = 5000
IP = local_ip
print(hostname)
print(IP)


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


def ServerInit():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((IP, PORT))
    s.listen(1)
    conn, address = s.accept()

    return s, conn, address


def ServerTerminate(conn):
    conn.close()


def ProtocolSelection(command):
    return 1


def generatePos():
    a = random.randint(400, 800)
    b = random.randint(400, 800)
    c = random.randint(400, 800)
    pos1 = [450, 0, 700]
    pos2 = [400, 100, 600]
    pos3 = [600, 0, 600]
    pos4 = [600, 150, 600]

    array = [pos1, pos2, pos3, pos4]
    random_pos = random.randint(0, len(array))
    return array[random_pos]


def main():
    # s = ServerInit()
    # print(Code.TARGET_DELIVERED.value)

    socket, conn, address = ServerInit()
    print('Connected by', address)
    while True:
        receive = conn.recv(BUFFER_SIZE).decode('utf-8')
        print("'" + receive + "'")

        if (receive == '0xFF'):
            data = str(generatePos())
            print("Object location request")
            conn.send(data.encode())

        # if (receive == Code.TARGET_DELIVERED.value):
        #     print("Object delivered")
        #
        # elif (receive== Code.STANDY_MOVED.value):
        #     print("Stand by")
        #
        # elif (receive == Code.REQUEST_LOCATION.value):
        #     data=str(generatePos())
        #     print("Object location request")
        #     socket.send(data.encode())
        #
        # elif (receive==Code.MOVING.value):
        #     print("Moving")
        #
        # elif (receive==Code.SERVER_SHUTDOWN.value):
        #     print("server closed")
        #     break;
        # else:
        #     pass;


if __name__ == "__main__":
    main()
