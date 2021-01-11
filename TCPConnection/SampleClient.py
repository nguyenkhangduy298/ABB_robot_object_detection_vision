import socket

HOST = '192.168.125.1'
PORT = 1025

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    server_address = (HOST, PORT)
    print('connecting to %s port ' + str(server_address))
    s.connect(server_address)
    print("Connected")

try:
    while True:
        msg = input('Client: ')
        s.sendall(msg.encode())

        if msg == "quit":
            break

        data = s.recv(1024)
        print('Server: ', data.decode("utf8"))
finally:
    print("End")
