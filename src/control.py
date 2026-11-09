import socket
import time
from typing import Optional

class RobotServer:
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 20002,
                 bufsize: int = 1024):
        self.host    = host
        self.port    = port
        self.bufsize = bufsize

        self._srv:  Optional[socket.socket] = None
        self.conn:  Optional[socket.socket] = None

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        conn, addr = srv.accept()
        self._srv = srv
        self.conn = conn

    def send(self, msg: str) -> None:
        if not self.conn:
            raise RuntimeError("Connection not established. Call start() first.")
        self.conn.sendall(msg.encode("utf-8"))

    def receive(self) -> Optional[str]:
        if not self.conn:
            raise RuntimeError("Connection not established. Call start() first.")
        data = self.conn.recv(self.bufsize)
        if not data:
            return None
        text = data.decode("utf-8").strip()
        return text

    def close(self) -> None:
        if self.conn:
            self.conn.close()
        if self._srv:
            self._srv.close()

    def move_delta(self, cx: int, cy: int):
        delta_x = str(int(cx*0.3))
        delta_y = str(int(cy*0.3))
        
        self.send("x")
        time.sleep(0.1)
        self.send(delta_x)
        print("move x")
        self.rbt_wait()
        self.send("y")
        time.sleep(0.1)
        self.send(delta_y)
        print("move y")
        self.rbt_wait()

    def rbt_wait(self):
        print("wait for finish signal...")
        while True:
            robot_msg = self.receive()
            if robot_msg == 'finish':
                break
            elif robot_msg == 'no':
                print("no action in robot")
                break
            else:
                continue
        print("start next action.")
        time.sleep(1)

if __name__ == "__main__":
    server = RobotServer(host="192.168.137.50", port=20002)
    server.start()
    server.move_delta(15,15)
    server.move_delta(15,15)
    server.move_delta(15,15)