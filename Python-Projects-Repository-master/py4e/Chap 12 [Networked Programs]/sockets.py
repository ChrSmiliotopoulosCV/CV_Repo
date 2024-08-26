import socket
# The line of code that follows makes a sockets that goes across the
# internet, and it is a stream socket (a series of characters that come
# one after the other, rather than a series of blocks of text.)
# The only thing we shoulg know about the following statement is that it
# makes a socket but it does not associate it.
mysock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
mysock.connect(('data.pr4e.org', 80))
cmd = ('GET http://data.pr4e.org/romeo.txt HTTP/1.0\n\n').encode()
mysock.send(cmd)

while True:
    data = mysock.recv(512)
    if (len(data) < 1):
        break
    print(data.decode())
mysock.close()