import mmap


# def regular_io(filename):
#     with open("InputOutputDemo.txt", "r+") as fileIO:
#         fio = fileIO.read()
#         print(fio)
#
#
# regular_io("InputOutputDemo.txt")


def mmap_io(filename):
    with open(filename, mode="r") as fileIO:
        with mmap.mmap(fileIO.fileno(), length=0, access=mmap.ACCESS_READ) as fileIO_obj:
            text = fileIO_obj.read()
            print(text)


mmap_io(("InputOutputDemo.txt"))
