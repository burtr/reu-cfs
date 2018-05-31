
#
# the hello world program (with a fancy twist)
#


def hello_world(name=""):
    if len(name)>0:
        print("Hello {}".format(name))
    else:
        print("Hello World!")


hello_world()

