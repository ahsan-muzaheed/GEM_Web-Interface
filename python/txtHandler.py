import os

targetPath = os.path.join(os.getcwd(), "python/results/result.txt")


def txtCreater():

    file1 = open(targetPath, "w").close()
    # file1 = open(targetPath, "a")


def txtAppender(content):

    file1 = open(targetPath, "a")

    for i in range(len(content)):
        print(content[i])
        file1.write(content[i] + "\n")

    file1.close()
