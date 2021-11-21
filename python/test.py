newdata = []

with open("requirements.txt") as f:

    data = f.read()
    data = data.split("\n")
    for i in data:
        if "@" not in i:
            newdata.append(i)
    # print(newdata)


file = open("requirements.txt", "w")
# print("".join(newdata) + "\n")
for i in newdata:
    print(i)
    file.write(i + "\n")
