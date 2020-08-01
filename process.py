import os

path = os.path.join(os.getcwd(), "Training")
os.chdir(path)
for child in os.listdir():
    dir = os.getcwd()
    os.chdir(os.path.join(dir, child))
    i = 0
    for label in os.listdir():
        name = child + "_example_" + str(i+1) + ".jpg"
        os.rename(label, name)
        i += 1
    os.chdir(dir)

path = os.path.join(os.getcwd(), "Training", "no_tumor")
os.chdir(path)
i = 0
for label in os.listdir():
    name = "no_tumor_example" + str(i+1) + ".jpg"
    os.rename(label, name)
    i += 1
