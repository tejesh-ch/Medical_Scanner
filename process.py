import os

path = os.path.join(os.getcwd(), "Training", "no_tumor")
os.chdir(path)
i = 0
for label in os.listdir():
    name = "no_tumor_example" + str(i+1) + ".jpg"
    os.rename(label, name)
    i += 1
