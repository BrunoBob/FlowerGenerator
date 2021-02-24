from PIL import Image
import os

INPUT_FOLDER = '../DataBase/flowers/'
OUTPUT_FOLDER = '../DataBase/scaledFlowersSmall/'

allFiles = os.listdir(INPUT_FOLDER)
step = int(round(len(allFiles) / 100))

for index,file in enumerate(allFiles):
    img = Image.open(os.path.join(INPUT_FOLDER,file))
    newImg = img.resize((100,100))
    newImg.save(os.path.join(OUTPUT_FOLDER,file))

    #Print progression
    if(index%step == 0):
        print("Progress : ", (index/step)+1, "%")
