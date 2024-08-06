import os

removefiles = []
removeimges = []

def checkfile(picPath, filePath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    files = os.listdir(filePath)
    for i, name in enumerate(files):
        txtFile = open(filePath + name)
        try:
            img = open(picPath + name[0:-4] + ".jpg")
        except:
            print("can not find the image")
            removefiles.append(name)
            continue
    # imges=os.listdir(picPath)
    # for i, name in enumerate(imges):

    #     if name[0:-4] not in files:
    #         print("can not find the txt")
    #         removeimges.append(name)
    #         continue
def removefile(txtPath):
    # 删除文件夹下所有文件
    for file in removefiles:
        os.remove(txtPath + file)
    
if __name__ == "__main__":
    # picPath = "dataset/JPEGImages/"  # 图片所在文件夹路径，后面的/一定要带上
    # txtPath = "dataset/YOLO/"  # txt所在文件夹路径，后面的/一定要带上
    # xmlPath = "dataset/annotations/"  # xml文件保存路径，后面的/一定要带上
    picPath = r"C:/Users\Alexander Hamilton/Desktop/work/plant/archive (5)/test/images/"  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = r"C:/Users/Alexander Hamilton/Desktop/work/plant/archive (5)/test/labels/"  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = r"C:/Users/Alexander Hamilton/Desktop/work/plant/archive (5)/test/voc/"  # xml文件保存路径，后面的/一定要带上
    checkfile(picPath, txtPath)
    removefile(txtPath)