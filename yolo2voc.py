from xml.dom.minidom import Document
import os
import cv2


# def makexml(txtPath, xmlPath, picPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径

removefiles = []

def makexml(picPath, txtPath, xmlPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    """
    dic = {'0': "mildew",  # 创建字典用来对类型进行转换
           '1': "rose_P01",  # 此处的字典要与自己的classes.txt文件中的类对应，且顺序要一致
           '2' :'rose_R02'
           }
    filenames = " "
    
    files = os.listdir(txtPath)
    for i, name in enumerate(files):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath + name)
        txtList = txtFile.readlines()
        # img = cv2.imread(picPath + name[0:-4] + ".jpg")
        try:
            img = cv2.imread(picPath + name[0:-4] + ".jpg")
            Pheight, Pwidth, Pdepth = img.shape
        except:
            print("can not find the image")
            removefiles.append(name)
            continue

        
        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        # filenames.append(filenamecontent)
        filenames = name[0:-4] + ".jpg"
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束

        f = open(xmlPath + name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()
        with open(xmlPath + Main_names + ".txt", 'a') as fil:
            fil.write(str(filenames)+"\n")
def removefile(txtPath):
    # 删除文件夹下所有文件
    for file in removefiles:
        os.remove(txtPath + file)


Main_names="test"  #test train 的名字
if __name__ == "__main__":
    # picPath = "dataset/JPEGImages/"  # 图片所在文件夹路径，后面的/一定要带上
    # txtPath = "dataset/YOLO/"  # txt所在文件夹路径，后面的/一定要带上
    # xmlPath = "dataset/annotations/"  # xml文件保存路径，后面的/一定要带上
    picPath = r"C:/Users\Alexander Hamilton/Desktop/work/plant/archive (5)/valid/images/"  # 图片所在文件夹路径，后面的/一定要带上
    txtPath = r"C:/Users/Alexander Hamilton/Desktop/work/plant/archive (5)/valid/labels/"  # txt所在文件夹路径，后面的/一定要带上
    xmlPath = r"C:/Users/Alexander Hamilton/Desktop/work/plant/archive (5)/valid/voc/"  # xml文件保存路径，后面的/一定要带上
    makexml(picPath, txtPath, xmlPath)
    # removefile(txtPath)#删除txt文件删除前记得备份
