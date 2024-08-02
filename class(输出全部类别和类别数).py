import os
import xml.dom.minidom

xml_path = r'C:\Users\lenovo\Desktop\agriculture\dataset\PlantDoc-Object-Detection-Dataset-master\Annotations/'
files = os.listdir(xml_path)

gt_dict = {}

if __name__ == '__main__':

    for xm in files:
        if xm.endswith('.xml'):
            xmlfile = xml_path + xm
            print(xmlfile)
            dom = xml.dom.minidom.parse(xmlfile)  # 读取xml文档
            root = dom.documentElement  # 得到文档元素对象
            filenamelist = root.getElementsByTagName("filename")
            filename = filenamelist[0].childNodes[0].data
            objectlist = root.getElementsByTagName("object")
            ##
            for objects in objectlist:
                namelist = objects.getElementsByTagName("name")
                objectname = namelist[0].childNodes[0].data
                if objectname == '-':
                    print(filename)
                if objectname in gt_dict:
                    gt_dict[objectname] += 1
                else:
                    gt_dict[objectname] = 1
                # for nl in namelist:
                #     objectname = nl.childNodes[0].data
                #     if objectname in gt_dict:
                #         gt_dict[objectname] += 1
                #     else:
                #         gt_dict[objectname] = 1
    dic = sorted(gt_dict.items(), key=lambda d: d[1], reverse=True)
    print(dic)
    print(len(dic))