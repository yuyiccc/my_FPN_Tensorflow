import shutil
import os

data_list_file_path  = '/home/laner/Desktop/FPN/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
out_root_path = '/home/laner/Desktop/FPN/data'
out_image_path = out_root_path+'/VOCdevkit/JPEGImages'
out_xml_path = out_root_path+'/VOCdevkit/Annotations'
source_root_path = '/home/laner/Desktop/FPN/'
source_image_path = source_root_path+'/VOCdevkit/VOC2012/JPEGImages'
source_xml_path = source_root_path+'/VOCdevkit/VOC2012/Annotations'



#if not os.path.exists(out_root_path):
#    os.mkdir(out_root_path)
if not os.path.exists(out_xml_path):
    os.makedirs(out_xml_path)
if not os.path.exists(out_image_path):
    os.makedirs(out_image_path)

data_name_list  = open(data_list_file_path).readlines()
for name_i in data_name_list:
    name_i =  name_i[:-1]
    im_file = os.path.join(source_image_path,name_i+'.jpg')
    xml_file = os.path.join(source_xml_path,name_i+'.xml')
    out_im_file = os.path.join(out_image_path,name_i+'.jpg')
    out_xml_file = os.path.join(out_xml_path,name_i+'.xml')
    shutil.copyfile(im_file,out_im_file)
    shutil.copyfile(xml_file,out_xml_file)

