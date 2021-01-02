import xml.dom.minidom as Dom
import os
import random
import yaml
import shutil


class ObjClass:
    def __init__(self, label_idx, label_name):
        self.label_idx = label_idx
        self.label_name = label_name


class Label:
    def __init__(self, label_name=None, x_center=None, y_center=None, width=None, height=None):
        self.label_idx = 0.
        self.label_name = label_name
        self.x_center = x_center
        self.y_center = y_center
        self.width = width
        self.height = height

    def set_label_idx(self, label_idx):
        self.label_idx = label_idx


def get_classes(yaml_file):
    obj_classes = []
    with open(yaml_file, 'r') as stream:
        data_load = yaml.safe_load(stream)
        # nc = data_load.get("nc", None)
        names = data_load.get("names", None)
        for idx, class_name in enumerate(names):
            obj_class = ObjClass(idx, class_name)
            obj_classes.append(obj_class)
    return obj_classes


def get_annotations(annotation_file):
    file = annotation_file
    dom = Dom.parse(file)
    root = dom.documentElement
    image_size = root.getElementsByTagName("size")[0]
    image_width = image_size.getElementsByTagName(
        "width")[0].childNodes[0].data
    image_height = image_size.getElementsByTagName("height")[
        0].childNodes[0].data
    objects = root.getElementsByTagName("object")
    annotations = []
    try:
        for obj in objects:
            name = obj.getElementsByTagName("name")[0].firstChild.nodeValue
            if obj.getElementsByTagName("bndbox"):
                # convert LabelImg xml bbox format to YOLO label format
                bndbox = obj.getElementsByTagName("bndbox")[0]
                xmin = bndbox.getElementsByTagName(
                    "xmin")[0].firstChild.nodeValue
                ymin = bndbox.getElementsByTagName(
                    "ymin")[0].firstChild.nodeValue
                xmax = bndbox.getElementsByTagName(
                    "xmax")[0].firstChild.nodeValue
                ymax = bndbox.getElementsByTagName(
                    "ymax")[0].firstChild.nodeValue
                dw = 1. / float(image_width)
                dh = 1. / float(image_height)
                w = float(xmax) - float(xmin)
                h = float(ymax) - float(ymin)
                x = (float(xmax) + float(xmin)) / 2.0 - 1
                y = (float(ymax) + float(ymin)) / 2.0 - 1
                x = x * dw
                y = y * dh
                w = w * dw
                h = h * dh
                label = Label(name, x, y, w, h)
                annotations.append(label)
            else:
                print("No bbox in annotation file!!!")

    except ZeroDivisionError:
        print(annotation_file + " " + "width or height is zero, remove it from dataset")

    return annotations


# covert label name to label index, start from 0...
def label_set_idx(annonations, obj_classes):
    labels = []
    for label in annonations:
        for obj_class in obj_classes:
            if obj_class.label_name == label.label_name:
                label.set_label_idx(obj_class.label_idx)
                labels.append(label)

    return labels


def write_labels(labels, label_file):
    with open(label_file, "w") as nlf:
        for label in labels:
            nlf.write(str(label.label_idx) +
                      " " +
                      str(label.x_center) +
                      " " +
                      str(label.y_center) +
                      " " +
                      str(label.width) +
                      " " +
                      str(label.height) +
                      "\n")


def prepare_labels_data(ori_images_dir, ori_annotations_dir, yolo5_labels_dir,
                        base_dir, custom_dir, images_dir, labels_dir):

    ori_images = os.listdir(ori_images_dir)
    print(len(ori_images))
    ori_annotations = os.listdir(ori_annotations_dir)
    print(len(ori_annotations))

    if ".DS_Store" in ori_images:
        ori_images.remove(".DS_Store")

    if os.path.exists(yolo5_labels_dir):
        shutil.rmtree(yolo5_labels_dir)
    os.mkdir(yolo5_labels_dir)

    for ori_image in ori_images:
        filename = ori_image.split('/')[-1].split('.')[0]
        annotation_file = str(filename) + ".xml"
        label_file = yolo5_labels_dir + "/" + annotation_file.replace(".xml", ".txt")
        if annotation_file in ori_annotations:
            annotations = get_annotations(ori_annotations_dir + "/" + annotation_file)
            if len(annotations) == 0:
                ori_annotations.remove(annotation_file)
            else:
                # labels = label_set_idx(annotations, obj_classes)  # all objects' label index is 0
                write_labels(annotations, label_file)
        else:
            # 如果image没有对应的标注文件(annotation file)，则产生一条全0的Lable(Lable.idx=-1)，写入label文件
            print("This image no related annotation")
            # label = Label()
            # label.set_label_zero()
            # label.label_idx = -1.
            # labels = [label]


def define_set(ori_images_dir, dataset, base_dir, custom_dir, images_dir, labels_dir, yolo5_labels_dir, set_name):
    images_set_dir = base_dir + "/" + custom_dir + "/" + images_dir + "/" + set_name
    labels_set_dir = base_dir + "/" + custom_dir + "/" + labels_dir + "/" + set_name
    if os.path.exists(images_set_dir):
        shutil.rmtree(images_set_dir)
    os.mkdir(images_set_dir)
    if os.path.exists(labels_set_dir):
        shutil.rmtree(labels_set_dir)
    os.mkdir(labels_set_dir)
    for file in dataset:
        label = file.split('/')[-1].split('.')[0] + ".txt"
        image = label.split('.')[0] + ".jpg"
        shutil.copy(yolo5_labels_dir + "/" + label, labels_set_dir + "/" + label)
        shutil.copy(ori_images_dir + "/" + image, images_set_dir + "/" + image)


def create_whiteboard_dataset():
    ori_images_dir = "../ML_finalproject/SCUT_HEAD_Part_B/JPEGImages"
    ori_annotations_dir = "../ML_finalproject/SCUT_HEAD_Part_B/Annotations"
    yolo5_labels_dir = "../ML_finalproject/SCUT_HEAD_Part_B/yolo5_labels"
    base_dir = os.path.abspath(os.path.dirname(__file__))
    custom_dir = "data/datasets"
    images_dir = "whiteboard/images"
    labels_dir = "whiteboard/labels"
    prepare_labels_data(ori_images_dir, ori_annotations_dir, yolo5_labels_dir,
                        base_dir, custom_dir, images_dir, labels_dir)


if __name__ == "__main__":
    create_whiteboard_dataset()
