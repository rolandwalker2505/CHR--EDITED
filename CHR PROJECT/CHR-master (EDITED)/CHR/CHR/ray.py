import csv
import os
import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import multiprocessing

object_categories = ['Gun', 'Knife', 'Wrench', 'Pliers', 'Scissors']

def read_image_label(file):
    print('[dataset] read ' + file)
    data = dict()
    with open(file, 'r') as f:
        for line in f:
            tmp = line.split(' ')
            name = tmp[0]
            label = int(tmp[-1])
            data[name] = label
    return data

def write_object_labels_csv(file, labeled_data):
    print('[dataset] write file %s' % file)
    with open(file, 'w', newline='') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(object_categories)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for (name, labels) in labeled_data.items():
            example = {'name': name}
            for i in range(5):
                example[fieldnames[i + 1]] = int(labels[i])
            writer.writerow(example)

def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

class XrayClassification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = root
        self.path_images = os.path.join(root, 'JPEGImage')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        path_csv = os.path.join(self.root, 'ImageSet', 'train_test_10')
        file_csv = os.path.join(path_csv, set + '.csv')

        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):
                os.makedirs(path_csv)
            labeled_data = {}  # Initialize empty labeled_data if file doesn't exist
            write_object_labels_csv(file_csv, labeled_data)

        self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        print('[dataset] X-ray classification set=%s number of classes=%d  number of images=%d' %
              (set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        print(path)

        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)

# Kiểm tra số lõi CPU của hệ thống
num_cores = multiprocessing.cpu_count()
print(f"Số lõi CPU: {num_cores}")

# Sử dụng DataLoader với số lượng worker hợp lý
def get_dataloader(dataset, batch_size=32, num_workers=None):
    if num_workers is None:
        num_workers = min(8, num_cores)
    return data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
