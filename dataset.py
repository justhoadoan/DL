from torch.utils.data import Dataset, Dataloader 
import os
import cv2
from PIL import Image
import pickle
from torchvision.transforms import Compose, ToTensor, Resize



class AnimalDataSet(Dataset):
    def __init__(self, root, istrain, transform=None):
        if istrain:
            data_path=os.path.join(root,"train")
        else:
            data_path=os.path.join(root,"test")
            #create datapath with train and test img
        categories=["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider","squirrel"]
        self.all_image_paths=[]
        self.all_labels=[]
        for index, category in enumerate(categories):
            category_path=os.path.join(data_path,category)
            for item in os.listdir(category_path):
                image_path=os.path.join(category_path,item)
                self.all_image_paths.append(image_path)
                self.all_labels.append(index)
        self.transform=transform
    def __len__(self):
        return len(self.all_labels)
    def __getitem__(self, item):
        image_path=self.all_image_paths[item]
        image=Image.open(image_path).convert("RGB")
        if self.transform:
            image=self.transform(image)
        label=self.all_labels[item]
        return image, label
if __name__=='__main__':
    transform=Compose([
        ToTensor(),
        Resize((224,224)),
    ])       
    dataset=AnimalDataSet(root='animal',istrain=True, transform=transform)
    dataloader=Dataloader(
        dataset=dataset,
        batch_size=16,
        num_worker=4,
        drop_last=True,
        shuffle=True
    )           


        
        
