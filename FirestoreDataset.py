# from io import BytesIO

# import firebase_admin
# import requests
# import torch
# import torchvision.transforms as transforms
# from firebase_admin import credentials, firestore
# from PIL import Image
# from torch.utils.data import Dataset

# # Initialize Firebase
# cred = credentials.Certificate('../fyp-project-83298-firebase-adminsdk-omga1-3c741ce672.json')
# firebase_admin.initialize_app(cred, {
#     'storageBucket': 'fyp-project-83298.appspot.com'
# }
# db = firestore.client()
# bucket = storage.bucket()

# class FirestoreDataset(Dataset):
#     def __init__(self):
#         self.metadata = list(db.collection('products').stream())
#         self.transform = transforms.ToTensor()  # Convert images to PyTorch tensors
#         self.root_dir = 'data/training_images'

#     def __len__(self):
#         return len(self.metadata)

#     def __getitem__(self, idx):
#         meta = self.metadata[idx].to_dict()
#         product_id = meta['product_id']
#         img_path = os.path.join(self.root_dir, f'{product_id}.png')
#         image = Image.open(img_path).convert('RGB')

#         if self.transform:
#             image = self.transform(image)

#         labels = torch.tensor([meta['color_tag'], meta['complexity_tag']])
#         weight = meta['weight']


#         return image, labels, weight

