import yaml
import numpy as np
from PIL import Image
import torch
from ptsemseg.models.hardnet import hardnet


## LOAD PRETRAIN AND SAVE
# with open('configs/hardnet.yml') as fp:
#     cfg = yaml.load(fp)

# model = get_model(cfg["model"], n_classes).to(device)
# model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
# weights = torch.load('weights/hardnet70_cityscapes_model.pkl')
# # weights = torch.load('weights/hardnet_petite_base.pth')
# model.load_state_dict(weights['model_state'])

# torch.save(model, 'bloop.pth')


## OUTPUT TO TEXT
# model = torch.load('weights/hardnet_petite_base.pth')
# model = torch.load('weights/hardnet70_cityscapes_model.pkl')
# output = []
# for i in model['model_state']:
#     output.append( i+':'+ str(np.shape(model['model_state'][i])) )
# with open('state_dict_cityscapes.txt', 'w') as f:
#     f.write( '\n'.join(output) )


# LOAD AND INFER
CITYSCAPES_COLORMAP = np.array( [
    [128, 64,128],  #road
    [244, 35,232],  #sidewalk
    [ 70, 70, 70],  #building
    [102,102,156],  #wall
    [190,153,153],  #fence
    [153,153,153],  #pole
    [250,170, 30],  #traffic light
    [220,220,  0],  #traffic sign
    [107,142, 35],  #vegetation
    [152,251,152],  #terrain
    [ 70,130,180],  #sky
    [220, 20, 60],
    [255,  0,  0],
    [  0,  0,142],
    [  0,  0, 70],
    [  0, 60,100],
    [  0, 80,100],
    [  0,  0,230],
    [119, 11, 32]
    
    ], dtype = np.uint8)

def process_img(img_path):
    img = Image.open(img_path).resize((2048,1024))
    img = np.array(img, dtype=np.float64)
    img = img[:, :, ::-1]  # RGB -> BGR
    
    mean = [0.406, 0.456, 0.485]
    mean = [item * 255 for item in mean]
    std = [0.225, 0.224, 0.229]
    std = [item * 255 for item in std]
    img = (img - mean)/std

    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float()

    return(img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = hardnet(n_classes=19).to(device)
model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
weights = torch.load('weights/hardnet70_cityscapes_model.pkl')
model.load_state_dict(weights['model_state'])
print('model is loaded')

img = process_img('bloop.jpg')

outputs = model(img.unsqueeze(0))
pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=0)

print(pred)
print(np.shape(pred))
vis = CITYSCAPES_COLORMAP[pred]
vis = Image.fromarray(vis)
vis.save('bloop_mask.png')