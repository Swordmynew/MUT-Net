import os
import torch
import argparse
import numpy as np
from scipy import misc

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.hub import load_state_dict_from_url

from utils.dataset import test_dataset as EvalDataset
from lib.SAE import SAE as Network

import os

def evaluator(model, val_root, map_save_path, trainsize=352):
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)

    model.eval()

    with torch.no_grad():
        for i in range(val_loader.size):
            image, gt, name, ori_img= val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()

            outputs = model(image)
            output = F.upsample(outputs[0][4], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            misc.imsave(map_save_path + name, output)
            print('>>> saving prediction at: {}'.format(map_save_path + name))
            
            
            #print(outputs[2][3])
            
         
            output = F.upsample(outputs[2][3], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            misc.imsave("/home/kangkejun/SAE_bi/result/snapshot/feature/" + name.replace('.png','') + '_3.png', output)
            print('>>> saving prediction at: {}'.format("/home/user_kkj/SAE_bi/result/snapshot/feature/" + name + "_3"))
            
        
            output = F.upsample(outputs[2][2], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            misc.imsave("/home/kangkejun/SAE_bi/result/snapshot/feature/" + name.replace('.png','') + '_2.png', output)
            print('>>> saving prediction at: {}'.format("/home/user_kkj/SAE_bi/result/snapshot/feature/" + name + "_2"))
            
         
            output = F.upsample(outputs[2][1], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            misc.imsave("/home/kangkejun/SAE_bi/result/snapshot/feature/" + name.replace('.png','') + '_1.png', output)
            print('>>> saving prediction at: {}'.format("/home/user_kkj/SAE/result/snapshot/feature/" + name + "_1"))
            
       
            output = F.upsample(outputs[2][0], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            misc.imsave("/home/kangkejun/SAE_bi/result/snapshot/feature/" + name.replace('.png','') + '_0.png', output)
            print('>>> saving prediction at: {}'.format("/home/user_kkj/SAE_bi/result/snapshot/feature/" + name + "_0"))
        
            if True:
                print(outputs[1][3].shape,outputs[2][3].shape)
                output = F.upsample(outputs[3][3], size=gt.shape, mode='bilinear', align_corners=False)
                #output = output.sum(dim=1).squeeze(1)  
                output = output.sigmoid().data.cpu().numpy().squeeze()
                output = (output - output.min()) / (output.max() - output.min() + 1e-8)
                misc.imsave("/home/kangkejun/SAE_bi/result/snapshot/edge_save/" + name.replace('.png','') +'_4.png', output)
                print('>>> saving prediction at: {}'.format("/home/user_kkj/SAE_bi/result/snapshot/edge_save/" + name + "_3"))

                output = F.upsample(outputs[3][2], size=gt.shape, mode='bilinear', align_corners=False)
                #output = output.sum(dim=1).squeeze(1)  
                output = output.sigmoid().data.cpu().numpy().squeeze()
                output = (output - output.min()) / (output.max() - output.min() + 1e-8)
                misc.imsave("/home/kangkejun/SAE_bi/result/snapshot/edge_save/" + name.replace('.png','') + '_3.png', output)
                print('>>> saving prediction at: {}'.format("/home/user_kkj/SAE_bi/result/snapshot/edge_save/" + name + "_2"))
     
                output = F.upsample(outputs[3][1], size=gt.shape, mode='bilinear', align_corners=False)
                #output = output.sum(dim=1).squeeze(1)  
                output = output.sigmoid().data.cpu().numpy().squeeze()
                output = (output - output.min()) / (output.max() - output.min() + 1e-8)
                misc.imsave("/home/kangkejun/SAE_bi/result/snapshot/edge_save/" + name.replace('.png','') + '_2.png', output)
                print('>>> saving prediction at: {}'.format("/home/user_kkj/SAE_bi/result/snapshot/edge_save/" + name + "_1"))
               
                output = F.upsample(outputs[3][0], size=gt.shape, mode='bilinear', align_corners=False)
                #output = output.sum(dim).squeeze(1)  
                output = output.sigmoid().data.cpu().numpy().squeeze()
                output = (output - output.min()) / (output.max() - output.min() + 1e-8)
                misc.imsave("/home/kangkejun/SAE_bi/result/snapshot/edge_save/" + name.replace('.png','') + '_1.png', output)
                print('>>> saving prediction at: {}'.format("/home/user_kkj/SAE_bi/result/snapshot/edge_save/" + name + "_0"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='PVTv2-B4',
                        choices=['EF-B4', 'EF-B4', 'PVTv2-B0', 'PVTv2-B1', 'PVTv2-B2',
                                 'PVTv2-B3', 'PVTv2-B4', 'swin-b', 'swin-l', 'swin-t', 'swin-s', 'res50'])
    parser.add_argument('--snap_path', type=str, default='./snapshot/DGNet/Net_epoch_best.pth',
                        help='train use gpu')
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='train use gpu')
    opt = parser.parse_args()

    txt_save_path = './result/{}/'.format(opt.snap_path.split('/')[-2])
    os.makedirs(txt_save_path, exist_ok=True)

    print('>>> configs:', opt)

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    elif opt.gpu_id == '2':
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        print('USE GPU 2')
    elif opt.gpu_id == '3':
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"
        print('USE GPU 3')

    cudnn.benchmark = True

    if opt.model == 'EF-B4':
        model = Network(channel=64, arc='EfficientNet-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'EF-B1':
        model = Network(channel=32, arc='EfficientNet-B1', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'PVTv2-B0':
        model = Network(channel=32, arc='PVTv2-B0', M=[8, 8, 8], N=[8, 16, 32]).cuda()
    elif opt.model == 'PVTv2-B1':
        model = Network(channel=64, arc='PVTv2-B1', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'PVTv2-B2':
        model = Network(channel=128, arc='PVTv2-B2', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'PVTv2-B3':
        model = Network(channel=128, arc='PVTv2-B3', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'PVTv2-B4':
        model = Network(channel=64, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'swin-t':
        model = Network(channel=96, arc='SwinTransformer-Tiny', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'swin-s':
        model = Network(channel=96, arc='SwinTransformer-Small', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'swin-b':
        model = Network(channel=128, arc='SwinTransformer-Base', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'swin-l':
        model = Network(channel=128, arc='SwinTransformer-Large', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'res50':
        model = Network(channel=256, arc='Res', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    else:
        raise Exception("Invalid Model Symbol: {}".format(opt.model))
    
    
    model = torch.nn.DataParallel(model)
    model.cuda()
    
    # TODO: remove FC layers from snapshots
    model.load_state_dict(torch.load(opt.snap_path), strict=False)
    model.eval()
    COD = ['CAMO', 'COD10K', 'NC4K', 'CHAMELEON']
    SOD = ['PASCAL-S', 'HKU-IS', 'ECSSD', 'DUTS-TE', 'DUT-OMRON', 'DAVIS-S']
    shadow = ['SBU-Test', 'ISTD_Test']
    defocus = ['CUHK', 'DUT']
    fore = ['AIM-500','AM-2K']
    remote = ['EORSSD']
    polyp = ['CVC-300','CVC-ClinicDB','CVC-ColonDB','ETIS-LaribPolypDB','Kvasir']
    transp =['easy','hard']
    mirror1 = ["MSD"]
    glass = ["GDD"]
    COD_train = ["TrainDataset"]
    mode = COD
    
    for data_name in mode:
        map_save_path = txt_save_path + "{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        if mode == COD:
            evaluator(model=model,
            val_root='/data1/kangkejun/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
        elif mode == SOD:
            evaluator(model=model,
            val_root='/data1/kangkejun/SOD/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
        elif mode == shadow:
            evaluator(model=model,
            val_root='/data1/kangkejun/shadow/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
        elif mode == defocus:
            evaluator(model=model,
            val_root='/data1/kangkejun/defocus/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
        elif mode == fore:
            evaluator(model=model,
            val_root='/data1/kangkejun/fore_ground/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
        elif mode == remote:
            evaluator(model=model,
            val_root='/data1/kangkejun/remote/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
        elif mode == polyp:
             evaluator(model=model,
             val_root='/data1/kangkejun/polyp/' + data_name + '/',
             map_save_path=map_save_path,
            trainsize=352)
        elif mode == transp:
             evaluator(model=model,
             val_root='/home/user_kkj/data1/kangkejun/mirror/Testset/' + data_name + '/',
             map_save_path=map_save_path,
            trainsize=352)
        elif mode == mirror1:
             evaluator(model=model,
             val_root='/home/user_kkj/data1/kangkejun/mirror1/' + data_name + '/',
             map_save_path=map_save_path,
            trainsize=352)
        elif mode == glass:
             evaluator(model=model,
             val_root='/home/user_kkj/data1/kangkejun/glass/' + data_name + '/',
             map_save_path=map_save_path,
            trainsize=352)
        elif mode == COD_train:
             evaluator(model=model,
             val_root='/data1/kangkejun/' + data_name + '/',
             map_save_path=map_save_path,
            trainsize=352)