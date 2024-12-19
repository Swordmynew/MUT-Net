# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
# torch libraries
import os
import logging
import numpy as np
from datetime import datetime
import time

from PIL import Image
from tensorboardX import SummaryWriter
from lib.SAE import SAE as Network
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import optim,nn
from torchvision.utils import make_grid
# customized libraries
import eval.python.metrics as Measure
from utils.utils import clip_gradient
from utils.dataset import get_loader, test_dataset
import torch_dct as DCT

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def involveloss(A,B):
    mask = (A > B).float()
    loss = F.mse_loss(A * mask, B * mask, reduction='mean')
    return loss

def structure_loss(pred, mask):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score

def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, (images, gts, grads, edges, image_max1, image_max2, image_max3, image_max4, image_max5, image_min1, image_min2, image_min3, image_min4, image_min5, range1, range2, range3, range4) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            image_max1 = image_max1.cuda()
            image_max2 = image_max2.cuda()
            image_max3 = image_max3.cuda()
            # image_max4 = image_max4.cuda()
            image_max5 = image_max5.cuda()
            image_min1 = image_min1.cuda()
            image_min2 = image_min2.cuda()
            image_min3 = image_min3.cuda()
            image_min5 = image_min5.cuda()
            range1 = range1.cuda()
            range2 = range2.cuda()
            range3 = range3.cuda()
            range4 = range4.cuda()
            edges = edges.cuda()
            preds = model(images)
            preds_gt = preds
            grads = grads.cuda()

            # first ablation without min and max loss
            loss_pred = (structure_loss(preds_gt[0][0], gts) + structure_loss(preds_gt[0][1], gts) + structure_loss(preds_gt[0][2], gts) + structure_loss(preds_gt[0][3], gts) + 4 * structure_loss(preds_gt[0][4], gts))/8
            # loss_max = (structure_loss(preds_gt[1][0], image_max1) + structure_loss(preds_gt[1][1], image_max2) + structure_loss(preds_gt[1][2], image_max3) + structure_loss(preds_gt[1][3], image_max5))/8
            # loss_min = (structure_loss(preds_gt[2][0], image_min1) + structure_loss(preds_gt[2][1], image_min2) + structure_loss(preds_gt[2][2], image_min3) + structure_loss(preds_gt[2][3], image_min5))/8
            # loss_involove3 = (involveloss(preds_gt[3][1],preds_gt[3][0]) + involveloss(preds_gt[3][2],preds_gt[3][1]) + involveloss(preds_gt[3][3],preds_gt[3][2]) + involveloss(edges,preds_gt[3][3]) + involveloss(edges,preds_gt[3][2]) + involveloss(edges,preds_gt[3][1]) + involveloss(edges,preds_gt[3][0]))*100/8
            loss_edge = (edge_loss_func1(preds[1][0], edges) + edge_loss_func1(preds[1][1], edges) + edge_loss_func1(preds[1][2], edges) + edge_loss_func1(preds[1][3], edges))*10
            # loss_crf = (involveloss(preds[3][0],preds[3][1])+involveloss(preds[3][1],preds[3][2])+involveloss(preds[3][2],preds[3][3]))/8*2
            # loss_grad = edge_loss_func1(preds[6], grads)
            # loss = loss_pred+loss_edge+loss_max+loss_min+loss_involove3+loss_crf
            loss = loss_pred+loss_edge
            #loss = loss_pred
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print(
                        '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_pred.data))
                logging.info(
                        '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} loss_pred: {:.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_pred.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'loss_pred': loss_pred.data,
                                    'Loss_total': loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                # grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('RGB', grid_image, step)
                # grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('GT', grid_image, step)
                # grid_image = make_grid(range1[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('GT_range1', grid_image, step)
                # grid_image = make_grid(range2[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('GT_range2', grid_image, step)
                # grid_image = make_grid(range3[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('GT_range3', grid_image, step)
                # grid_image = make_grid(range4[0].clone().cpu().data, 1, normalize=True)
                # writer.add_image('GT_range4', grid_image, step)
                # res = preds[0][4][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_final0', torch.tensor(res), step, dataformats='HW')
                # res = preds[0][3][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_final1', torch.tensor(res), step, dataformats='HW')
                # res = preds[0][2][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_final2', torch.tensor(res), step, dataformats='HW')
                # res = preds[0][1][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_final3', torch.tensor(res), step, dataformats='HW')
                # res = preds[3][0][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_edge', torch.tensor(res), step, dataformats='HW')
                # res = preds[3][1][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_edge1', torch.tensor(res), step, dataformats='HW')
                # res = preds[3][2][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_edge2', torch.tensor(res), step, dataformats='HW')
                # res = preds[3][3][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Pred_edge3', torch.tensor(res), step, dataformats='HW')
                # res = preds[4][0][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Range0', torch.tensor(res), step, dataformats='HW')
                # res = preds[4][1][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Range1', torch.tensor(res), step, dataformats='HW')
                # res = preds[4][2][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Range2', torch.tensor(res), step, dataformats='HW')
                # res = preds[4][3][0].clone()
                # res = res.sigmoid().data.cpu().numpy().squeeze()
                # res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # writer.add_image('Range3', torch.tensor(res), step, dataformats='HW')


        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
        #return loss_all
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch + 1))
        print('Save checkpoints successfully!')
        raise


def val(test_loader, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_metric_dict, best_score, best_epoch, bestSmeasure, bestwFmeasure, bestMAE, bestadpEm, bestmeanEm, bestmaxEm, bestadpFm, bestmeanFm, bestmaxFm, bestber, bestF1, bestAUC
    FM = Measure.Fmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    WFM = Measure.WeightedFmeasure()
    MAE = Measure.MAE()
    BER = Measure.BER()
    F1 = Measure.F1Measure()
    AUC = Measure.AUCMeasure()
    metrics_dict = dict()

    model.eval()
    with torch.no_grad():
        for i in range(test_loader.size):
            image, gt, _, _ = test_loader.load_data()
            #
            # ycbcr = ycbcr.cuda()
            #
            # num_batchsize = ycbcr.shape[0]
            # size = ycbcr.shape[2]
            #
            # ycbcr = ycbcr.reshape(num_batchsize, 3, size // 8, 8, size // 8, 8).permute(0, 2, 4, 1, 3, 5)
            # ycbcr = DCT.dct_2d(ycbcr, norm='ortho')
            # ycbcr = ycbcr.reshape(num_batchsize, size // 8, size // 8, -1).permute(0, 3, 1, 2)
            #
            gt = np.asarray(gt, np.float32)
            image = image.cuda()

            res = model(image)
            # print(res)
            # print(res)
            # image = res.cpu().clone().detach().numpy()
            # image = image[0].squeeze(0)
            # image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            # # image = image.squeeze(0)
            # # print(image)
            # # image = image.transpose(1, 2, 0)
            # image = Image.fromarray(np.uint8(image*255))
            # image.show()

            res = F.upsample(res[0][4], size=gt.shape, mode='bilinear', align_corners=False)
            # print(res.shape)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)

            FM.step(pred=res, gt=gt)
            SM.step(pred=res, gt=gt)
            EM.step(pred=res, gt=gt)
            MAE.step(pred=res, gt=gt)
            WFM.step(pred=res, gt=gt)
            BER.step(pred=res, gt=gt)
            F1.step(pred=res, gt=gt)
            AUC.step(pred=res, gt=gt)

        fm = FM.get_results()["fm"]
        wfm = WFM.get_results()["wfm"]
        sm = SM.get_results()["sm"]
        em = EM.get_results()["em"]
        mae = MAE.get_results()["mae"]
        ber = BER.get_results()["ber"]
        f1 = F1.get_results()["f1"]
        auc = AUC.get_results()["auc"]

        Smeasure = sm
        wFmeasure = wfm
        MAE = mae
        adpEm = em["adp"]
        meanEm = em["curve"].mean()
        maxEm = em["curve"].max()
        adpFm = fm["adp"]
        meanFm = fm["curve"].mean()
        maxFm = fm["curve"].max()
        BER = ber
        print(f1,auc,bestF1,bestAUC)
        metrics_dict.update(Sm=SM.get_results()['sm'])
        metrics_dict.update(mxFm=FM.get_results()['fm']['curve'].max().round(3))
        metrics_dict.update(mxEm=EM.get_results()['em']['curve'].max().round(3))

        cur_score = metrics_dict['Sm'] + metrics_dict['mxFm'] + metrics_dict['mxEm']

        if epoch == 1:
            best_score = cur_score
            print(
                '[Cur Epoch: {}] Metrics (Smeasure:{}; wFmeasure:{}; MAE:{}; adpEm:{}; meanEm:{}; maxEm:{}; adpFm:{}; meanFm:{}; maxFm:{}; BER:{}; F1:{}; AUC:{})'
                .format(epoch, Smeasure, wFmeasure, MAE, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm, BER, f1, auc)
            )
            logging.info(
                '[Cur Epoch: {}] Metrics (Smeasure:{}; wFmeasure:{}; MAE:{}; adpEm:{}; meanEm:{}; maxEm:{}; adpFm:{}; meanFm:{}; maxFm:{}; BER:{}; F1:{}; AUC:{})'
                .format(epoch, Smeasure, wFmeasure, MAE, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm, BER, f1, auc)
            )
        else:
            if cur_score > best_score:
                best_metric_dict = metrics_dict
                best_score = cur_score
                best_epoch = epoch

                bestSmeasure = Smeasure
                bestwFmeasure = wFmeasure
                bestMAE = MAE
                bestadpEm = adpEm
                bestmeanEm = meanEm
                bestmaxEm = maxEm
                bestadpFm = adpFm
                bestmeanFm = meanFm
                bestmaxFm = maxFm
                bestber = BER
                bestF1 = f1
                bestAUC = auc
                torch.save(model.state_dict(), save_path + 'Net_epoch_best.pth')
                torch.save(model.state_dict(), save_path + 'Net_epoch_{}.pth'.format(epoch))
                print('>>> save state_dict successfully! best epoch is {}.'.format(epoch))
            else:
                print('>>> not find the best epoch -> continue training ...')
            print(
                '[Cur Epoch: {}] Metrics (Smeasure:{}; wFmeasure:{}; MAE:{}; adpEm:{}; meanEm:{}; maxEm:{}; adpFm:{}; meanFm:{}; maxFm:{}; BER:{})\n[Best Epoch: {}] Metrics (Smeasure:{}; wFmeasure:{}; MAE:{}; adpEm:{}; meanEm:{}; maxEm:{}; adpFm:{}; meanFm:{}; maxFm:{}; BER:{}; F1:{}; AUC:{})'
                .format(epoch, Smeasure, wFmeasure, MAE, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm, BER, best_epoch,
                        bestSmeasure, bestwFmeasure, bestMAE, bestadpEm, bestmeanEm, bestmaxEm, bestadpFm, bestmeanFm,
                        bestmaxFm, bestber, bestF1, bestAUC)
            )
            logging.info(
                '[Cur Epoch: {}] Metrics (Smeasure:{}; wFmeasure:{}; MAE:{}; adpEm:{}; meanEm:{}; maxEm:{}; adpFm:{}; meanFm:{}; maxFm:{}; BER:{})\n[Best Epoch: {}] Metrics (Smeasure:{}; wFmeasure:{}; MAE:{}; adpEm:{}; meanEm:{}; maxEm:{}; adpFm:{}; meanFm:{}; maxFm:{}; BER:{}; F1:{}; AUC:{})'
                .format(epoch, Smeasure, wFmeasure, MAE, adpEm, meanEm, maxEm, adpFm, meanFm, maxFm, BER, best_epoch,
                        bestSmeasure, bestwFmeasure, bestMAE, bestadpEm, bestmeanEm, bestmaxEm, bestadpFm, bestmeanFm,
                        bestmaxFm, bestber, bestF1, bestAUC)
            )


def dice_loss(predict, target):
    smooth = 1
    p = 2
    valid_mask = torch.ones_like(target)
    predict = predict.contiguous().view(predict.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)
    valid_mask = valid_mask.contiguous().view(valid_mask.shape[0], -1)
    num = torch.sum(torch.mul(predict, target) * valid_mask, dim=1) * 2 + smooth
    den = torch.sum((predict.pow(p) + target.pow(p)) * valid_mask, dim=1) + smooth
    loss = 1 - num / den
    return loss.mean()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100000, help='epoch number')  # 100
    parser.add_argument('--lr', type=float, default=(1e-4)/2, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')  # 12
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--model', type=str, default='PVTv2-B4',
                        choices=['EF-B4', 'EF-B4', 'PVTv2-B0', 'PVTv2-B1', 'PVTv2-B2',
                                 'PVTv2-B3', 'PVTv2-B4', 'swin-b', 'swin-l', 'swin-t', 'swin-s', 'res50'])
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--train_root', type=str, default='../dataset/TrainDataset/',
                        help='the training rgb images root')
    parser.add_argument('--val_root', type=str, default='../dataset/TestDataset/COD10K/',
                        help='the test rgb images root')
    parser.add_argument('--gpu_id', type=str, default='1',
                        help='train use gpu')
    parser.add_argument('--save_path', type=str, default='snapshot/snapshot/',
                        help='the path to save model and log')
    opt = parser.parse_args()

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
    else:
        pass
    cudnn.benchmark = True

    # build the model
    if opt.model == 'EF-B4':
        model = Network(channel=64, arc='EfficientNet-B4', M=[8, 8, 8], N=[4, 8, 16])
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
        model = Network(channel=128, arc='PVTv2-B4', M=[8, 8, 8], N=[4, 8, 16])
    elif opt.model == 'swin-t':
        model = Network(channel=96, arc='SwinTransformer-Tiny', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'swin-s':
        model = Network(channel=96, arc='SwinTransformer-Small', M=[8, 8, 8], N=[4, 8, 16])
    elif opt.model == 'swin-b':
        model = Network(channel=128, arc='SwinTransformer-Base', M=[8, 8, 8], N=[4, 8, 16])
    elif opt.model == 'swin-l':
        model = Network(channel=128, arc='SwinTransformer-Large', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    elif opt.model == 'res50':
        model = Network(channel=128, arc='Res', M=[8, 8, 8], N=[4, 8, 16]).cuda()
    else:
        raise Exception("Invalid Model Symbol: {}".format(opt.model))

    #grad_loss_func = torch.nn.MSELoss()
    #class_weights = torch.tensor([10., 1.]).cuda()
    #edge_loss_func1 = torch.nn.CrossEntropyLoss(weight=class_weights)
    edge_loss_func1 = torch.nn.MSELoss()
    edge_loss_func = DiceLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #from thop import profile
    #input = torch.randn(1, 1, 3, 352, 352).cuda()
    #flops, params = profile(model.cuda(), inputs = (input))
    #t_all = []

    #for i in range(100):
    #    t1 = time.time()
    #    input1 = torch.randn(1, 3, 352, 352).cuda()
    #    y = model(input1)
    #    t2 = time.time()
    #    t_all.append(t2 - t1)

    #print('average time:', np.mean(t_all) / 1)
    #print('average fps:',1 / np.mean(t_all))

    #print(flops/1000000000)
    #print(get_parameter_number(model))
    if opt.load is not None:
      
        pretrained_dict = torch.load(opt.load)
        print('load model from ', opt.load)
    
        model_dict = model.state_dict()
    
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
    
        model.load_state_dict(model_dict, strict=False)

    model = nn.DataParallel(model, device_ids=[0,1])
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=opt.train_root + 'Imgs/',
                              gt_root=opt.train_root + 'GT/',
                              grad_root=opt.train_root + 'frequence-foreground/',
                              edge_root=opt.train_root + 'Edge/',
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=8)
                              #num_workers = 0)

    val_loader = test_dataset(image_root=opt.val_root + 'Imgs/',
                              gt_root=opt.val_root + 'GT/',
                              testsize=opt.trainsize)
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=save_path + 'log.log',
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info(">>> current mode: network-train/val")
    logging.info('>>> config: {}'.format(opt))
    print('>>> config: : {}'.format(opt))

    step = 0
    writer = SummaryWriter(save_path + 'summary')

    best_score = 0
    best_epoch = 0

    bestSmeasure = 0
    bestwFmeasure = 0
    bestMAE = 100
    bestadpEm = 0
    bestmeanEm = 0
    bestmaxEm = 0
    bestadpFm = 0
    bestmeanFm = 0
    bestmaxFm = 0
    bestber = 100
    bestF1 = 0
    bestAUC = 0

    ln_schedule = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
    # cosine_schedule = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',factor=0.1,patience=10,verbose=False,min_lr=1e-08)
    cosine_schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=200, eta_min=(1e-5)/2)
    print(">>> start train...")
    #loss = 100000
    for epoch in range(1, opt.epoch):
        # schedule
        if epoch<=60000:
            cosine_schedule.step()
        else:
            ln_schedule.step()
        cosine_schedule.step()
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
        logging.info('>>> current lr: {}'.format(cosine_schedule.get_lr()[0]))
        # val(val_loader, model, epoch, save_path, writer)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loader, model, epoch, save_path, writer)
