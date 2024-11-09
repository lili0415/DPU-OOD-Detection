from mmaction.apis import init_recognizer
import torch
import argparse
import tqdm
import os
import numpy as np
import torch.nn as nn
import random
from dataloader_video_flow_epic import EPICDOMAIN
import torch.nn.functional as F
from scipy import spatial

class LogitNormLoss(nn.Module):
    def __init__(self, tau=0.04):
        super(LogitNormLoss, self).__init__()
        self.tau = tau

    def forward(self, x, target):
        norms = torch.norm(x, p=2, dim=-1, keepdim=True) + 1e-7
        logit_norm = torch.div(x, norms) / self.tau
        return F.cross_entropy(logit_norm, target)

def wasserstein_distance(x, y):
    # Flatten the inputs
    x_flat = x.view(x.size(0), -1)
    y_flat = y.view(y.size(0), -1)
    
    # Compute the pairwise distances
    pairwise_distances = torch.cdist(x_flat, y_flat, p=2)
    
    # Compute the Wasserstein distance
    wasserstein_dist = torch.mean(torch.sort(pairwise_distances, dim=1).values[:, 0])
    
    return wasserstein_dist

def hellinger_distance(p, q):
    # Ensure inputs sum to 1 (probability distributions)
    p = torch.clamp(p, min=1e-6)  # Avoid division by zero
    q = torch.clamp(q, min=1e-6)  # Avoid division by zero

    # Compute Hellinger distance
    distance = torch.sqrt(torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2, dim=-1)) / torch.sqrt(torch.tensor(2.0))
    return distance.mean()

def normalized_prediction_entropy(logits):
    # Apply softmax to convert logits into probabilities
    probs = F.softmax(logits, dim=-1)
    
    # Calculate entropy
    entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1)  # Adding a small epsilon to avoid log(0)
    
    # Normalize entropy
    max_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float))
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy

def train_one_step(model, clip, labels, flow, model_flow, epoch_i):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)
    m=torch.arccos(torch.zeros(1)).item() * 2/18
    T=0.05
    beta = 0.8

    with torch.no_grad():
        audio_feat = model_flow.module.backbone.get_feature(flow)
        x_slow, x_fast = model.module.backbone.get_feature(clip) 
        v_feat = (x_slow.detach(), x_fast.detach())  
        
    v_feat = model.module.backbone.get_predict(v_feat)
    v_predict, v_emd = model.module.cls_head(v_feat)

    audio_feat = model_flow.module.backbone.get_predict(audio_feat.detach())
    f_predict, f_emd = model_flow.module.cls_head(audio_feat)

    predict = mlp_cls(v_emd, f_emd)

    if args.use_single_pred:
        loss = (criterion(predict, labels) + criterion(v_predict, labels) + criterion(f_predict, labels)) / 3
    else:
        loss = criterion(predict, labels)

    if args.use_irm:
        positive_pairs = []
        for num in range(args.bsz): 
            positive_pairs.append(labels==labels[num])
        positive_pairs = torch.stack(positive_pairs,0).to(device, non_blocking=True)

        batch_Feature_v = torch.cosine_similarity(v_emd.unsqueeze(1), v_emd.unsqueeze(0), dim=-1)
        batch_Feature_v = torch.clamp(batch_Feature_v, min=-1+(1e-7), max=1-(1e-7))
        batch_Feature_v = torch.arccos(batch_Feature_v)
        batch_Feature_v = torch.cos(torch.add(batch_Feature_v,positive_pairs * m))
        batch_Feature_v = torch.div(batch_Feature_v, T)
        batch_Feature_v = torch.exp(batch_Feature_v)

        batch_Feature_f = torch.cosine_similarity(f_emd.unsqueeze(1), f_emd.unsqueeze(0), dim=-1)
        batch_Feature_f = torch.clamp(batch_Feature_f, min=-1+(1e-7), max=1-(1e-7))
        batch_Feature_f = torch.arccos(batch_Feature_f)
        batch_Feature_f = torch.cos(torch.add(batch_Feature_f,positive_pairs * m))
        batch_Feature_f = torch.div(batch_Feature_f, T)
        batch_Feature_f = torch.exp(batch_Feature_f)

        cl_loss_v = F.normalize(batch_Feature_v,p=1,dim=0)
        cl_loss_f = F.normalize(batch_Feature_f,p=1,dim=0)

        cl_loss_v = cl_loss_v*positive_pairs
        cl_loss_f = cl_loss_f*positive_pairs

        irm_loss = []
        final_cl_loss = []
        for row in range(args.bsz):
            loss_value_v = cl_loss_v[row][cl_loss_v[row].nonzero(as_tuple=True)]
            loss_value_f = cl_loss_f[row][cl_loss_f[row].nonzero(as_tuple=True)]

            # irm_loss.append(torch.var(loss_value_v,unbiased=False))
            irm_loss.append(torch.mean(torch.Tensor([torch.var(loss_value_v,unbiased=False),torch.var(loss_value_f,unbiased=False)])))
            
            final_cl_loss.append(torch.mean(torch.Tensor([torch.sum(-torch.log(loss_value_v),dim=0), torch.sum(-torch.log(loss_value_f),dim=0)])))
        
        final_cl_loss = torch.stack(final_cl_loss,0)

        final_cl_loss = torch.mean(final_cl_loss,dim=0)

        irm_loss = torch.stack(irm_loss,0)
        final_irm_loss = torch.mean(irm_loss,dim=0)

        
        
        batch_means = []
        for class_idx in range(num_class):
            class_mask = (labels == class_idx)  # 获取该类的mask
            if class_mask.sum() > 0:  # 如果该类在当前batch中存在
                class_mean = v_emd[class_mask].mean(dim=0)  # 计算该类的v_emd均值
                batch_means.append(class_mean)
            else:
                batch_means.append(prototypes[class_idx])  # 如果该类不存在，使用prototype
        batch_means = torch.stack(batch_means)

        for class_idx in range(num_class):
            class_mask = (labels == class_idx)
            if class_mask.sum() > 0:
                class_var = v_emd[class_mask].var(dim=0, unbiased=False)  # 计算该类的方差
                update_speed = torch.clamp(1.0 / (class_var + 1e-6), min=0.01, max=0.5)  # 动态更新速度
                update_speed = update_speed / class_mask.sum()  # 除以该类样本的数量
                with torch.no_grad():
                    # 更新prototypes，使用batch_means和prototypes的差作为更新方向
                    prototypes[class_idx] = beta * prototypes[class_idx] + (1 - beta) * (batch_means[class_idx] - prototypes[class_idx]) * update_speed

        # for class_idx in range(num_class):
        #     class_mask = (labels == class_idx)
        #     if class_mask.sum() > 0:
        #         class_var = v_emd[class_mask].var(dim=0, unbiased=False)  # 计算该类的方差
        #         update_speed = torch.clamp(1.0 / (class_var + 1e-6), min=0.01, max=0.2)  # 动态更新速度
        #         with torch.no_grad():
        #             prototypes[class_idx] = prototypes[class_idx] * (1 - update_speed) + batch_means[class_idx] * update_speed
        prototype_similarity = torch.cosine_similarity(v_emd, prototypes[labels], dim=-1)
        contrastive_loss = -torch.log(prototype_similarity).mean()
        loss = loss + 0.2 * final_cl_loss + 2 * final_irm_loss + 0.1 * contrastive_loss

    if args.use_dynamic_a2d:
        predicted_v_without_gt = []
        predicted_a_without_gt = []

        for i in range(len(v_predict)):
            label = labels[i].item()
            predicted_array_without_gt_v = torch.cat([v_predict[i, :label], v_predict[i, (label + 1):]], dim=0)
            predicted_v_without_gt.append(predicted_array_without_gt_v.unsqueeze(0))

            predicted_array_without_gt_a = torch.cat([f_predict[i, :label], f_predict[i, (label + 1):]], dim=0)
            predicted_a_without_gt.append(predicted_array_without_gt_a.unsqueeze(0))

        predicted_v_without_gt = torch.cat(predicted_v_without_gt, dim=0)
        predicted_a_without_gt = torch.cat(predicted_a_without_gt, dim=0)

        softmax_v = F.softmax(predicted_v_without_gt, dim=1)
        softmax_a = F.softmax(predicted_a_without_gt, dim=1)

        # 初始化 A2D 损失
        a2d_loss = 0

        # 对每个样本计算其 v_emd 和 prototype 的距离，并动态调整 a2d_ratio
        for i in range(len(v_emd)):
            label = labels[i].item()
            prototype = prototypes[label]

             # 如果 epoch_i 小于 2，使用固定的 a2d_ratio
            if epoch_i < 2:
                a2d_ratio = args.a2d_ratio  # 固定值，比如 args.a2d_fixed_ratio = 0.5
            else:
                # 从第 3 个 epoch 开始，计算 v_emd 和 prototype 的距离并动态调整 a2d_ratio
                distance = torch.cosine_similarity(v_emd[i], prototype, dim=-1)  # 计算余弦相似度
                distance = torch.clamp(distance, min=-1 + 1e-7, max=1 - 1e-7)  # 避免数值不稳定
                a2d_ratio = (1.0 - torch.sigmoid(distance)) * 3.2  # 使用 1-sigmoid(distance)，0.3是超参数，建议是a2d_ratio的两倍左右

            # 计算 A2D 损失
            if args.a2d_max_l1:
                a2d_loss += a2d_ratio * -F.l1_loss(softmax_v[i], softmax_a[i])
            elif args.a2d_max_l2:
                a2d_loss += a2d_ratio * -F.mse_loss(softmax_v[i], softmax_a[i])
            elif args.a2d_max_hellinger:
                a2d_loss += a2d_ratio * -hellinger_distance(softmax_v[i], softmax_a[i])
            elif args.a2d_max_wasserstein:
                a2d_loss1 = -wasserstein_distance(softmax_v[i], softmax_a[i])
                a2d_loss2 = -wasserstein_distance(softmax_a[i], softmax_v[i])
                a2d_loss += a2d_ratio * (a2d_loss1 + a2d_loss2) / 2


        # 将 A2D 损失平均化，并加权到总损失中
        loss = loss + a2d_loss / len(v_emd)

    if args.use_a2d:
        predicted_v_without_gt = []
        predicted_a_without_gt = []
        for i in range(len(v_predict)):
            label = labels[i].item()
            predicted_array_without_gt_v = torch.cat([v_predict[i, :label], v_predict[i, (label+1):]], dim=0)
            predicted_v_without_gt.append(predicted_array_without_gt_v.unsqueeze(0))
            predicted_array_without_gt_a = torch.cat([f_predict[i, :label], f_predict[i, (label+1):]], dim=0)
            predicted_a_without_gt.append(predicted_array_without_gt_a.unsqueeze(0))

        predicted_v_without_gt = torch.cat(predicted_v_without_gt, dim=0)
        predicted_a_without_gt = torch.cat(predicted_a_without_gt, dim=0)

        if args.a2d_max_l1:
            a2d_loss = -nn.L1Loss()(nn.Softmax(dim=1)(predicted_v_without_gt), nn.Softmax(dim=1)(predicted_a_without_gt))
        elif args.a2d_max_l2:
            a2d_loss = -nn.MSELoss()(nn.Softmax(dim=1)(predicted_v_without_gt), nn.Softmax(dim=1)(predicted_a_without_gt))
        elif args.a2d_max_hellinger:
            a2d_loss = -hellinger_distance(F.softmax(predicted_v_without_gt, dim=1), F.softmax(predicted_a_without_gt, dim=1))
        elif args.a2d_max_wasserstein:
            a2d_loss1 = -wasserstein_distance(F.softmax(predicted_v_without_gt, dim=1), F.softmax(predicted_a_without_gt, dim=1))
            a2d_loss2 = -wasserstein_distance(F.softmax(predicted_a_without_gt, dim=1), F.softmax(predicted_v_without_gt, dim=1))
            a2d_loss = (a2d_loss1 + a2d_loss2) / 2
        
        loss = loss + args.a2d_ratio * a2d_loss

    if args.use_npmix:
        output = torch.cat((v_emd, f_emd), dim=1)
        sum_temp = 0
        for index in range(num_class):
            sum_temp += number_dict[index]
        a2d_loss_ood = torch.zeros(1).cuda()[0]
        ood_entropy_loss = torch.zeros(1).cuda()[0]
        if (sum_temp == num_class * args.sample_number) and (epoch_i < args.start_epoch):
            target_numpy = labels.cpu().data.numpy()
            for index in range(len(labels)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat(
                    (data_dict[dict_key][1:],
                        output[index].detach().view(1, -1)), 0)


        elif (sum_temp == num_class * args.sample_number) and (epoch_i >= args.start_epoch):
            target_numpy = labels.cpu().data.numpy()
            for index in range(len(labels)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat(
                    (data_dict[dict_key][1:],
                        output[index].detach().view(1, -1)), 0)

            for index in range(num_class):
                rows_with_label_i = data_dict[index]
                id_feature_proto[index] = torch.mean(rows_with_label_i, dim=0)

            tree = spatial.KDTree(id_feature_proto.cpu())

            for index in range(num_class):
                rows_with_label_i = data_dict[index]
                id_feature_proto_i = id_feature_proto[index]
                dis, ind = tree.query(id_feature_proto_i.cpu(), k=args.nn_k)
                ind = ind[1:]
                index1 = np.random.choice(rows_with_label_i.shape[0], 1)
                index_nn = np.random.choice(ind)
                rows_with_label_j = data_dict[index_nn]
                index2 = np.random.choice(rows_with_label_j.shape[0], 1)
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                ood_data_sample = (lam * rows_with_label_i[index1] + (1 - lam) * rows_with_label_j[index2])

                if index == 0:
                    ood_samples = ood_data_sample
                else:
                    ood_samples = torch.cat(
                        (ood_samples, ood_data_sample), 0)

            if len(ood_samples) != 0:
                v_pred_ood = model.module.cls_head.fc_cls(ood_samples[:,:v_dim])
                a_pred_ood = model_flow.module.cls_head.fc_cls(ood_samples[:,v_dim:])
                if args.max_ood_l1:
                    a2d_loss_ood = -nn.L1Loss()(nn.Softmax(dim=1)(v_pred_ood), nn.Softmax(dim=1)(a_pred_ood))
                elif args.max_ood_l2:
                    a2d_loss_ood = -nn.MSELoss()(nn.Softmax(dim=1)(v_pred_ood), nn.Softmax(dim=1)(a_pred_ood))
                elif args.max_ood_hellinger:
                    a2d_loss_ood = -hellinger_distance(F.softmax(v_pred_ood, dim=1), F.softmax(a_pred_ood, dim=1))
                elif args.max_ood_wasserstein:
                    a2d_loss_ood1 = -wasserstein_distance(F.softmax(v_pred_ood, dim=1), F.softmax(a_pred_ood, dim=1))
                    a2d_loss_ood2 = -wasserstein_distance(F.softmax(a_pred_ood, dim=1), F.softmax(v_pred_ood, dim=1))
                    a2d_loss_ood = (a2d_loss_ood1 + a2d_loss_ood2) / 2

                # max_ood_entropy
                v_pred_ood_ent = normalized_prediction_entropy(v_pred_ood)
                a_pred_ood_ent = normalized_prediction_entropy(a_pred_ood)
                ood_entropy_loss = -(torch.mean(v_pred_ood_ent) + torch.mean(a_pred_ood_ent)) / 2

        else:
            target_numpy = labels.cpu().data.numpy()
            for index in range(len(labels)):
                dict_key = target_numpy[index]

                if number_dict[dict_key] < args.sample_number:
                    data_dict[dict_key][number_dict[
                        dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        loss = loss + args.ood_entropy_ratio * ood_entropy_loss + args.a2d_ratio_ood * a2d_loss_ood

    optim.zero_grad()
    loss.backward()
    optim.step()
    return predict, loss

def validate_one_step(model, clip, labels, flow, model_flow):
    clip = clip['imgs'].cuda().squeeze(1)
    labels = labels.cuda()
    flow = flow['imgs'].cuda().squeeze(1)

    with torch.no_grad():
        x_slow, x_fast = model.module.backbone.get_feature(clip) 
        v_feat = (x_slow.detach(), x_fast.detach())  

        v_feat = model.module.backbone.get_predict(v_feat)
        v_predict, v_emd = model.module.cls_head(v_feat)

        audio_feat = model_flow.module.backbone.get_feature(flow)  
        audio_feat = model_flow.module.backbone.get_predict(audio_feat)
        f_predict, f_emd = model_flow.module.cls_head(audio_feat)
       
        predict = mlp_cls(v_emd, f_emd)

    loss = criterion(predict, labels)

    return predict, loss


class Encoder(nn.Module):
    def __init__(self, input_dim=2816, out_dim=8):
        super(Encoder, self).__init__()
        self.enc_net = nn.Linear(input_dim, out_dim)
       
    def forward(self, vfeat, afeat):
        feat = torch.cat((vfeat, afeat), dim=1)
        return self.enc_net(feat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--datapath', type=str, default='/path/to/video_datasets/',
                        help='datapath')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='lr')
    parser.add_argument('--bsz', type=int, default=16,
                        help='batch_size')
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument('--save_checkpoint', action='store_true')
    parser.add_argument('--save_best', action='store_true')
    parser.add_argument("--opt", type=str, default='adam')
    parser.add_argument('--resumef', action='store_true')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--appen", type=str, default='')
    parser.add_argument('--use_single_pred', action='store_true')
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--a2d_ratio', type=float, default=0.5,
                        help='a2d_ratio')
    parser.add_argument("--sample_number", type=int, default=65)
    parser.add_argument('--ood_entropy_ratio', type=float, default=0.5,
                        help='ood_entropy_ratio')
    parser.add_argument("--start_epoch", type=int, default=10)

    parser.add_argument('--use_a2d', action='store_true')
    parser.add_argument('--use_npmix', action='store_true')
    parser.add_argument('--a2d_max_l1', action='store_true')
    parser.add_argument('--a2d_max_l2', action='store_true')
    parser.add_argument('--a2d_max_hellinger', action='store_true')
    parser.add_argument('--a2d_max_wasserstein', action='store_true')

    parser.add_argument('--max_ood_hellinger', action='store_true')
    parser.add_argument('--max_ood_wasserstein', action='store_true')
    parser.add_argument('--max_ood_l1', action='store_true')
    parser.add_argument('--max_ood_l2', action='store_true')
    parser.add_argument('--a2d_ratio_ood', type=float, default=0.5,
                        help='a2d_ratio_ood')

    parser.add_argument("--nn_k", type=int, default=3)
    parser.add_argument('--mixup_alpha', type=float, default=10.0,
                        help='mixup_alpha')

    parser.add_argument('--logit_norm_tau', type=float, default=0.04,
                        help='logit_norm_tau')
    parser.add_argument('--logit_norm', action='store_true')

    parser.add_argument("--dataset", type=str, default='EPIC')
    parser.add_argument("--use_irm", action='store_true')
    parser.add_argument("--use_dynamic_a2d", action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # init_distributed_mode(args)
    config_file = 'configs/recognition/slowfast/slowfast_r101_8x8x1_256e_kinetics400_rgb.py'
    checkpoint_file = 'pretrained_models/slowfast_r101_8x8x1_256e_kinetics400_rgb_20210218-0dd54025.pth'

    config_file_flow = 'configs/recognition/slowonly/slowonly_r50_8x8x1_256e_kinetics400_flow.py'
    checkpoint_file_flow = 'pretrained_models/slowonly_r50_8x8x1_256e_kinetics400_flow_20200704-6b384243.pth'

    # assign the desired device.
    device = 'cuda:0' # or 'cpu'
    device = torch.device(device)

    v_dim = 2304
    f_dim = 2048

    num_class = 4

    # build the model from a config file and a checkpoint file
    model = init_recognizer(config_file, checkpoint_file, device=device, use_frames=True)
    model.cls_head.fc_cls = nn.Linear(v_dim, num_class).cuda()
    cfg = model.cfg
    model = torch.nn.DataParallel(model)

    model_flow = init_recognizer(config_file_flow, checkpoint_file_flow, device=device,use_frames=True)
    model_flow.cls_head.fc_cls = nn.Linear(f_dim, num_class).cuda()
    cfg_flow = model_flow.cfg
    model_flow = torch.nn.DataParallel(model_flow)

    mlp_cls = Encoder(input_dim=v_dim+f_dim, out_dim=num_class)
    mlp_cls = mlp_cls.cuda()

    base_path = "checkpoints/"
    if not os.path.exists(base_path):
        os.mkdir(base_path)
    base_path_model = "models/"
    if not os.path.exists(base_path_model):
        os.mkdir(base_path_model)

    log_name = "log_video_flow_%s_near_ood_lr_%s_bsz_%s_%s_%s"%(str(args.dataset), str(args.lr), str(args.bsz), str(args.nepochs), args.opt)

    if args.logit_norm:
        log_name = log_name + '_logit_norm_' + str(args.logit_norm_tau)
    if args.use_single_pred:
        log_name = log_name + '_single_pred'
    
    if args.use_a2d:
        if args.a2d_max_l1:
            log_name = log_name + '_a2d_max_l1_' + str(args.a2d_ratio)
        elif args.a2d_max_l2:
            log_name = log_name + '_a2d_max_l2_' + str(args.a2d_ratio)
        elif args.a2d_max_wasserstein:
            log_name = log_name + '_a2d_max_wasserstein_' + str(args.a2d_ratio)
        elif args.a2d_max_hellinger:
            log_name = log_name + '_a2d_max_hellinger_' + str(args.a2d_ratio)

    if args.use_npmix:
        if args.max_ood_l1:
            log_name = log_name + '_max_ood_l1_' + str(args.a2d_ratio_ood)
        elif args.max_ood_l2:
            log_name = log_name + '_max_ood_l2_' + str(args.a2d_ratio_ood)
        elif args.max_ood_wasserstein:
            log_name = log_name + '_max_ood_wasserstein_' + str(args.a2d_ratio_ood)
        elif args.max_ood_hellinger:
            log_name = log_name + '_max_ood_hellinger_' + str(args.a2d_ratio_ood)

        log_name = log_name + '_entropy_' + str(args.ood_entropy_ratio)
        log_name = log_name + '_start_epoch_' + str(args.start_epoch)
        log_name = log_name + '_nn_k_' + str(args.nn_k) + '_mixup_alpha_' + str(args.mixup_alpha)

    log_name = log_name + args.appen
    log_path = base_path + log_name + '.csv'
    print(log_path)
    
    if args.logit_norm:
        criterion = LogitNormLoss(tau=args.logit_norm_tau)
    else:
        criterion = nn.CrossEntropyLoss() 

    criterion = criterion.cuda()
    batch_size = args.bsz

    params = list(model.module.backbone.fast_path.layer4.parameters()) + list(
        model.module.backbone.slow_path.layer4.parameters()) +list(model.module.cls_head.parameters())+list(model_flow.module.backbone.layer4.parameters()) +list(model_flow.module.cls_head.parameters())
    params = params + list(mlp_cls.parameters())

    if args.opt == 'adam':
        optim = torch.optim.Adam(params, lr=args.lr, weight_decay=1e-4)
    elif args.opt == 'sgd':
        optim = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)

    number_dict = {}
    for i in range(num_class):
        number_dict[i] = 0
    feature_dim = v_dim+f_dim
    data_dict = torch.zeros(num_class, args.sample_number, feature_dim).cuda()
    id_feature_proto = torch.zeros(num_class, feature_dim).cuda()

    BestLoss = float("inf")
    BestEpoch = 0
    BestAcc = 0
    BestTestAcc = 0

    if args.resumef:
        resume_file = base_path_model + log_name + '.pt'
        print("Resuming from ", resume_file)
        checkpoint = torch.load(resume_file)
        starting_epoch = checkpoint['epoch']+1
    
        BestLoss = checkpoint['BestLoss']
        BestEpoch = checkpoint['BestEpoch']
        BestAcc = checkpoint['BestAcc']
        BestTestAcc = checkpoint['BestTestAcc']

        model.load_state_dict(checkpoint['model_state_dict'])
        model_flow.load_state_dict(checkpoint['model_flow_state_dict'])
        optim.load_state_dict(checkpoint['optimizer'])
        mlp_cls.load_state_dict(checkpoint['mlp_cls_state_dict'])
    else:
        print("Training From Scratch ..." )
        starting_epoch = 0

    print("starting_epoch: ", starting_epoch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_dataset = EPICDOMAIN(split='train', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=True,
                                                   pin_memory=(device.type == "cuda"), drop_last=True)

    val_dataset = EPICDOMAIN(split='val', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)

    test_dataset = EPICDOMAIN(split='test', cfg=cfg, cfg_flow=cfg_flow, datapath=args.datapath)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=args.num_workers, shuffle=False,
                                                   pin_memory=(device.type == "cuda"), drop_last=False)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}


    prototypes = torch.zeros(num_class, v_dim).to(device) 
    with open(log_path, "a") as f:
        for epoch_i in range(starting_epoch, args.nepochs):
            print("Epoch: %02d" % epoch_i)
            for split in ['train', 'val', 'test']:
                acc = 0
                count = 0
                total_loss = 0
                print(split)
                model.train(split == 'train')
                model_flow.train(split == 'train')
                mlp_cls.train(split == 'train')
                with tqdm.tqdm(total=len(dataloaders[split])) as pbar:
                    for (i, (clip, spectrogram, labels)) in enumerate(dataloaders[split]):
                        if split=='train':
                            predict1, loss = train_one_step(model, clip, labels, spectrogram, model_flow, epoch_i)
                        else:
                            predict1, loss = validate_one_step(model, clip, labels, spectrogram, model_flow)

                        total_loss += loss.item() * batch_size
                        _, predict = torch.max(predict1.detach().cpu(), dim=1)

                        acc1 = (predict == labels).sum().item()
                        acc += int(acc1)
                        count += predict1.size()[0]
                        pbar.set_postfix_str(
                            "Average loss: {:.4f}, Current loss: {:.4f}, Accuracy: {:.4f}".format(total_loss / float(count),
                                                                                                  loss.item(),
                                                                                                  acc / float(count)))
                        pbar.update()

                    if split == 'val':
                        currentvalAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestLoss = total_loss / float(count)
                            BestEpoch = epoch_i
                            BestAcc = acc / float(count)
                            

                    if split == 'test':
                        currenttestAcc = acc / float(count)
                        if currentvalAcc >= BestAcc:
                            BestTestAcc = currenttestAcc
                            if args.save_best:
                                save = {
                                    'epoch': epoch_i,
                                    'BestLoss': BestLoss,
                                    'BestEpoch': BestEpoch,
                                    'BestAcc': BestAcc,
                                    'BestTestAcc': BestTestAcc,
                                    'model_state_dict': model.state_dict(),
                                    'model_flow_state_dict': model_flow.state_dict(),
                                    'optimizer': optim.state_dict(),
                                }
                                save['mlp_cls_state_dict'] = mlp_cls.state_dict()

                                torch.save(save, base_path_model + log_name + '_best.pt')

                        if args.save_checkpoint:
                            save = {
                                'epoch': epoch_i,
                                'BestLoss': BestLoss,
                                'BestEpoch': BestEpoch,
                                'BestAcc': BestAcc,
                                'BestTestAcc': BestTestAcc,
                                'model_state_dict': model.state_dict(),
                                'model_flow_state_dict': model_flow.state_dict(),
                                'optimizer': optim.state_dict(),
                            }
                            save['mlp_cls_state_dict'] = mlp_cls.state_dict()
                            torch.save(save, base_path_model + log_name + '.pt')
                        
                    f.write("{},{},{},{}\n".format(epoch_i, split, total_loss / float(count), acc / float(count)))
                    f.flush()

                    print('acc on epoch ', epoch_i)
                    print("{},{},{}\n".format(epoch_i, split, acc / float(count)))
                    print('BestValAcc ', BestAcc)
                    print('BestTestAcc ', BestTestAcc)
                    
                    if split == 'test':
                        f.write("CurrentBestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
                        f.flush()

        f.write("BestEpoch,{},BestLoss,{},BestValAcc,{},BestTestAcc,{} \n".format(BestEpoch, BestLoss, BestAcc, BestTestAcc))
        f.flush()

        print('BestValAcc ', BestAcc)
        print('BestTestAcc ', BestTestAcc)

    f.close()
