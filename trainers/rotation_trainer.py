import os
import tqdm
import torch
import importlib
import numpy as np
from trainers.base_trainer import BaseTrainer
from trainers.utils.loss_utils import *
from evaluation.evaluation_metrics import *


class Trainer(BaseTrainer):
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args

        encoder_lib = importlib.import_module(cfg.models.encoder.type)
        self.encoder = encoder_lib.ImageEncoder(cfg.models.encoder)
        self.encoder.cuda()
        print("Encoder:")
        print(self.encoder)

        dn_lib = importlib.import_module(cfg.models.rotationnet.type)
        self.rotation_net = dn_lib.RotationNet(cfg.models.rotationnet)
        self.rotation_net.cuda()
        print("rotationnet:")
        print(self.rotation_net)

        dn_lib_y = importlib.import_module(cfg.models.rotationnet_y.type)
        self.rotation_net_y = dn_lib_y.RotationNet(cfg.models.rotationnet_y)
        self.rotation_net_y.cuda()
        print("rotationnet_y:")
        print(self.rotation_net_y)

        dn_lib_z = importlib.import_module(cfg.models.rotationnet_z.type)
        self.rotation_net_z = dn_lib_z.RotationNet(cfg.models.rotationnet_z)
        self.rotation_net_z.cuda()
        print("rotationnet_z:")
        print(self.rotation_net_z)

        # The optimizer
        if not (hasattr(self.cfg.trainer, "opt_enc") and
                hasattr(self.cfg.trainer, "opt_dn")):
            self.cfg.trainer.opt_enc = self.cfg.trainer.opt
            self.cfg.trainer.opt_dn = self.cfg.trainer.opt

        if getattr(self.cfg.trainer.opt_enc, "scheduler", None) is not None:
            self.opt_enc, self.scheduler_enc = get_opt(
                self.encoder.parameters(), self.cfg.trainer.opt_enc)
        else:
            self.opt_enc = get_opt(
                self.encoder.parameters(), self.cfg.trainer.opt_enc)
            self.scheduler_enc = None

        if getattr(self.cfg.trainer.opt_dn, "scheduler", None) is not None:
            self.opt_dn, self.scheduler_dn = get_opt(
                list(self.rotation_net.parameters()) + list(self.rotation_net_y.parameters()) +
                list(self.rotation_net_z.parameters()), self.cfg.trainer.opt_dn)
        else:
            self.opt_dn = get_opt(
                list(self.rotation_net.parameters()) + list(self.rotation_net_y.parameters()) +
                list(self.rotation_net_z.parameters()), self.cfg.trainer.opt_dn)
            self.scheduler_dn = None

        self.classification = getattr(self.cfg.trainer, "classification", True)
        self.pairwise_type = getattr(self.cfg.trainer, "pairwise_type", "concat")
        self.rotation_parameterization = getattr(self.cfg.trainer, "rotation_parameterization", True)

        # Prepare save directory
        os.makedirs(os.path.join(cfg.save_dir, "checkpoints"), exist_ok=True)

    def epoch_end(self, epoch, writer=None):
        if self.scheduler_dn is not None:
            self.scheduler_dn.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_dn_lr', self.scheduler_dn.get_lr()[0], epoch)
        if self.scheduler_enc is not None:
            self.scheduler_enc.step(epoch=epoch)
            if writer is not None:
                writer.add_scalar(
                    'train/opt_enc_lr', self.scheduler_enc.get_lr()[0], epoch)

    def update(self, data_full, no_update=False):
        img1 = data_full['img1'].cuda()
        img2 = data_full['img2'].cuda()
        rotation_x1 = data_full['rotation_x1']
        rotation_y1 = data_full['rotation_y1']
        rotation_x2 = data_full['rotation_x2']
        rotation_y2 = data_full['rotation_y2']
        if not no_update:
            self.encoder.train()
            self.rotation_net.train()
            self.rotation_net_y.train()
            self.rotation_net_z.train()
            self.opt_enc.zero_grad()
            self.opt_dn.zero_grad()

        batch_size = img1.size(0)
        gt_rmat = compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size)
        if self.rotation_parameterization:
            angle_x, angle_y, angle_z = compute_angle(rotation_x1, rotation_x2, rotation_y1, rotation_y2)
        else:
            angle_x, angle_y, angle_z = compute_euler_angles_from_rotation_matrices(gt_rmat)

        image_feature_map1 = self.encoder(img1)
        image_feature_map2 = self.encoder(img2)

        # pairwise operation
        if self.pairwise_type == "concat":
            pairwise_feature = torch.cat([image_feature_map1, image_feature_map2], dim=1)
        elif self.pairwise_type == "cost_volume":
            pairwise_feature = compute_correlation_volume_pairwise(image_feature_map1, image_feature_map2, num_levels=1)
        elif self.pairwise_type == "correlation_volume":
            pairwise_feature = compute_correlation_volume_pairwise(image_feature_map1, image_feature_map2, num_levels=4)

        # loss type
        if not self.classification:
            # regression loss
            out_rmat, out_rotation = self.rotation_net(pairwise_feature)
            res1 = rotation_loss_reg(out_rmat, gt_rmat)
            loss = res1['loss']
        else:
            # classification loss
            _, out_rotation_x = self.rotation_net(pairwise_feature)
            _, out_rotation_y = self.rotation_net_y(pairwise_feature)
            _, out_rotation_z = self.rotation_net_z(pairwise_feature)
            _, rotation_x = torch.topk(out_rotation_x, 1, dim=-1)
            _, rotation_y = torch.topk(out_rotation_y, 1, dim=-1)
            _, rotation_z = torch.topk(out_rotation_z, 1, dim=-1)
            loss_x = rotation_loss_class(out_rotation_x, angle_x)
            loss_y = rotation_loss_class(out_rotation_y, angle_y)
            loss_z = rotation_loss_class(out_rotation_z, angle_z)

            loss = loss_x + loss_y + loss_z
            res1 = {"loss": loss, "loss_x": loss_x, "loss_y": loss_y, "loss_z": loss_z}

        if not no_update:
            loss.backward()
            self.opt_enc.step()
            self.opt_dn.step()
        else:
            self.opt_enc.zero_grad()
            self.opt_dn.zero_grad()
        train_info = {}
        train_info.update(res1)
        train_info.update({"loss": loss})
        return train_info

    def log_train(self, train_info, train_data, writer=None,
                  step=None, epoch=None, visualize=False):
        if writer is not None:
            for k, v in train_info.items():
                if not ('loss' in k) and not ('Error' in k):
                    continue
                if step is not None:
                    writer.add_scalar('train/' + k, v, step)
                else:
                    assert epoch is not None
                    writer.add_scalar('train/' + k, v, epoch)

    def validate(self, test_loader, epoch, val_angle=False):
        print("Validation")
        out_rmat_array = None
        gt_rmat_array = None
        gt_rmat1_array = None
        out_rmat1_array = None
        all_res = {}

        with torch.no_grad():
            self.encoder.eval()
            self.rotation_net.eval()
            self.rotation_net_y.eval()
            self.rotation_net_z.eval()
            for data_full in tqdm.tqdm(test_loader):
                img1 = data_full['img1'].cuda()
                img2 = data_full['img2'].cuda()
                rotation_x1 = data_full['rotation_x1']
                rotation_y1 = data_full['rotation_y1']
                rotation_x2 = data_full['rotation_x2']
                rotation_y2 = data_full['rotation_y2']

                batch_size = img1.size(0)
                gt_rmat = compute_gt_rmat(rotation_x1, rotation_y1, rotation_x2, rotation_y2, batch_size)
                if self.rotation_parameterization:
                    angle_x, angle_y, angle_z = compute_angle(rotation_x1, rotation_x2, rotation_y1, rotation_y2)
                else:
                    angle_x, angle_y, angle_z = compute_euler_angles_from_rotation_matrices(gt_rmat)

                image_feature_map1 = self.encoder(img1)
                image_feature_map2 = self.encoder(img2)

                if self.pairwise_type == "concat":
                    pairwise_feature = torch.cat([image_feature_map1, image_feature_map2], dim=1)
                elif self.pairwise_type == "cost_volume":
                    pairwise_feature = compute_correlation_volume_pairwise(image_feature_map1, image_feature_map2,
                                                                           num_levels=1)
                elif self.pairwise_type == "correlation_volume":
                    pairwise_feature = compute_correlation_volume_pairwise(image_feature_map1, image_feature_map2,
                                                                           num_levels=4)

                if not self.classification:
                    out_rmat, _ = self.rotation_net(pairwise_feature)
                    out_rmat1 = None
                else:
                    _, out_rotation_x = self.rotation_net(pairwise_feature)
                    _, out_rotation_y = self.rotation_net_y(pairwise_feature)
                    _, out_rotation_z = self.rotation_net_z(pairwise_feature)
                    if self.rotation_parameterization:
                        out_rmat, out_rmat1 = compute_out_rmat(out_rotation_x, out_rotation_y, out_rotation_z, batch_size)
                    else:                        
                        out_rmat, out_rmat1 = compute_out_rmat_from_euler(out_rotation_x, out_rotation_y, out_rotation_z, batch_size)
                
                if gt_rmat_array is None:
                    gt_rmat_array = gt_rmat
                else:
                    gt_rmat_array = torch.cat((gt_rmat_array, gt_rmat))
                if out_rmat_array is None:
                    out_rmat_array = out_rmat
                else:
                    out_rmat_array = torch.cat((out_rmat_array, out_rmat))
                if val_angle:
                    gt_rmat1 = compute_rotation_matrix_from_viewpoint(rotation_x1, rotation_y1, batch_size).view(batch_size, 3, 3).cuda()
                    if gt_rmat1_array is None:
                        gt_rmat1_array = gt_rmat1
                    else:
                        gt_rmat1_array = torch.cat((gt_rmat1_array, gt_rmat1))
                    if out_rmat1_array is None:
                        out_rmat1_array = out_rmat1
                    else:
                        out_rmat1_array = torch.cat((out_rmat1_array, out_rmat1))

            res_error = evaluation_metric_rotation(out_rmat_array, gt_rmat_array)
            if val_angle:
                angle_error = evaluation_metric_rotation_angle(out_rmat_array, gt_rmat_array, gt_rmat1_array, out_rmat1_array)
                res_error.update(angle_error)

            # mean, median, max, std, 10deg
            for k, v in res_error.items():
                v = v.view(-1).detach().cpu().numpy()
                if k == "gt_angle" or v.size == 0:
                    continue
                mean = np.mean(v)
                median = np.median(v)
                error_max = np.max(v)
                std = np.std(v)
                count_10 = (v <= 10).sum(axis=0)
                percent_10 = np.true_divide(count_10, v.shape[0])
                all_res.update({k + '/mean': mean, k + '/median': median, k + '/max': error_max, k + '/std': std,
                                k + '/10deg': percent_10})
        print("Validation Epoch:%d " % epoch, all_res)
        return all_res

    def log_val(self, val_info, writer=None, step=None, epoch=None):
        if writer is not None:
            for k, v in val_info.items():
                if step is not None:
                    if 'vis' in k:
                        writer.add_image(k, v, step)
                    else:
                        writer.add_scalar(k, v, step)
                else:
                    if 'vis' in k:
                        writer.add_image(k, v, epoch)
                    else:
                        writer.add_scalar(k, v, epoch)

    def save(self, epoch=None, step=None, appendix=None):
        d = {
            'opt_enc': self.opt_enc.state_dict(),
            'opt_dn': self.opt_dn.state_dict(),
            'dn': self.rotation_net.state_dict(),
            'dny': self.rotation_net_y.state_dict(),
            'dnz': self.rotation_net_z.state_dict(),
            'enc': self.encoder.state_dict(),
            'epoch': epoch,
            'step': step
        }
        if appendix is not None:
            d.update(appendix)
        #save_name = "epoch_%s_iters_%s.pt" % (epoch, step)
        save_name = "epoch_%s.pt" % (epoch)
        path = os.path.join(self.cfg.save_dir, "checkpoints", save_name)
        torch.save(d, path)
        remove_name = "epoch_%s.pt" % (epoch-1)
        remove_path = os.path.join(self.cfg.save_dir, "checkpoints", remove_name)
        if os.path.exists(remove_path):
            os.remove(remove_path)

    def resume(self, path, strict=True, resume_encoder=False, test=False, **args):
        ckpt = torch.load(path)
        self.encoder.load_state_dict(ckpt['enc'], strict=strict)
        if not resume_encoder:
            self.rotation_net.load_state_dict(ckpt['dn'], strict=strict)
            self.rotation_net_y.load_state_dict(ckpt['dny'], strict=strict)
            self.rotation_net_z.load_state_dict(ckpt['dnz'], strict=strict)
            if not test:
                self.opt_enc.load_state_dict(ckpt['opt_enc'])
                self.opt_dn.load_state_dict(ckpt['opt_dn'])
            start_epoch = ckpt['epoch']
        else:
            start_epoch = 0
        return start_epoch

    def test(self, opt, *arg, **kwargs):
        raise NotImplementedError("Trainer [test] not implemented.")
