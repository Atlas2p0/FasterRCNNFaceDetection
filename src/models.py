import torch.nn as nn
import config
from preprocessing import generate_anchor_boxes, filter_valid_bboxes, calculate_iou, assign_targets_to_anchors, sample_minibatch, create_bbox_deltas, apply_deltas_to_boxes, clamp_boxes_to_img_boundary, filter_proposals
import torch
import torch.nn.functional as F
from torchvision.ops import roi_align
class RPN(nn.Module):
    def __init__(self, in_channels= 512,
                 num_anchors_per_location= config.NUM_ANCHORS_PER_LOC,
                 pre_nms_topk= 2000,
                 post_nms_topk= 1000,
                 scales= config.ANCHOR_SCALES_2,
                 aspect_ratios= config.ANCHOR_RATIOS,
                 nms_threshold= 0.7):  # 5 scales × 3 ratios
        super(RPN, self).__init__()
        self.num_anchors_per_loc= num_anchors_per_location
        self.rpn_conv= nn.Conv2d(in_channels,
                             in_channels, kernel_size= 3,
                             stride= 1, padding= 1)
        self.cls_layer= nn.Conv2d(in_channels,
                            num_anchors_per_location,
                            kernel_size= 1,
                            stride= 1)
        self.reg_layer= nn.Conv2d(in_channels,
                            self.num_anchors_per_loc * 4,
                            kernel_size= 1,
                            stride= 1)
        self.relu= nn.ReLU(inplace= True)

        for layer in [self.rpn_conv, self.cls_layer, self.reg_layer]:
            torch.nn.init.normal_(layer.weight, std=0.01)
            torch.nn.init.constant_(layer.bias, 0)

        # Hyperparams for proposal generation
        self.pre_nms_topk= pre_nms_topk
        self.post_nms_topk= post_nms_topk
        self.nms_threshold= nms_threshold
        self.scales= scales
        self.aspect_ratios= aspect_ratios
        self._anchor_cache= {}
    def _generate_anchors(self, features):
        B, _, H, W= features.shape
        key= (H, W)
        if key not in self._anchor_cache:
            anchors= generate_anchor_boxes(
                image_shape= (config.IMAGE_SIZE_RESHAPED[1], config.IMAGE_SIZE_RESHAPED[0]),
                features= features[0:1],
                anchor_scales= self.scales,
                anchor_ratios= self.aspect_ratios
            ).squeeze(0)
            self._anchor_cache[key]= anchors
        anchors= self._anchor_cache[key]
        return [anchors] * B

    def forward(self, feat, image_shapes, gt_boxes= None):

        B, C, H, W= feat.shape
        device= feat.device

        # 1. Shared conv heads
        rpn_features= self.relu(self.rpn_conv(feat)) # (B, C, H, W)
        cls_logits= self.cls_layer(rpn_features) # (B, A_per_loc, H, W)
        reg_deltas= self.reg_layer(rpn_features) # (B, A_per_loc * 4, H, W)

        cls_logits= cls_logits.permute(0, 2, 3, 1).contiguous().view(B, -1)#(A, )
        reg_deltas= reg_deltas.permute(0, 2, 3, 1).contiguous().view(B, -1, 4) #(A, 4)

        # 2. Anchors (same for every image in batch)
        anchors_list= self._generate_anchors(feat)

        # 3. Loop over batch
        proposals_list, scores_list= [], []
        cls_list, reg_list, anchor_list= [], [], []

        # Training tensors
        sampled_cls_list, sampled_reg_list, sampled_labels_list, sampled_reg_targets_list= [], [], [], []

        for b in range(B):
            anchors= anchors_list[b].to(device)
            proposals= apply_deltas_to_boxes(anchors, reg_deltas[b])
            proposals= clamp_boxes_to_img_boundary(proposals, image_shapes[b])

            proposals, scores= filter_proposals(
                proposals,
                cls_logits[b],
                image_shapes[b],
                pre_nms_topk= self.pre_nms_topk,
                post_nms_topk= self.post_nms_topk,
                nms_thresh= self.nms_threshold
            )

            proposals_list.append(proposals)
            scores_list.append(scores)
            cls_list.append(cls_logits[b])
            reg_list.append(reg_deltas[b])
            anchor_list.append(anchors)

            if proposals.numel() == 0:
                continue

            # training targets
            if gt_boxes is not None:
                gt= filter_valid_bboxes(gt_boxes[b].to(device))
                if gt.numel() == 0:
                    continue

                labels, matched_gt_idx= assign_targets_to_anchors(anchors, gt)
                keep= sample_minibatch(labels)
                if keep.numel() == 0:
                    continue

                filtered_labels= labels[keep]
                reg_targets= create_bbox_deltas(anchors[keep], gt[matched_gt_idx[keep]])

                sampled_cls_list.append(cls_logits[b][keep])
                sampled_reg_list.append(reg_deltas[b][keep])
                sampled_labels_list.append(filtered_labels)
                sampled_reg_targets_list.append(reg_targets)

        if gt_boxes is not None and len(sampled_cls_list) == 0:
            zero= torch.tensor(0.0, device= device, requires_grad= True)
            return {
                "sampled_cls_logits": zero,
                "sampled_labels": zero,
                "sampled_reg_deltas": zero,
                "sampled_reg_targets": zero}

        out= {
            "proposals": proposals_list,
            "scores": scores_list,
            "cls_logits": cls_list,
            "reg_deltas": reg_list,
            "anchors": anchor_list,
        }

        if gt_boxes is not None and len(sampled_cls_list) > 0:
            out.update({
                "sampled_cls_logits": torch.cat(sampled_cls_list, 0),
                "sampled_reg_deltas": torch.cat(sampled_reg_list, 0),
                "sampled_labels": torch.cat(sampled_labels_list, 0),
                "sampled_reg_targets": torch.cat(sampled_reg_targets_list, 0),
            })
        return out
    
class RoIHead(nn.Module):
    def __init__(self,
                 in_channels= 512,
                 num_classes= 1,
                 roi_output_size= 14,
                 fc_hidden= 2048,
                 dropout_rate= 0.5):
        super(RoIHead, self).__init__()
        self.num_classes= num_classes
        self.roi_output_size= roi_output_size

        self.fc= nn.Sequential(
            nn.AdaptiveAvgPool2d(roi_output_size),
            nn.Flatten(),
            nn.Linear(in_channels * roi_output_size * roi_output_size, fc_hidden),
            nn.ReLU(inplace= True),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, fc_hidden),
            nn.ReLU(inplace= True),
            nn.Dropout(dropout_rate)
        )
        self.cls_head= nn.Linear(fc_hidden, num_classes + 1)
        self.reg_head= nn.Linear(fc_hidden, (num_classes + 1) * 4)

        for m in [self.cls_head, self.reg_head]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.cls_head.weight, std= 0.01)
        nn.init.normal_(self.reg_head.weight, std= 0.001)
    def _empty_forward(self, device):
        return {
            'cls_logits': torch.empty(0, self.num_classes + 1, device=device),
            'bbox_deltas': torch.empty(0, (self.num_classes + 1) * 4, device=device),
            'cls_loss': torch.tensor(0.0, device=device, requires_grad=True),
            'reg_loss': torch.tensor(0.0, device=device, requires_grad=True)
        }
    def forward(self, features, proposals, image_shapes, gt_boxes= None):
        device= features.device
        B, C, feat_H, feat_W= features.shape
        img_H, img_W= image_shapes[0]
        spatial_scale= feat_H / img_H

        if gt_boxes is not None:
            sampled_proposals, sampled_labels, sampled_targets= self.sample_rois_for_training(proposals, gt_boxes)
            if len(sampled_proposals) == 0:
                return self._empty_forward(device)


            rois= []
            roi_batch_idx= []
            # 1. Convert proposals to RoI-Align format
            for b, prop in enumerate(sampled_proposals):
                rois.append(prop)
                roi_batch_idx.append(torch.full((prop.shape[0],), b, device= device, dtype= torch.float))

        else:
            rois= []
            roi_batch_idx= []
            for b, prop in enumerate(proposals):
                if prop.numel() > 0:
                    rois.append(prop)
                    roi_batch_idx.append(torch.full((prop.shape[0],), b, device= device, dtype= torch.float))

        if len(rois) == 0:
            return self._empty_forward(device)

        rois= torch.cat(rois, dim= 0)
        batch_idx= torch.cat(roi_batch_idx)
        # 3. RoI-Align
        roi_features= roi_align(
            features,
            torch.cat([batch_idx.unsqueeze(1), rois], dim= 1),
            output_size= self.roi_output_size,
            spatial_scale= spatial_scale,
            sampling_ratio= -1
        )

        x= self.fc(roi_features)
        cls_logits= self.cls_head(x)
        bbox_deltas= self.reg_head(x)

        losses= {}
        if gt_boxes is not None:
            cls_loss, reg_loss= self.compute_losses(cls_logits, bbox_deltas, sampled_labels, sampled_targets)
            losses= dict(cls_loss= cls_loss, reg_loss= reg_loss)

        return dict(cls_logits= cls_logits,
                    bbox_deltas= bbox_deltas,
                    **losses)

    def compute_losses(self, cls_logits, bbox_deltas, sampled_labels, sampled_targets):
        device= cls_logits.device

        if len(sampled_labels) == 0:
            return torch.tensor(0.0, device= device, requires_grad= True), torch.tensor(0.0, device= device, requires_grad= True)

        labels= torch.cat(sampled_labels)
        targets= torch.cat(sampled_targets)

        cls_loss= F.cross_entropy(cls_logits, labels, label_smoothing= 0.1)

        pos_mask= labels == 1

        if pos_mask.sum() > 0:
            pos_bbox_deltas= bbox_deltas[pos_mask]
            pos_targets= targets[pos_mask]

            pos_bbox_deltas= pos_bbox_deltas.view(-1, self.num_classes + 1, 4)[:, 1, :]

            reg_loss= F.smooth_l1_loss(pos_bbox_deltas, pos_targets, beta= 1.0)
        else:
            reg_loss= torch.tensor(0.0, device= device, requires_grad= True)
        return cls_loss, reg_loss

    def sample_rois_for_training(self, proposals, gt_boxes,
                                 batch_size_per_image= 512,
                                 positive_fraction= 0.4,
                                 fg_iou_thresh= 0.7,
                                 bg_iou_thresh_hi= 0.3,
                                 bg_iou_thresh_lo= 0.05):
        sampled_proposals= []
        sampled_labels= []
        sampled_targets= []

        for b, (props, gt) in enumerate(zip(proposals, gt_boxes)):
            valid_gt= filter_valid_bboxes(gt)
            if props.numel() == 0 or valid_gt.numel() == 0:
                continue
            iou_mat= calculate_iou(props, valid_gt)
            max_ious, gt_assignment= iou_mat.max(dim= 1)

            labels= torch.zeros(props.shape[0], dtype= torch.long, device= props.device)
            fg_mask= max_ious >= fg_iou_thresh
            labels[fg_mask]= 1

            bg_mask= (max_ious < bg_iou_thresh_hi) & (max_ious >= bg_iou_thresh_lo)
            labels[bg_mask]= 0

            ignore_mask= ~(fg_mask | bg_mask)
            labels[ignore_mask]= -1

            fg_inds= torch.where(labels == 1)[0]
            bg_inds= torch.where(labels == 0)[0]

            num_fg= min(int(batch_size_per_image * positive_fraction), len(fg_inds))
            num_bg= min(batch_size_per_image - num_fg, len(bg_inds))


            if len(fg_inds) > num_fg:
                fg_inds= fg_inds[torch.randperm(len(fg_inds))[:num_fg]]
            if len(bg_inds) > num_bg:
                bg_inds= bg_inds[torch.randperm(len(bg_inds))[:num_bg]]

            keep_inds= torch.cat([fg_inds, bg_inds])
            sampled_proposals.append(props[keep_inds])
            sampled_labels.append(labels[keep_inds])

            if len(fg_inds) > 0:
                fg_props= props[keep_inds[:len(fg_inds)]]
                matched_gt= valid_gt[gt_assignment[keep_inds[:len(fg_inds)]]]
                reg_targets= create_bbox_deltas(fg_props, matched_gt)

                full_targets= torch.zeros(len(keep_inds), 4, device= props.device)
                full_targets[:len(fg_inds)]= reg_targets
                sampled_targets.append(full_targets)
            else:
                sampled_targets.append(torch.zeros((len(keep_inds), 4), device= props.device))

        return sampled_proposals, sampled_labels, sampled_targets