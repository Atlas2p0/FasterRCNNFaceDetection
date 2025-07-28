import torch.nn as nn
import config
from preprocessing import FaceDataset, generate_anchor_boxes, filter_valid_bboxes, calculate_iou, assign_targets_to_anchors, sample_minibatch, create_bbox_deltas, apply_deltas_to_boxes, clamp_boxes_to_img_boundary, filter_proposals
import torch
class RPN(nn.Module):
    def __init__(self, in_channels= 512,
                 num_anchors_per_location= config.NUM_ANCHORS_PER_LOC,
                 pre_nms_topk= 1000,
                 post_nms_topk= 300,
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

        A_per_loc= self.num_anchors_per_loc
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