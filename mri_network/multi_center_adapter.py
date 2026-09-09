import torch
import torch.nn as nn
from mri_network.promptmrplusV2 import PromptMR
import re
import fastmri
import math
import torch.nn.functional as F
from typing import Tuple, List


class FeatureAdapter(nn.Module):
    """特征级别的适配器 - 更稳定的版本"""

    def __init__(self, center_type=None, contrast_type=None):
        super().__init__()
        self.center_type = center_type
        self.contrast_type = contrast_type

        # 轻量级的残差适配器，添加BatchNorm提高稳定性
        self.adapter = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

        # 自适应权重，初始化为较小值
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        # x: [B, H, W]
        if len(x.shape) == 3:
            x_input = x.unsqueeze(1)  # [B, 1, H, W]
        else:
            x_input = x

        # 生成适配残差
        residual = self.adapter(x_input)  # [B, 1, H, W]

        # 限制alpha范围，防止过大的调整
        alpha_clamped = torch.clamp(self.alpha, -0.1, 0.1)

        # 应用适配
        adapted = x_input + alpha_clamped * residual

        if len(x.shape) == 3:
            return adapted.squeeze(1)  # [B, H, W]
        else:
            return adapted


class HCMPathologyAdapter(nn.Module):
    """HCM pathology-specific adapter - applied after center and contrast adapters.

    Exposes intermediate features from the 2nd ReLU layer for Mahalanobis gating.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 32, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(32, 1, 3, 1, 1)
        self.tanh = nn.Tanh()
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x, return_features=False):
        if len(x.shape) == 3:
            x_input = x.unsqueeze(1)
        else:
            x_input = x

        h = self.relu1(self.bn1(self.conv1(x_input)))
        features = self.relu2(self.bn2(self.conv2(h)))  # [B, 32, H', W']
        residual = self.tanh(self.conv3(features))

        alpha_clamped = torch.clamp(self.alpha, -0.1, 0.1)
        adapted = x_input + alpha_clamped * residual

        if len(x.shape) == 3:
            adapted = adapted.squeeze(1)

        if return_features:
            return adapted, features
        return adapted


class HCMMahalanobisGating:
    """Mahalanobis distance gating for the HCM pathology adapter.

    Collects features during training, computes distribution statistics,
    and gates the HCM adapter output at inference based on feature distance.
    """

    def __init__(self, feature_dim=32, threshold_percentile=0.95, epsilon=1e-6):
        self.feature_dim = feature_dim
        self.threshold_percentile = threshold_percentile
        self.epsilon = epsilon
        self.feature_buffer = []
        self.mean = None
        self.precision = None
        self.threshold = None
        self.is_fitted = False

    def collect_features(self, features):
        pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        self.feature_buffer.append(pooled.detach().cpu())

    def fit(self):
        all_features = torch.cat(self.feature_buffer, dim=0)
        self.mean = all_features.mean(dim=0)

        centered = all_features - self.mean
        cov = (centered.T @ centered) / (centered.shape[0] - 1)
        cov += self.epsilon * torch.eye(self.feature_dim)
        self.precision = torch.linalg.inv(cov)

        from scipy.stats import chi2
        self.threshold = chi2.ppf(self.threshold_percentile, df=self.feature_dim)

        self.is_fitted = True
        self.feature_buffer.clear()

    def compute_distance(self, features):
        pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
        centered = pooled - self.mean.to(pooled.device)
        left = centered @ self.precision.to(pooled.device)
        dist_sq = (left * centered).sum(dim=1)
        return torch.clamp(dist_sq, min=0)

    def gate(self, hcm_output, center_contrast_output, features, threshold=None):
        if not self.is_fitted:
            return hcm_output

        thr = threshold if threshold is not None else self.threshold
        dists = self.compute_distance(features)
        mask = (dists < thr).float().view(-1, 1, 1)

        gated = mask * hcm_output + (1 - mask) * center_contrast_output
        return gated

    def save(self, path):
        torch.save({
            'mean': self.mean,
            'precision': self.precision,
            'threshold': torch.tensor(self.threshold),
            'feature_dim': self.feature_dim,
            'threshold_percentile': self.threshold_percentile,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, weights_only=True)
        self.mean = ckpt['mean']
        self.precision = ckpt['precision']
        self.threshold = ckpt['threshold'].item()
        self.feature_dim = ckpt['feature_dim']
        self.threshold_percentile = ckpt['threshold_percentile']
        self.is_fitted = True


class MultiCenterAdaptivePromptMR(nn.Module):
    """多中心自适应PromptMR - 带UNet风格归一化"""

    def __init__(self, base_promptmr_config):
        super().__init__()
        # 基础PromptMR模型
        self.base_model = PromptMR(**base_promptmr_config)

        # 使用更稳定的特征适配器
        self.center_adapters = nn.ModuleDict({
            'Center001': FeatureAdapter(center_type='UIH_30T_umr780'),
            'Center002': FeatureAdapter(center_type='Siemens_30T_CIMA'),
            'Center003': FeatureAdapter(center_type='UIH_30T_umr880'),
            'Center005': FeatureAdapter(center_type='Mixed_Scanners'),
            'Center006': FeatureAdapter(center_type='Siemens_30T_Prisma'),
            'Center007': FeatureAdapter(center_type='Siemens_Mixed')
        })

        # 对比度特定适配器
        self.contrast_adapters = nn.ModuleDict({
            'Cine': FeatureAdapter(contrast_type='cardiac_motion'),
            'LGE': FeatureAdapter(contrast_type='scar_enhancement'),
            'Mapping': FeatureAdapter(contrast_type='quantitative'),
            'T1w': FeatureAdapter(contrast_type='t1_weighted'),
            'T2w': FeatureAdapter(contrast_type='t2_weighted'),
            'Perfusion': FeatureAdapter(contrast_type='dynamic'),
            'T1rho': FeatureAdapter(contrast_type='t1rho_specific'),
            'Flow2d': FeatureAdapter(contrast_type='flow_encoding'),
            'BlackBlood': FeatureAdapter(contrast_type='vessel_suppression')
        })

        # 病理特定适配器
        self.pathology_adapters = nn.ModuleDict({
            'HCM': HCMPathologyAdapter()
        })

        # Mahalanobis gating for HCM adapter
        self.hcm_gating = HCMMahalanobisGating(feature_dim=32)
        
        # Flag to control HCM collection dynamically based on running script
        self.enable_hcm_collection = False

    def norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """参考NormPromptUnet的归一化方法"""
        b, h, w = x.shape
        x_flat = x.reshape(b, h * w)

        mean = x_flat.mean(dim=1).view(b, 1, 1)
        std = x_flat.std(dim=1).view(b, 1, 1)

        # 防止除零
        std = torch.clamp(std, min=1e-8)

        x_normalized = (x - mean) / std
        return x_normalized, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """参考NormPromptUnet的反归一化方法"""
        return x * std + mean

    def pad(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[List[int], List[int], int, int]]:
        """参考NormPromptUnet的padding方法"""
        _, h, w = x.shape
        w_mult = ((w - 1) | 7) + 1
        h_mult = ((h - 1) | 7) + 1
        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x: torch.Tensor, h_pad: List[int], w_pad: List[int],
              h_mult: int, w_mult: int) -> torch.Tensor:
        """参考NormPromptUnet的unpadding方法"""
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def extract_metadata_from_filenames(self, filenames):
        """从文件名提取元数据"""
        if isinstance(filenames, str):
            filenames = [filenames]
            
        center_ids = []
        contrast_types_list = []
        
        for filename in filenames:
            filename = str(filename)
            # 提取中心ID
            center_match = re.search(r'(Center\d+)', filename)
            center_id = center_match.group(1) if center_match else 'unknown'

            # 提取对比度类型
            contrast_types = ['BlackBlood', 'Cine', 'LGE', 'Perfusion', 'Mapping', 'T1rho', 'T1w', 'T2w', 'Flow2d']
            contrast_type = 'unknown'
            for ct in contrast_types:
                if ct in filename:
                    contrast_type = ct
                    break
                    
            center_ids.append(center_id)
            contrast_types_list.append(contrast_type)

        return center_ids, contrast_types_list

    def apply_adaptations(self, features, center_ids, contrast_types, temporal_idx=None):
        """应用适配器 - 使用UNet风格的归一化

        Args:
            features: Reconstructed image features [B, H, W]
            center_ids: Center identifier strings list
            contrast_types: Contrast type strings list
            temporal_idx: Temporal frame index (int or None). When provided,
                HCM features are only collected for ED phase (temporal_idx == 0).
        """

        # 输入检查
        if torch.isnan(features).any() or torch.isinf(features).any():
            print("WARNING: Input features contain NaN/Inf, skipping adaptation")
            return features

        # 保存原始形状
        original_shape = features.shape

        # Step 1: 归一化 (参考NormPromptUnet)
        try:
            features_norm, mean, std = self.norm(features)
        except Exception as e:
            print(f"ERROR in normalization: {e}")
            return features

        # 检查归一化结果
        if torch.isnan(features_norm).any() or torch.isinf(features_norm).any():
            print("WARNING: Normalization produced NaN/Inf")
            return features

        # Step 2: Padding (参考NormPromptUnet)
        try:
            features_padded, pad_sizes = self.pad(features_norm)
        except Exception as e:
            print(f"ERROR in padding: {e}")
            features_padded, pad_sizes = features_norm, None

        adapted = features_padded

        # Step 3: 应用适配器
        try:
            batch_size = features_padded.shape[0]
            adapted_batch = []
            
            if isinstance(center_ids, str):
                center_ids = [center_ids] * batch_size
            if isinstance(contrast_types, str):
                contrast_types = [contrast_types] * batch_size

            for b in range(batch_size):
                f_b = features_padded[b:b+1]
                c_id = center_ids[b]
                c_type = contrast_types[b]
                
                # 中心适配
                if c_id in self.center_adapters:
                    center_adapted = self.center_adapters[c_id](f_b)
                    if not (torch.isnan(center_adapted).any() or torch.isinf(center_adapted).any()):
                        f_b = center_adapted

                # 对比度适配
                if c_type in self.contrast_adapters:
                    contrast_adapted = self.contrast_adapters[c_type](f_b)
                    if not (torch.isnan(contrast_adapted).any() or torch.isinf(contrast_adapted).any()):
                        f_b = contrast_adapted

                # 病理适配 with Mahalanobis gating
                if 'HCM' in self.pathology_adapters:
                    pathology_adapted, hcm_features = self.pathology_adapters['HCM'](f_b, return_features=True)
                    if not (torch.isnan(pathology_adapted).any() or torch.isinf(pathology_adapted).any()):
                        if self.training:
                            f_b = pathology_adapted
                        elif self.hcm_gating.is_fitted:
                            f_b = self.hcm_gating.gate(
                                hcm_output=pathology_adapted,
                                center_contrast_output=f_b,
                                features=hcm_features
                            )
                        else:
                            f_b = pathology_adapted

                    # Collect features during training if explicitly enabled
                    sample_temporal_idx = temporal_idx[b] if isinstance(temporal_idx, list) else temporal_idx
                    is_ed_phase = sample_temporal_idx is None or sample_temporal_idx == 0
                    if self.training and self.enable_hcm_collection and is_ed_phase and not torch.isnan(hcm_features).any():
                        self.hcm_gating.collect_features(hcm_features)
                
                adapted_batch.append(f_b)
                
            adapted = torch.cat(adapted_batch, dim=0)

        except Exception as e:
            print(f"ERROR in adaptation: {e}")
            adapted = features_padded  # 回退到padding后的归一化特征

        # Step 4: Unpadding (参考NormPromptUnet)
        if pad_sizes is not None:
            try:
                adapted = self.unpad(adapted, *pad_sizes)
            except Exception as e:
                print(f"ERROR in unpadding: {e}")
                # 如果unpadding失败，尝试简单的裁剪
                target_h, target_w = original_shape[1], original_shape[2]
                current_h, current_w = adapted.shape[1], adapted.shape[2]
                if current_h >= target_h and current_w >= target_w:
                    h_start = (current_h - target_h) // 2
                    w_start = (current_w - target_w) // 2
                    adapted = adapted[:, h_start:h_start + target_h, w_start:w_start + target_w]

        # Step 5: 反归一化 (参考NormPromptUnet)
        try:
            adapted = self.unnorm(adapted, mean, std)
        except Exception as e:
            print(f"ERROR in unnormalization: {e}")
            return features

        # 最终检查
        if torch.isnan(adapted).any() or torch.isinf(adapted).any():
            print("WARNING: Final adapted result contains NaN/Inf, returning original features")
            return features

        # 确保输出形状正确
        if adapted.shape != original_shape:
            print(f"WARNING: Shape mismatch after adaptation: {adapted.shape} vs {original_shape}")
            try:
                adapted = F.interpolate(adapted.unsqueeze(1), size=original_shape[1:],
                                        mode='bilinear', align_corners=False).squeeze(1)
            except:
                return features

        return adapted

    def save_gating(self, path):
        self.hcm_gating.save(path)

    def load_gating(self, path):
        self.hcm_gating.load(path)

    def forward(self, masked_kspace, mask, filenames=None, slice_num=None, num_slc=None):
        # 基础重建
        recon_image = self.base_model(masked_kspace, mask)

        if torch.isnan(recon_image).any():
            raise ValueError("Reconstructed image contains NaNs and processing cannot continue safely.")

        # 如果没有文件名信息，直接返回基础结果
        if filenames is None:
            return recon_image

        # 提取元数据
        center_ids, contrast_types = self.extract_metadata_from_filenames(filenames)

        # Compute per-sample temporal index from slice_num and num_slc
        temporal_idx = None
        if slice_num is not None and num_slc is not None:
            try:
                batch_size = recon_image.shape[0]
                temporal_idx = []
                for i in range(batch_size):
                    s_num = slice_num[i] if isinstance(slice_num, (list, tuple, torch.Tensor)) else slice_num
                    n_slc = num_slc[i] if isinstance(num_slc, (list, tuple, torch.Tensor)) else num_slc
                    temporal_idx.append(int(s_num) // int(n_slc))
            except Exception:
                temporal_idx = None

        # 应用适配（使用UNet风格的归一化）
        adapted_image = self.apply_adaptations(recon_image, center_ids, contrast_types, temporal_idx)

        return adapted_image