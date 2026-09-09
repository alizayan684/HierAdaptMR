import pathlib
from argparse import ArgumentParser
from torch.utils.data import Subset
from data_loading.mri_data import *
from torch.cuda.amp import  GradScaler
import time
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
import logging
from data_loading.transforms import CmrxReconDataTransform
from data_loading.data_module import DataModule
from data_loading.subsample import CmrxRecon25MaskFunc
import torch
import numpy as np
from mri_network.multi_center_adapter import MultiCenterAdaptivePromptMR
import re
from mri_utils.losses import SSIMLoss
import torch.nn as nn


def create_combined_dataloader(args, base_train_dataset, epoch=0):
    """
    组合稳定采样和性能感知权重的数据加载器
    先进行稳定的分层采样，然后应用性能感知权重
    """

    # 第一步：进行分层采样
    if args.use_subset:
        epoch_seed = args.seed + epoch * 1000
        print(f"\n=== 步骤1: 创建训练子集 Epoch {epoch} (比例: {args.subset_ratio}) ===")
        subset_dataset = uniform_stratified_sampling(
            base_train_dataset,
            subset_ratio=args.subset_ratio,
            random_state=epoch_seed
        )
        working_dataset = subset_dataset
    else:
        working_dataset = base_train_dataset

    # 第二步：应用性能感知的采样权重
    print(f"\n=== 步骤2: 应用性能感知权重 ===")

    # 根据性能分组定义采样权重
    high_performance_centers = ['Center005_GE', 'Center006_UIH']
    low_performance_centers = ['Center002']
    medium_performance_centers = ['Center001', 'Center003', 'Center005_Siemens', 'Center006_Siemens']

    # 计算每个样本的权重
    sample_weights = []
    center_count = {}

    # 获取实际的样本列表
    if hasattr(working_dataset, 'indices'):  # 如果是Subset
        actual_samples = [working_dataset.dataset.raw_samples[i] for i in working_dataset.indices]
    else:  # 如果是原始数据集
        actual_samples = working_dataset.raw_samples

    for sample in actual_samples:
        filename = str(sample.fname)
        weight = 1.0  # 默认权重
        center_type = 'unknown'

        # 确定中心类型并分配权重
        for center in high_performance_centers:
            if center in filename:
                weight = 0.8  # 高性能中心降低采样权重
                center_type = 'high_performance'
                break
        else:
            for center in low_performance_centers:
                if center in filename:
                    weight = 3.0  # 低性能中心大幅增加采样权重
                    center_type = 'low_performance'
                    break
            else:
                for center in medium_performance_centers:
                    if center in filename:
                        weight = 1.5  # 中等性能中心适度增加采样权重
                        center_type = 'medium_performance'
                        break

        sample_weights.append(weight)
        center_count[center_type] = center_count.get(center_type, 0) + 1

    # 打印统计信息
    print(f"=== 组合采样统计 ===")
    total_samples = len(actual_samples)
    print(f"总样本数: {total_samples}")
    for center_type, count in center_count.items():
        percentage = (count / total_samples) * 100
        print(f"{center_type}: {count} samples ({percentage:.1f}%)")
    print("=" * 30)

    # 创建加权采样器
    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return torch.utils.data.DataLoader(
        working_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

def uniform_stratified_sampling(dataset, subset_ratio=0.4, random_state=42):
    """
    修正的层次化均匀分层抽样
    适配实际的JSON文件格式
    对于Center002和Center006_Siemens_30T_Prisma使用全部数据
    """
    np.random.seed(random_state)

    all_indices = list(range(len(dataset)))

    # 层次化数据结构：Center → Vendor → Modality → Patient → [indices]
    hierarchical_data = {}

    # 按层次结构分组数据
    for idx in all_indices:
        center, vendor, modality, patient = extract_hierarchical_info_from_filename_fixed(dataset, idx)

        # 构建层次结构
        if center not in hierarchical_data:
            hierarchical_data[center] = {}
        if vendor not in hierarchical_data[center]:
            hierarchical_data[center][vendor] = {}
        if modality not in hierarchical_data[center][vendor]:
            hierarchical_data[center][vendor][modality] = {}
        if patient not in hierarchical_data[center][vendor][modality]:
            hierarchical_data[center][vendor][modality][patient] = []

        hierarchical_data[center][vendor][modality][patient].append(idx)

    selected_indices = []

    # 按层次结构进行采样
    for center in sorted(hierarchical_data.keys()):
        for vendor in sorted(hierarchical_data[center].keys()):
            for modality in sorted(hierarchical_data[center][vendor].keys()):
                # 检查是否为特殊中心，使用全部数据
                use_all_data = (center == 'Center002' or
                                (center == 'Center006' and 'Siemens_30T_Prisma' in vendor))

                # 获取当前组合下的所有患者
                patients = list(hierarchical_data[center][vendor][modality].keys())
                total_patients = len(patients)

                if use_all_data:
                    # 使用全部患者数据
                    selected_patients = patients
                    logging.info(f"Using ALL data for {center} {vendor} {modality}: {total_patients} patients")
                else:
                    # 计算需要采样的患者数量
                    n_patients_to_sample = max(1, int(total_patients * subset_ratio))

                    # 随机选择患者
                    if n_patients_to_sample >= total_patients:
                        selected_patients = patients
                    else:
                        selected_patients = np.random.choice(
                            patients, n_patients_to_sample, replace=False
                        ).tolist()

                # 收集选中患者的所有数据（包括该患者的所有序列）
                group_indices = []
                for patient in selected_patients:
                    patient_indices = hierarchical_data[center][vendor][modality][patient]
                    group_indices.extend(patient_indices)

                selected_indices.extend(group_indices)

    total_original = len(dataset)
    total_selected = len(selected_indices)

    logging.info(f"\n=== 总体统计 ===")
    logging.info(f" 总文件数: {total_original} → {total_selected} ({total_selected / total_original * 100:.1f}%)")
    logging.info("=" * 60)

    return Subset(dataset, selected_indices)


def extract_hierarchical_info_from_filename_fixed(dataset, idx):
    """
    修正的文件名解析函数
    适配实际格式：train/Center001_UIH_30T_umr780_Cine_P001_cine_lax_3ch.h5
    """
    try:
        if hasattr(dataset, 'raw_samples'):
            file_path = str(dataset.raw_samples[idx].fname)
        else:
            return 'unknown', 'unknown', 'unknown', 'unknown'

        # 获取文件名（不包含路径）
        file_name = file_path.split('/')[-1]

        # 移除.h5扩展名
        if file_name.endswith('.h5'):
            file_name = file_name[:-3]

        # 分割文件名
        parts = file_name.split('_')

        if len(parts) >= 6:
            # 实际格式解析：Center001_UIH_30T_umr780_Cine_P001_cine_lax_3ch
            center = parts[0]  # Center001
            vendor_info = f"{parts[1]}_{parts[2]}_{parts[3]}"  # UIH_30T_umr780
            patient = parts[4]  # P002
            modality = parts[5]  # cine
            return center, vendor_info, modality, patient

        else:
            # 如果格式不符合预期，尝试备选解析
            return parse_alternative_filename_format_fixed(file_name)

    except Exception as e:
        print(f"解析文件名时出错: {file_path}, 错误: {e}")
        return 'unknown', 'unknown', 'unknown', 'unknown'


def parse_alternative_filename_format_fixed(file_name):
    """
    备选的文件名解析方法 - 适配实际格式
    """
    try:
        parts = file_name.split('_')

        center = 'unknown'
        vendor = 'unknown'
        modality = 'unknown'
        patient = 'unknown'

        # 查找center
        for i, part in enumerate(parts):
            if 'Center' in part:
                center = part
                break

        # 查找patient (格式：P001, P002等)
        for part in parts:
            if part.startswith('P') and len(part) > 1 and part[1:].isdigit():
                patient = part
                break

        # 查找modality (在文件名中的位置)
        modality_keywords = ['BlackBlood', 'Cine', 'LGE', 'Perfusion', 'Mapping', 'T1rho', 'T1w', 'T2w', 'Flow2d']
        for part in parts:
            if part in modality_keywords:
                modality = part
                break

        # 查找vendor (通常在center后面)
        vendor_keywords = ['UIH', 'Siemens', 'GE']
        vendor_parts = []
        for part in parts:
            if any(keyword in part for keyword in vendor_keywords):
                vendor_parts.append(part)
            elif part.endswith('T') and part[:-1].replace('.', '').isdigit():  # 磁场强度
                vendor_parts.append(part)
            elif any(model in part for model in ['umr', 'Prisma', 'Vida', 'Aera', 'voyager', 'CIMA']):
                vendor_parts.append(part)

        if vendor_parts:
            vendor = '_'.join(vendor_parts)

        return center, vendor, modality, patient

    except:
        return 'unknown', 'unknown', 'unknown', 'unknown'


def create_stable_train_dataloader(args, base_train_dataset, epoch=0):
    """
    创建稳定的训练数据加载器 - 每个epoch重新采样
    """
    if args.use_subset:
        epoch_seed = args.seed + epoch * 1000
        print(f"\n=== 创建训练子集 Epoch {epoch} (比例: {args.subset_ratio}) ===")
        subset_dataset = uniform_stratified_sampling(
            base_train_dataset,
            subset_ratio=args.subset_ratio,
            random_state=epoch_seed
        )
        dataset = subset_dataset
    else:
        dataset = base_train_dataset

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        drop_last=True
    )


def get_base_dataset(args, train=True):
    """获取基础数据集，不进行采样"""
    if train:
        mask_func = CmrxRecon25MaskFunc(
            num_low_frequencies=args.num_low_frequencies,
            num_adj_slices=args.num_adj_slices,
            seed=None
        )
        transform = CmrxReconDataTransform(mask_func=mask_func, use_seed=False)
    else:
        mask_func = CmrxRecon25MaskFunc(
            num_low_frequencies=args.num_low_frequencies,
            num_adj_slices=args.num_adj_slices,
            seed=42
        )
        transform = CmrxReconDataTransform(mask_func=mask_func, use_seed=True)

    data_module = DataModule(
        slice_dataset='data_loading.mri_data.CmrxReconSliceDataset',
        data_path=args.data_path,
        train_transform=transform if train else None,
        val_transform=transform if not train else None,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # 返回完整数据集
    return data_module.train_dataloader().dataset if train else data_module.val_dataloader().dataset


def create_train_dataloader(args, base_train_dataset, epoch):
    """为每个epoch创建训练数据加载器 - 每次重新采样"""
    if args.use_subset:
        # 使用epoch作为随机种子的一部分，确保每个epoch采样不同
        epoch_seed = args.seed + epoch * 1000

        print(f"\n=== Epoch {epoch + 1}: 训练数据重新采样 ===")
        subset_dataset = uniform_stratified_sampling(
            base_train_dataset,
            subset_ratio=args.subset_ratio,
            random_state=epoch_seed
        )
        dataset = subset_dataset
    else:
        dataset = base_train_dataset

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,  # 训练时打乱
        num_workers=args.num_workers,
        pin_memory=True
    )


def create_val_dataloader(args, base_val_dataset):
    """创建验证数据加载器 - 固定不变"""
    return torch.utils.data.DataLoader(
        base_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # 验证时不打乱
        num_workers=args.num_workers,
        pin_memory=True
    )


def train_epoch(args, train_loader, model, optimizer, scaler, epoch, writer):
    model.train()
    running_loss = 0.0
    accumulation_steps = 8
    optimizer.zero_grad()

    # 初始化损失函数
    ssim_loss_fn = SSIMLoss().to(args.device)

    # 定义中心特异性损失权重
    center_loss_weights = {
        'Center002': 2.0,  # 重点优化Center002
        'Center001': 1.2,
        'Center003': 1.2,
        'Center005_Siemens': 1.1,
        'Center006_Siemens': 1.1,
        'Center005_GE': 0.8,     # 已经表现很好，权重较小
        'Center006_UIH': 0.8,    # 已经表现很好，权重较小
    }

    # 用于记录最后一个batch的损失
    last_loss_recons_ssim = None
    last_total_loss = None

    for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        masked_kspace = batch.masked_kspace.float().to(args.device)
        mask = batch.mask.float().to(args.device)
        target = batch.target.float().to(args.device)

        # 获取文件名用于元数据提取
        filenames = batch.fname if hasattr(batch, 'fname') else None
        slice_num = batch.slice_num if hasattr(batch, 'slice_num') else None
        num_slc = batch.num_slc if hasattr(batch, 'num_slc') else None

        # 确定每个样本的中心权重
        batch_size = recons_pred.shape[0] if 'recons_pred' in dir() else masked_kspace.shape[0]
        loss_weights = []
        if filenames is not None:
            fnames = filenames if isinstance(filenames, (list, tuple)) else [filenames]
            for fname in fnames:
                fname_str = str(fname)
                center_name = 'default'
                for center in center_loss_weights.keys():
                    if center in fname_str:
                        center_name = center
                        break
                loss_weights.append(center_loss_weights.get(center_name, 1.0))
        else:
            loss_weights = [1.0] * batch_size

        with torch.cuda.amp.autocast():
            recons_pred = model(masked_kspace, mask, filenames, slice_num, num_slc)

        # 修正函数名
        base_loss = ssim_loss_fn(recons_pred.unsqueeze(1), target.unsqueeze(1),
                                data_range=batch.max_value.to(args.device))

        # 应用中心特异性权重 (per-sample)
        if batch_size > 1:
            weight_tensor = torch.tensor(loss_weights, device=base_loss.device, dtype=base_loss.dtype)
            weighted_loss = (base_loss * weight_tensor).mean()
        else:
            weighted_loss = base_loss * loss_weights[0]

        total_loss = weighted_loss / accumulation_steps

        # 检查总损失
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"WARNING: NaN/Inf detected in total_loss at batch {i}, center: {center_name}")
            continue

        # 反向传播
        if scaler is not None:
            scaler.scale(total_loss).backward()
        else:
            total_loss.backward()

        # 梯度更新
        if (i + 1) % accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 降低梯度裁剪阈值
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        running_loss += base_loss.item()

        # 修正变量名
        last_loss_recons_ssim = base_loss  # 使用base_loss而不是未定义的loss_recons_ssim
        last_total_loss = total_loss

    # 处理最后不完整的accumulation
    if len(train_loader) % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    # 记录到tensorboard
    if last_loss_recons_ssim is not None and last_total_loss is not None:
        writer.add_scalar(f'Train/loss_recons_ssim', last_loss_recons_ssim.item(), epoch)
        writer.add_scalar(f'Train/total_loss', last_total_loss.item() * accumulation_steps, epoch)

    return running_loss / len(train_loader)


def validate(args, val_loader, model, writer, epoch):
    model.eval()
    running_ssim = 0.0
    num_batches = 0
    valid_batches = 0  #  跟踪有效的batch数量
    ssim_loss_fn = SSIMLoss().to(args.device)
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validating"):
            try:
                fully_kspace, masked_kspace, mask, target = (
                    batch.fully_kspace.float().to(args.device),
                    batch.masked_kspace.float().to(args.device),
                    batch.mask.float().to(args.device),
                    batch.target.float().to(args.device)
                )

                # 获取文件名
                filenames = batch.fname if hasattr(batch, 'fname') else None
                slice_num = batch.slice_num if hasattr(batch, 'slice_num') else None
                num_slc = batch.num_slc if hasattr(batch, 'num_slc') else None

                #  输入数据检查
                if torch.isnan(masked_kspace).any() or torch.isinf(masked_kspace).any():
                    print(f"WARNING: NaN/Inf in validation input, skipping batch")
                    num_batches += 1
                    continue

                if torch.isnan(target).any() or torch.isinf(target).any():
                    print(f"WARNING: NaN/Inf in validation target, skipping batch")
                    num_batches += 1
                    continue

                with torch.cuda.amp.autocast():
                        recons_pred = model(masked_kspace, mask, filenames, slice_num, num_slc)

                # 计算SSIM
                ssim = 1- ssim_loss_fn(recons_pred.unsqueeze(1), target.unsqueeze(1),
                                                data_range=batch.max_value.to(args.device))

                #  检查SSIM结果
                if torch.isnan(ssim) or torch.isinf(ssim):
                    print(f"WARNING: NaN/Inf SSIM, skipping batch")
                    num_batches += 1
                    continue

                running_ssim += ssim.item()
                valid_batches += 1

            except Exception as e:
                print(f"ERROR in validation batch: {e}")
                # 继续处理下一个batch
                pass

            num_batches += 1

    #  检查是否有有效的batch
    if valid_batches == 0:
        print("WARNING: No valid batches in validation!")
        avg_ssim = 0.0
    else:
        avg_ssim = running_ssim / valid_batches

    #  记录统计信息
    print(f"Validation: {valid_batches}/{num_batches} valid batches, avg SSIM: {avg_ssim:.4f}")

    # 记录到tensorboard
    if writer:
        writer.add_scalar('Validation/SSIM', avg_ssim, epoch)
        writer.add_scalar('Validation/ValidBatches', valid_batches, epoch)
        writer.add_scalar('Validation/TotalBatches', num_batches, epoch)

    return 1 - avg_ssim, avg_ssim


def setup_optimizer_with_performance_based_lr(model, args, center_performance_map):
    """根据中心性能设置不同的学习率"""

    # 根据性能分组
    high_performance_centers = ['Center005_GE', 'Center006_UIH']
    low_performance_centers = ['Center002']
    medium_performance_centers = ['Center001', 'Center003', 'Center005_Siemens', 'Center006_Siemens']

    # 为不同性能的中心设置不同学习率
    param_groups = []

    # 基础模型 - 保持较低学习率
    param_groups.append({
        'params': model.base_model.parameters(),
        'lr': args.lr * 0.01,
        'weight_decay': args.weight_decay
    })

    # 为不同性能的中心适配器设置不同学习率
    for center_name, adapter in model.center_adapters.named_children():
        if any(hp in center_name for hp in high_performance_centers):
            lr_multiplier = 0.01  # 高性能中心用很小的学习率
            print(f"High performance center {center_name}: lr = {args.lr * lr_multiplier}")
        elif any(lp in center_name for lp in low_performance_centers):
            lr_multiplier = 0.1  # 低性能中心用较大的学习率
            print(f"Low performance center {center_name}: lr = {args.lr * lr_multiplier}")
        elif any(mp in center_name for mp in medium_performance_centers):
            lr_multiplier = 0.05  # 中等性能中心用中等学习率
            print(f"Medium performance center {center_name}: lr = {args.lr * lr_multiplier}")
        else:
            lr_multiplier = 0.03  # 未分类的中心用默认学习率
            print(f"Default center {center_name}: lr = {args.lr * lr_multiplier}")

        param_groups.append({
            'params': adapter.parameters(),
            'lr': args.lr * lr_multiplier,
            'weight_decay': args.weight_decay * 0.1
        })

    # 对比度适配器
    param_groups.append({
        'params': model.contrast_adapters.parameters(),
        'lr': args.lr * 0.02,
        'weight_decay': args.weight_decay * 0.1
    })

    optimizer = torch.optim.AdamW(param_groups, eps=1e-8)
    return optimizer


def save_checkpoint(state, is_best, checkpoint_dir, filename='checkpoint.pth.tar'):
    """保存检查点，包括最佳SSIM模型"""
    # 确保目录存在
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 保存当前epoch的检查点
    filepath = os.path.join(checkpoint_dir, filename)
    torch.save(state, filepath)

    # 如果是最佳模型，额外保存一份
    if is_best:
        best_filepath = os.path.join(checkpoint_dir, 'best_ssim_model.pth.tar')
        torch.save(state, best_filepath)
        print(f"New best SSIM model saved: {best_filepath}")


def cli_main(args):
    device = torch.device(args.device)
    if args.gpus > 1:
        print(f"WARNING: --gpus={args.gpus} is set but multi-GPU training is not implemented. "
              f"Running on single device: {device}")

    # 使用多中心自适应模型
    base_promptmr_config = {
        'num_cascades': 12,
        'num_adj_slices': 5,
        'n_feat0': 48,
        'feature_dim': [72, 96, 120],
        'prompt_dim': [24, 48, 72],
        'sens_n_feat0': 24,
        'sens_feature_dim': [36, 48, 60],
        'sens_prompt_dim': [12, 24, 36],
        'len_prompt': [5, 5, 5],
        'prompt_size': [64, 32, 16],
        'n_enc_cab': [2, 3, 3],
        'n_dec_cab': [2, 2, 3],
        'n_skip_cab': [1, 1, 1],
        'n_bottleneck_cab': 3,
        'no_use_ca': False,
        'adaptive_input': True,  # adaptive input
        'n_buffer': 4,  # buffer size in adaptive input, fixed to 4
        'n_history': 11,  # historical feature aggregation
    }

    model = MultiCenterAdaptivePromptMR(base_promptmr_config).to(device)

    best_SSIM = 0.0
    best_val_loss = float('inf')

    if args.use_checkpoint:
        checkpoint = torch.load(args.pretrained, map_location=args.device, weights_only=True)
        pretrained_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(pretrained_state_dict, strict=False)
        if 'hcm_gating' in checkpoint and checkpoint['hcm_gating'] is not None:
            model.hcm_gating.mean = checkpoint['hcm_gating']['mean']
            model.hcm_gating.precision = checkpoint['hcm_gating']['precision']
            model.hcm_gating.threshold = checkpoint['hcm_gating']['threshold']
            model.hcm_gating.is_fitted = True
            print("HCM gating stats loaded from checkpoint")
        if args.resume:
            # Same-stage resume: restore best-metric thresholds so previously
            # saved best model is not overwritten by worse epochs.
            if 'best_SSIM' in checkpoint:
                best_SSIM = checkpoint['best_SSIM']
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            print(f"Resuming same-stage training: best_SSIM={best_SSIM:.6f}, best_val_loss={best_val_loss:.6f}")
        else:
            # Transfer learning from a backbone checkpoint: best metrics belong
            # to a different model/stage, so keep the fresh 0.0/inf thresholds.
            print("Transfer learning from backbone checkpoint: best metrics reset (best_SSIM=0.0)")
        del checkpoint  # Release the memory used by the state_dict
        torch.cuda.empty_cache()  # Clear the cache to free up memory
    else:
        print("  No pretrained model loaded!")
        print("  All parameters will be trainable (not recommended for adaptation training)")

    # 设置日志
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(os.path.join(args.experiments_output, 'train_multicenter.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    writer = SummaryWriter(log_dir=args.experiments_output)

    # 使用不同的学习率策略
    # trainable_params = [p for p in model.parameters() if p.requires_grad]
    # 使用性能感知的优化器（修正这里）
    center_performance_map = {}  # 可以从配置文件加载
    optimizer = setup_optimizer_with_performance_based_lr(model, args, center_performance_map)

    # 改用ReduceLROnPlateau调度器，更适合finetune
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )
    scaler = GradScaler()

    total_training_time = 0
    epoch_count = 0
    epochs_no_improve = 0
    patience = 6  # 增加耐心值，因为多中心训练可能需要更多时间收敛

    # 获取基础数据集
    base_train_dataset = get_base_dataset(args, train=True)
    val_loaders = create_val_dataloader(args, get_base_dataset(args, train=False))
    logging.info(
        f'Training slices: {len(base_train_dataset.raw_samples)}, Validating slices:{len(val_loaders.dataset)}')

    # 添加数据集统计
    log_dataset_statistics(base_train_dataset, logger)

    for epoch in range(args.max_epochs):
        # 每个epoch重新创建训练数据加载器
        train_loaders = create_combined_dataloader(args, base_train_dataset, epoch)

        epoch_start_time = time.time()
        epoch_train_loss = train_epoch(args, train_loaders, model, optimizer, scaler, epoch, writer)
        epoch_val_loss, epoch_val_SSIM = validate(args, val_loaders, model, writer, epoch)

        epoch_time = time.time() - epoch_start_time
        total_training_time += epoch_time
        epoch_count += 1

        logger.info(f"Epoch {epoch_count}, Train Loss: {epoch_train_loss:.6f}, "
                    f"Validation Loss: {epoch_val_loss:.6f}, Validation SSIM: {epoch_val_SSIM:.6f}, "
                    f"Time: {epoch_time:.2f} seconds")

        # 判断是否为最佳模型
        is_best = epoch_val_SSIM > best_SSIM
        if is_best:
            best_SSIM = epoch_val_SSIM
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            logger.info(f"New best SSIM: {best_SSIM:.6f}")
        else:
            epochs_no_improve += 1

        # 保存检查点
        save_checkpoint({
            'epoch': epoch_count,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
            'best_SSIM': best_SSIM,
            'current_val_loss': epoch_val_loss,
            'current_SSIM': epoch_val_SSIM,
            'hcm_gating': {
                'mean': model.hcm_gating.mean,
                'precision': model.hcm_gating.precision,
                'threshold': model.hcm_gating.threshold,
            } if model.hcm_gating.is_fitted else None
        }, is_best, args.experiments_output, filename=f'checkpoint_epoch_{epoch_count}.pth.tar')

        # 早停检查
        if epochs_no_improve >= patience:
            logger.info(f"Early stopping triggered after {epoch + 1} epochs. "
                        f"No improvement in SSIM for {patience} epochs.")
            logger.info(f"Best SSIM achieved: {best_SSIM:.6f}")
            break

        logger.info(f"Total Training Time after {epoch_count} epochs: {total_training_time:.2f} seconds")
        # 学习率调度
        scheduler.step(epoch_val_SSIM)


    # 训练结束后的统计
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best SSIM achieved: {best_SSIM:.6f}")
    logger.info(f"Total training time: {total_training_time:.2f} seconds")
    logger.info("=" * 50)


def log_dataset_statistics(dataset, logger):
    """记录数据集统计信息"""
    center_stats = {}
    contrast_stats = {}

    for i in range(len(dataset.raw_samples)):
        filename = str(dataset.raw_samples[i].fname)

        # 统计中心分布
        center_match = re.search(r'(Center\d+)', filename)
        if center_match:
            center = center_match.group(1)
            center_stats[center] = center_stats.get(center, 0) + 1

        # 统计对比度分布
        contrast_types = ['BlackBlood', 'Cine', 'LGE', 'Perfusion', 'Mapping', 'T1rho', 'T1w', 'T2w', 'Flow2d']
        for ct in contrast_types:
            if ct in filename:
                contrast_stats[ct] = contrast_stats.get(ct, 0) + 1
                break

    logger.info("=" * 50)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 50)
    logger.info("Center Distribution:")
    for center, count in sorted(center_stats.items()):
        logger.info(f"  {center}: {count} samples")

    logger.info("Contrast Distribution:")
    for contrast, count in sorted(contrast_stats.items()):
        logger.info(f"  {contrast}: {count} samples")
    logger.info("=" * 50)


def build_args():
    parser = ArgumentParser()
    parser.add_argument("--data_path", default=pathlib.Path('/media/ruru/ad31566c-e032-4ffa-a8cf-751b9dbab424/work/CMRxRecon2025/preprocess1'))
    parser.add_argument("--experiments_output", default=pathlib.Path('/home/ruru/Documents/work/CMR2025/cmr2025_R1/output'))
    parser.add_argument("--pretrained", default=pathlib.Path('/home/ruru/Documents/work/CMR2025/summary_results/backbone_promptmrV2/result8/checkpoint_epoch_0.pth.tar'))
    parser.add_argument("--use_checkpoint", default=True, help="Use checkpoint (default: False)")
    parser.add_argument("--resume", action="store_true", help="Resume same-stage training: restore best_SSIM/best_val_loss from --pretrained checkpoint. Omit for transfer learning from a backbone checkpoint (best metrics reset to 0.0/inf).")
    parser.add_argument("--use_subset", default=True)
    parser.add_argument("--subset_ratio", default=0.3)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--gpus", default=1, type=int, help="Number of GPUs to use (currently single-device only)")
    parser.add_argument("--device", default='cuda')
    parser.add_argument("--num_workers", default=1, type=int, help="Number of workers to use in data loader")
    parser.add_argument("--lr", default=0.00015, type=float, help="Adam learning rate")
    parser.add_argument("--lr_step_size", default=3, type=int, help="Epoch at which to decrease step size")
    parser.add_argument("--lr_gamma", default=0.9, type=float, help="Extent to which step size should be decreased")
    parser.add_argument("--weight_decay", default=1e-4, type=float, help="Strength of weight decay regularization")
    parser.add_argument("--log_every_n_epochs", default=1, type=int, help="Log outputs to TensorBoard every N epochs")
    parser.add_argument("--seed", default=42, help="random seed")
    parser.add_argument("--max_epochs", default=20, type=int, help="max number of epochs")
    parser.add_argument('--num_low_frequencies', type=int, nargs='+', default=[20], help='Number of low frequency lines to sample')
    parser.add_argument('--num_adj_slices', type=int, default=5, help='Number of adjacent slices for k-t sampling')
    parser.add_argument('--task_type', type=str, default='regular_task1')

    args = parser.parse_args()
    if not args.experiments_output.exists():
        args.experiments_output.mkdir(parents=True)

    return args


if __name__ == "__main__":
    args = build_args()
    cli_main(args)
