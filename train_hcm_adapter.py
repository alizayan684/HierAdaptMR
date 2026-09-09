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
    创建动态的训练数据加载器 - 每个epoch重新采样
    """
    if args.use_subset:
        epoch_seed = args.seed + epoch * 1000
        print(f"\n=== 重新采样训练子集 Epoch {epoch} (比例: {args.subset_ratio}) ===")
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

def normalize_kspace(kspace, epsilon=1e-8):
    kspace_magnitude = torch.sqrt(kspace[..., 0]**2 + kspace[..., 1]**2)
    max_val = kspace_magnitude.amax(dim=(-2, -3), keepdim=True)
    normalized_kspace = kspace_magnitude / (max_val + epsilon)
    return normalized_kspace

def stable_rss(input_tensor, dim=1, eps=1e-8):
    """More numerically stable RSS implementation"""
    # Ensure non-negative values
    input_tensor = torch.clamp(input_tensor, min=0)

    # Square the values
    squared = input_tensor ** 2

    # Sum along the specified dimension
    summed = torch.sum(squared, dim=dim, keepdim=False)

    # Add epsilon to prevent sqrt(0)
    summed = torch.clamp(summed, min=eps)

    # Take square root
    result = torch.sqrt(summed)

    return result


def train_epoch(args, train_loader, model, optimizer, scaler, epoch, writer):
    model.train()
    running_loss = 0.0
    accumulation_steps = 8
    optimizer.zero_grad()

    # 初始化优化的损失函数
    # ssim_loss_fn = SSIMOptimizedLoss().to(args.device)
    ssim_loss_fn = SSIMLoss().to(args.device)

    #  用于记录最后一个batch的损失（用于tensorboard）
    last_loss_recons_ssim = None
    last_total_loss = None

    for i, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch}")):
        # fully_kspace = batch.fully_kspace.float().to(args.device)
        masked_kspace = batch.masked_kspace.float().to(args.device)
        mask = batch.mask.float().to(args.device)
        target = batch.target.float().to(args.device)

        # 获取文件名用于元数据提取
        filenames = batch.fname if hasattr(batch, 'fname') else None
        slice_num = batch.slice_num if hasattr(batch, 'slice_num') else None
        num_slc = batch.num_slc if hasattr(batch, 'num_slc') else None

        # 取第3个线圈（索引2）
        # fully_kspace = torch.chunk(fully_kspace, 5, dim=1)[2]

        with torch.cuda.amp.autocast():
                recons_pred = model(masked_kspace, mask, filenames, slice_num, num_slc)

        loss_recons_ssim = ssim_loss_fn(recons_pred.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value.to(args.device))
        # # 提取对比度类型用于损失计算
        # contrast_type = None
        # if filenames is not None:
        #     filename = str(filenames[0]) if isinstance(filenames, (list, tuple)) else str(filenames)
        #     contrast_types = ['BlackBlood', 'Cine', 'LGE', 'Perfusion', 'Mapping', 'T1rho', 'T1w', 'T2w', 'Flow2d']
        #     for ct in contrast_types:
        #         if ct in filename:
        #             contrast_type = ct
        #             break
        #
        # # 优化的SSIM损失
        # loss_recons_ssim = ssim_loss_fn(recons_pred, target, contrast_type, filenames)
        #
        # #  在每个batch中检查NaN
        # if torch.isnan(loss_recons_ssim) or torch.isinf(loss_recons_ssim):
        #     print(f"WARNING: NaN/Inf detected in loss_recons_ssim at batch {i}")
        #     print(
        #         f"Recons_pred stats: min={recons_pred.min():.6f}, max={recons_pred.max():.6f}, mean={recons_pred.mean():.6f}")
        #     print(f"Target stats: min={target.min():.6f}, max={target.max():.6f}, mean={target.mean():.6f}")
        #     continue  # 跳过这个batch而不是崩溃

        # 总损失
        total_loss = loss_recons_ssim
        total_loss = total_loss / accumulation_steps

        #  检查总损失
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"WARNING: NaN/Inf detected in total_loss at batch {i}")
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
                optimizer.step()
            optimizer.zero_grad()

        running_loss += loss_recons_ssim.item()

        #  保存最后一个有效batch的损失用于记录
        last_loss_recons_ssim = loss_recons_ssim
        last_total_loss = total_loss

    #  处理最后不完整的accumulation
    if len(train_loader) % accumulation_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3)
            optimizer.step()
        optimizer.zero_grad()

    #  记录到tensorboard（使用最后一个有效batch的损失）
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
    target_model = model.module if hasattr(model, 'module') else model
    target_model.enable_hcm_collection = True

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

    fh = logging.FileHandler(os.path.join(args.experiments_output, 'train_hcm_adapter.log'))
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)
    writer = SummaryWriter(log_dir=args.experiments_output)

    # Freeze all models except pathology adapters
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.pathology_adapters.named_parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {'params': model.pathology_adapters.parameters(), 'lr': args.lr}
    ], weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
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
        train_loaders = create_stable_train_dataloader(args, base_train_dataset, epoch)

        epoch_start_time = time.time()
        epoch_train_loss = train_epoch(args, train_loaders, model, optimizer, scaler, epoch, writer)
        
        # Fit HCM gating at the end of every epoch so it's ready for validation
        target_model_local = model.module if hasattr(model, 'module') else model
        if hasattr(target_model_local, 'hcm_gating') and target_model_local.hcm_gating.feature_buffer:
            target_model_local.hcm_gating.fit()

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

        # 学习率调度
        scheduler.step()

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

    # 训练结束后的统计
    logger.info("=" * 50)
    logger.info("TRAINING COMPLETED")
    logger.info(f"Best SSIM achieved: {best_SSIM:.6f}")
    logger.info(f"Total training time: {total_training_time:.2f} seconds")

    # Fit Mahalanobis gating distribution from collected features
    if model.hcm_gating.feature_buffer:
        model.hcm_gating.fit()
        gating_path = os.path.join(args.experiments_output, 'hcm_gating_stats.pt')
        model.save_gating(gating_path)
        logger.info(f"HCM gating stats saved to {gating_path}")
        logger.info(f"  Threshold (chi2, p={model.hcm_gating.threshold_percentile}): "
                    f"{model.hcm_gating.threshold:.4f}")
        logger.info(f"  Feature dim: {model.hcm_gating.feature_dim}")
    else:
        logger.info("No HCM features collected during training, skipping gating stats")

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
    parser.add_argument("--lr", default=0.0002, type=float, help="Adam learning rate")
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
