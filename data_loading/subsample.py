import contextlib
from typing import Optional, Sequence, Tuple, Union
import numpy as np
import torch

@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[Union[int, Tuple[int, ...]]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass ``MaskFunc``
    and overwrite the ``sample_mask`` logic. See examples in ``RandomMaskFunc``
    and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                "Number of center fractions should match number of accelerations "
                "if allow_any_combination is False."
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space mask: 1)
        the center mask (e.g., for sensitivity map calculation) and 2) the
        acceleration mask (for the edge of k-space). Both of these masks, as
        well as the integer of low frequency samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).

        Returns:
            A 3-tuple contaiing 1) the mask for the center of k-space, 2) the
            mask for the high frequencies of k-space, and 3) the integer count
            of low frequency samples.
        """
        num_cols = shape[-2]
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking (for equispaced masks).
            num_low_frequencies: Integer count of low-frequency lines sampled.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.

        Returns:
            A mask for hte low spatial frequencies of k-space.
        """
        num_cols = shape[-2]
        mask = np.zeros(num_cols, dtype=np.float32)
        pad = (num_cols - num_low_freqs + 1) // 2
        mask[pad: pad + num_low_freqs] = 1
        assert mask.sum() == num_low_freqs

        return mask

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]
        
    def _get_ti_adj_idx_list(self,ti, num_t_in_volume):
        '''
        get the circular adjacent indices of the temporal axis for the given ti.
        '''
        start_lim, end_lim = -(num_t_in_volume//2), (num_t_in_volume//2+1)
        start, end = max(self.start_adj,start_lim), min(self.end_adj,end_lim)
        # Generate initial list of indices
        ti_idx_list = [(i + ti) % num_t_in_volume for i in range(start, end)]
        # duplicate padding if necessary
        replication_prefix = max(start_lim-self.start_adj,0) * ti_idx_list[0:1]
        replication_suffix = max(self.end_adj-end_lim,0) * ti_idx_list[-1:]

        return replication_prefix + ti_idx_list + replication_suffix
    
class RandomMaskFunc(MaskFunc):
    """
    Creates a random sub-sampling mask of a given shape. FastMRI multi-coil knee dataset uses this mask type.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        prob = (num_cols / acceleration - num_low_frequencies) / (
            num_cols - num_low_frequencies
        )

        return self.rng.uniform(size=num_cols) < prob


class EquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset = self.rng.randint(0, high=round(acceleration))

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class EquispacedMaskFractionFunc(MaskFunc):
    """
    Equispaced mask with approximate acceleration matching. FastMRI multi-coil brain dataset uses this mask type.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        # determine acceleration rate by adjusting for the number of low frequencies
        adjusted_accel = (acceleration * (num_low_frequencies - num_cols)) / (
            num_low_frequencies * acceleration - num_cols
        )
        if offset is None:
            offset = self.rng.randint(0, high=round(adjusted_accel))

        mask = np.zeros(num_cols)
        accel_samples = np.arange(offset, num_cols - 1, adjusted_accel)
        accel_samples = np.around(accel_samples).astype(np.uint)
        mask[accel_samples] = 1.0

        return mask

class FixedLowRandomMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines and a fixed number of low-frequency lines. CMRxRecon dataset uses this mask type.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_low_frequencies, acceleration = self.choose_acceleration()
        num_low_frequencies = int(num_low_frequencies)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, 0, num_low_frequencies
            ),
            shape,
        )
        return center_mask, acceleration_mask, num_low_frequencies

    def sample_kt_mask(self, shape, offset, num_adj_slices, slice_idx, num_t,num_slc, rng):
        if not hasattr(self, 'start_adj'):
            self.start_adj, self.end_adj = -(num_adj_slices//2), num_adj_slices//2+1

        num_cols = shape[-2]
        num_low_frequencies = rng.choice(self.center_fractions)
        acceleration = rng.choice(self.accelerations)

        mask = []
        for _ in range(num_t): #num_adj_slices
            center_mask = self.reshape_mask(
                self.calculate_center_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask(
                rng.uniform(size=num_cols) < 1/acceleration, ##* use the rng from cmrxrecon25maskfunc
                shape,
            )
            mask.append(torch.max(center_mask, acceleration_mask))


        mask = torch.cat(mask, dim=0)

        ti = slice_idx//num_slc
        select_list = self._get_ti_adj_idx_list(ti,num_t)
        mask = mask[select_list]
    
        return mask, num_low_frequencies

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:

        prob = 1/acceleration

        return self.rng.uniform(size=num_cols) < prob
        

class FixedLowEquiSpacedMaskFunc(MaskFunc):
    """
    Sample data with equally-spaced k-space lines and a fixed number of low-frequency lines. CMRxRecon dataset uses this mask type.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def sample_mask(self, shape, offset):

        num_cols = shape[-2]
        num_low_frequencies, acceleration = self.choose_acceleration()
        num_low_frequencies = int(num_low_frequencies)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )
        return center_mask, acceleration_mask, num_low_frequencies

    def sample_uniform_mask(self, shape, offset, rng, acc=None):

        num_cols = shape[-2]
        num_low_frequencies = rng.choice(self.center_fractions)

        if acc:
            acceleration = int(acc)
        else:
            acceleration = rng.choice(self.accelerations)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(shape, num_low_frequencies), shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, offset, num_low_frequencies
            ),
            shape,
        )
        mask = torch.max(center_mask, acceleration_mask)
        return mask, num_low_frequencies

    def sample_kt_mask(self, shape, offset, num_adj_slices, slice_idx, num_t,num_slc, rng, seed):
        ##* important: need to use the rng from cmrxrecon25maskfunc; so validation is reproduceable;

        if not hasattr(self, 'start_adj'):
            self.start_adj, self.end_adj = -(num_adj_slices//2), num_adj_slices//2+1

        num_cols = shape[-2]
        num_low_frequencies = rng.choice(self.center_fractions)
        acceleration = rng.choice(self.accelerations)

        if offset is None:
            offset=0
        num_low_frequencies = int(num_low_frequencies)
        if seed is None: ##* training
            ti = rng.randint(num_t)
        else: ##* validation
            ti = slice_idx//num_slc
        select_list = self._get_ti_adj_idx_list(ti,num_t)
        mask = []
        for _offset in select_list:
            center_mask = self.reshape_mask(
                self.calculate_center_mask(shape, num_low_frequencies), shape
            )
            acceleration_mask = self.reshape_mask(
                self.calculate_acceleration_mask(
                    num_cols, acceleration, _offset%acceleration, num_low_frequencies
                ),
                shape,
            )
            mask.append(torch.max(center_mask, acceleration_mask))
        mask = torch.cat(mask, dim=0)

        return mask, num_low_frequencies

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int],
        num_low_frequencies: int,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Not used.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        if offset is None:
            offset=0

        mask = np.zeros(num_cols, dtype=np.float32)
        mask[offset::acceleration] = 1

        return mask


class FixedLowGaussianMaskFunc(MaskFunc):
    """
    Sample data with Gaussian k-t sampling pattern and a fixed number of low-frequency lines.
    Based on the ktGaussianSampling.m implementation.
    """

    def __init__(self,
                 center_fractions: Sequence[int],
                 accelerations: Sequence[int],
                 allow_any_combination: bool = True,
                 alpha: float = 0.28,
                 sigma_factor: float = 5.0,  # Matlab中sig = ny/5
                 seed: Optional[int] = None):
        super().__init__(center_fractions, accelerations, allow_any_combination, seed)
        self.alpha = alpha
        self.sigma_factor = sigma_factor

    def sample_kt_mask(self, shape, offset, num_adj_slices, slice_idx, num_t, num_slc, rng, seed=None, acc=None):
        if not hasattr(self, 'start_adj'):
            self.start_adj, self.end_adj = -(num_adj_slices // 2), num_adj_slices // 2 + 1

        # Get mask parameters
        num_low_frequencies = rng.choice(self.center_fractions)
        if acc:
            acceleration = int(acc)
        else:
            acceleration = rng.choice(self.accelerations)

        # Get dimensions
        nx, ny = shape[-3], shape[-2]

        # Determine temporal index
        if seed is None:  # Training
            ti = rng.randint(num_t)
        else:  # Validation
            ti = slice_idx // num_slc

        # Generate k-t Gaussian mask
        kt_mask = self._generate_kt_gaussian_mask(
            nx, ny, num_t, num_low_frequencies, acceleration,
            self.alpha, self.sigma_factor, rng
        )

        # Select adjacent temporal frames
        select_list = self._get_ti_adj_idx_list(ti, num_t)

        # Extract selected frames and reshape
        mask_list = []
        for t_idx in select_list:
            frame_mask = kt_mask[:, :, t_idx]  # [nx, ny]

            # Convert to tensor and transpose
            frame_tensor = torch.from_numpy(frame_mask.T.astype(np.float32))  # [ny, nx]

            # Add batch dimensions to match target shape
            target_dims = len(shape)
            while frame_tensor.dim() < target_dims:
                frame_tensor = frame_tensor.unsqueeze(0)

            mask_list.append(frame_tensor)

        mask = torch.cat(mask_list, dim=0)
        mask = mask.permute(0, 3, 2, 1)
        return mask.float(), int(num_low_frequencies)

    def _generate_kt_gaussian_mask(self, nx, ny, nt, ncalib, R, alpha, sigma_factor, rng):
        """k-t高斯采样"""
        # Phase encoding positions - 与Matlab一致
        p1 = np.arange(-np.floor(ny / 2), np.ceil(ny / 2), dtype=int)

        # Temporal resolution
        tr = int(np.round(ny / R))

        # Initialize arrays
        ti = np.zeros(tr * nt, dtype=int)
        ph = np.zeros(tr * nt, dtype=int)

        # Gaussian envelope - 与Matlab完全一致
        sig = ny / sigma_factor
        prob = 0.1 + alpha / (1 - alpha + 1e-10) * np.exp(-(p1 ** 2) / (1 * sig ** 2))

        # Generate temporal sampling pattern
        t1 = []
        ind = 0

        # 使用与Matlab相同的随机种子生成方式
        tmp_seeds = (1e6 * rng.random(nt)).astype(int)

        # 时间循环范围与Matlab一致
        for i in range(-int(np.floor(nt / 2)), int(np.ceil(nt / 2))):
            # Find existing samples at this temporal position
            a_indices = [idx for idx, val in enumerate(t1) if val == i]
            n_tmp = tr - len(a_indices)

            if n_tmp > 0:
                # Create probability distribution
                prob_tmp = prob.copy()
                # 正确处理已采样位置的概率置零
                for a_idx in a_indices:
                    if a_idx < len(prob_tmp):
                        prob_tmp[a_idx] = 0

                # 使用正确的种子索引
                seed_idx = i + int(np.floor(nt / 2))
                if 0 <= seed_idx < len(tmp_seeds):
                    frame_rng = np.random.RandomState(tmp_seeds[seed_idx])
                    p_tmp = self._weighted_random_sample(prob_tmp, frame_rng, n_tmp, p1)

                    # Store indices
                    ti[ind:ind + n_tmp] = i
                    ph[ind:ind + n_tmp] = p_tmp
                    ind += n_tmp

                    # Update t1
                    t1.extend([i] * n_tmp)

        # Remove duplicates (ktdup equivalent)
        ti_valid, ph_valid = self._remove_kt_duplicates(ph[:ind], ti[:ind], ny, nt)

        # Create sampling pattern
        samp = np.zeros((ny, nt))

        # 索引计算与Matlab一致
        for t_idx, p_idx in zip(ti_valid, ph_valid):
            t_pos = int(t_idx + np.floor(nt / 2))
            p_pos = int(p_idx + np.floor(ny / 2))

            if 0 <= t_pos < nt and 0 <= p_pos < ny:
                samp[p_pos, t_pos] = 1

        # Add ACS - 与Matlab zpad函数一致
        acs = self._create_acs_mask_matlab_style(nx, ny, nt, ncalib)

        # 与Matlab的permute和repmat操作一致
        ktus = np.tile(samp[np.newaxis, :, :], (nx, 1, 1))

        # Combine masks
        mask_temp = ktus + acs
        mask = (mask_temp > 0).astype(np.float32)

        return mask

    def _create_acs_mask_matlab_style(self, nx, ny, nt, ncalib):
        """创建与Matlab zpad函数一致的ACS mask"""
        # Matlab中的ACS是 zpad(ones(nx,ncalib,nt),[nx,ny,nt])
        acs_core = np.ones((nx, ncalib, nt))
        acs = np.zeros((nx, ny, nt))

        # 计算填充位置
        start_y = (ny - ncalib) // 2
        end_y = start_y + ncalib

        acs[:, start_y:end_y, :] = acs_core
        return acs

    def _weighted_random_sample(self, prob, rng, n_samples, positions):
        """
        加权随机采样，等价于Matlab的randp函数
        """
        # 处理概率为0的情况
        if np.sum(prob) == 0:
            # 如果所有概率都是0，均匀采样
            sampled_indices = rng.choice(len(positions), size=n_samples, replace=True)
        else:
            # 归一化概率
            prob_norm = prob / np.sum(prob)

            # 基于概率分布采样
            sampled_indices = rng.choice(
                len(positions),
                size=n_samples,
                p=prob_norm,
                replace=True
            )

        return positions[sampled_indices]

    def _remove_kt_duplicates(self, ph, ti, ny, nt):
        """
        移除重复的(phase, time)对 - 等价于ktdup函数
        """
        # 实现ktdup算法
        # 转换到正索引
        ph_shifted = ph + int(np.ceil((ny + 1) / 2))
        ti_shifted = ti + int(np.ceil((nt + 1) / 2))

        # 计算线性索引
        pt = (ti_shifted - 1) * ny + ph_shifted

        # 找到重复值
        unique_pt, unique_indices = np.unique(pt, return_index=True)

        # 如果有重复，需要重新分配
        if len(unique_pt) < len(pt):
            # 获取空闲位置
            all_positions = set(range(1, ny * nt + 1))
            occupied_positions = set(unique_pt)
            empty_positions = list(all_positions - occupied_positions)

            # 重新分配重复位置
            pt_corrected = pt.copy()
            used_indices = set(unique_indices)

            empty_idx = 0
            for i in range(len(pt)):
                if i not in used_indices and empty_idx < len(empty_positions):
                    # 找到最近的空位置
                    current_pos = pt[i]
                    distances = [abs(pos - current_pos) for pos in empty_positions]
                    nearest_idx = np.argmin(distances)
                    pt_corrected[i] = empty_positions.pop(nearest_idx)

            pt = pt_corrected

        # 转换回phase和time索引
        ph_final = ((pt - 1) % ny + 1) - int(np.ceil((ny + 1) / 2))
        ti_final = np.ceil(pt / ny) - int(np.ceil((nt + 1) / 2))

        return ti_final.astype(int), ph_final.astype(int)


class FixedLowRadialMaskFunc(MaskFunc):
    """径向采样"""

    def __init__(self,
                 center_fractions: Sequence[int],
                 accelerations: Sequence[int],
                 allow_any_combination: bool = True,
                 angle4next: float = 180.0,
                 crop_corner: bool = True,
                 seed: Optional[int] = None):
        super().__init__(center_fractions, accelerations, allow_any_combination, seed)
        self.angle4next = angle4next
        self.crop_corner = crop_corner

    def sample_kt_mask(self,
                       shape: Sequence[int],
                       offset=None,
                       num_adj_slices=None,
                       slice_idx=None,
                       num_t=None,
                       num_slc=None,
                       rng=None,
                       seed: Optional[Union[int, Tuple[int, ...]]] = None,
                       acc=None) -> Tuple[torch.Tensor, int]:
        """
        Sample a k-t radial mask.
        """
        if len(shape) < 3:
            raise ValueError(f"Shape must have at least 3 dimensions for k-t sampling, got {len(shape)}")

        # Extract spatial and temporal dimensions
        *batch_dims, nx, ny, nt = shape

        # Set random seed if provided — use local RandomState to avoid corrupting global numpy state
        if seed is not None:
            if isinstance(seed, (tuple, list)):
                seed = sum(seed) % (2 ** 32)
            used_rng = np.random.RandomState(seed)
        else:
            # Use provided rng or self.rng
            if rng is not None:
                used_rng = rng
            else:
                used_rng = self.rng

        if acc:
            acceleration = int(acc)
        else:
            acceleration = rng.choice(self.accelerations)
        # Choose random acceleration and center fraction
        if self.allow_any_combination:
            # acceleration = used_rng.choice(self.accelerations)
            center_fraction = used_rng.choice(self.center_fractions)
        else:
            choice = used_rng.randint(0, len(self.accelerations))
            # acceleration = self.accelerations[choice]
            center_fraction = self.center_fractions[choice]

        # Calculate number of calibration lines
        ncalib = int(center_fraction)

        # Use num_t parameter for temporal dimension, not nt from shape
        # (shape[-1] is 2 for complex real/imag, not the actual temporal frames)
        actual_nt = num_t if num_t is not None else nt

        # Generate the k-t radial mask with correct temporal dimension
        mask_3d = self._generate_kt_radial_mask(
            nx=nx,
            ny=ny,
            nt=actual_nt,
            ncalib=ncalib,
            R=acceleration,
            angle4next=self.angle4next,
            crop_corner=self.crop_corner
        )

        # Handle temporal selection if needed
        if num_adj_slices is not None and slice_idx is not None and num_t is not None and num_slc is not None:
            if not hasattr(self, 'start_adj'):
                self.start_adj, self.end_adj = -(num_adj_slices // 2), num_adj_slices // 2 + 1

            # Determine temporal index
            if seed is None:  # Training
                ti = used_rng.randint(actual_nt)
            else:  # Validation
                ti = slice_idx // num_slc

            # Get adjacent indices with proper bounds checking
            select_list = self._get_ti_adj_idx_list_safe(ti, actual_nt, num_adj_slices)

            # Extract selected frames
            mask_list = []
            for t_idx in select_list:
                # Add bounds checking
                if 0 <= t_idx < actual_nt:
                    frame_mask = mask_3d[:, :, t_idx]  # [nx, ny]
                else:
                    # Handle out-of-bounds by using the closest valid frame
                    t_idx_safe = max(0, min(t_idx, actual_nt - 1))
                    frame_mask = mask_3d[:, :, t_idx_safe]

                # Convert to tensor and transpose
                frame_tensor = torch.from_numpy(frame_mask.T.astype(np.float32))  # [ny, nx]

                # Add batch dimensions to match target shape
                target_dims = len(shape)
                while frame_tensor.dim() < target_dims:
                    frame_tensor = frame_tensor.unsqueeze(0)

                mask_list.append(frame_tensor)

            mask_tensor = torch.cat(mask_list, dim=0)
        else:
            # No temporal selection, use full mask
            if batch_dims:
                # Add batch dimensions
                mask_shape = tuple(batch_dims) + mask_3d.shape
                mask = np.broadcast_to(mask_3d[None, ...], mask_shape)
            else:
                mask = mask_3d

            # Convert to tensor
            mask_tensor = torch.from_numpy(mask.astype(np.float32))

            # Add complex dimension if needed (for complex data)
            if mask_tensor.dim() == len(shape):
                mask_tensor = mask_tensor.unsqueeze(-1)  # Add complex dimension

        mask_tensor = mask_tensor.permute(0, 3, 2, 1)
        return mask_tensor.float(), ncalib

    def _get_ti_adj_idx_list_safe(self, ti, actual_nt, num_adj_slices):
        """
        Get the circular adjacent indices with proper bounds checking.
        """
        if not hasattr(self, 'start_adj'):
            self.start_adj, self.end_adj = -(num_adj_slices // 2), num_adj_slices // 2 + 1

        start_lim, end_lim = -(actual_nt // 2), (actual_nt // 2 + 1)
        start, end = max(self.start_adj, start_lim), min(self.end_adj, end_lim)

        # Generate initial list of indices with proper modulo
        ti_idx_list = [(i + ti) % actual_nt for i in range(start, end)]

        # Handle padding with bounds checking
        replication_prefix = max(start_lim - self.start_adj, 0) * [ti_idx_list[0]] if ti_idx_list else []
        replication_suffix = max(self.end_adj - end_lim, 0) * [ti_idx_list[-1]] if ti_idx_list else []

        return replication_prefix + ti_idx_list + replication_suffix
    def _generate_kt_radial_mask(self, nx, ny, nt, ncalib, R, angle4next, crop_corner):
        """径向mask生成"""
        # Calculate sampling rate and number of beams
        rate = 1.0 / R
        beams = int(np.floor(rate * 180))

        # Determine auxiliary matrix size
        if crop_corner:
            a = max(nx, ny)
        else:
            a = int(np.ceil(np.sqrt(2) * max(nx, ny)))

        # Create base radial line
        aux = np.zeros((a, a), dtype=np.float32)
        aux[int(np.round(a / 2)), :] = 1  # 水平线通过中心

        # Calculate angle step
        angle_step = 180.0 / beams

        # Initialize k-t undersampling mask
        ktus = np.zeros((nx, ny, nt), dtype=np.float32)

        # 时间循环从1开始（与Matlab一致）
        for i in range(nt):
            # 角度计算与Matlab完全一致
            start_angle = angle4next * i  # Matlab: angle4next*(i-1), 但i从1开始
            angles = np.arange(start_angle, start_angle + 180, angle_step)

            # Initialize frame mask
            frame_mask = np.zeros((nx, ny), dtype=np.float32)

            # Add each radial line
            for angle in angles:
                # 需要添加crop函数的Python实现
                rotated_line = self._matlab_crop(
                    self._matlab_imrotate(aux, angle), nx, ny
                )
                frame_mask += rotated_line

            ktus[:, :, i] = (frame_mask > 0).astype(np.float32)

        # ACS创建与Matlab一致
        acs = self._create_radial_acs_mask(nx, ny, nt, ncalib)

        # Combine masks
        mask_temp = ktus + acs
        mask = (mask_temp > 0).astype(np.float32)

        return mask

    def _matlab_imrotate(self, image, angle):
        """模拟Matlab的imrotate函数"""
        from scipy.ndimage import rotate
        # Matlab的imrotate默认是双线性插值，但对于mask我们使用最近邻
        return rotate(image, angle, reshape=False, order=0, mode='constant', cval=0)

    def _matlab_crop(self, image, target_nx, target_ny):
        """模拟Matlab的crop函数"""
        h, w = image.shape

        # 计算裁剪起始位置（从中心裁剪）
        start_x = int(np.floor(h / 2)) + 1 + int(np.ceil(-target_nx / 2))
        end_x = int(np.floor(h / 2)) + int(np.ceil(target_nx / 2))
        start_y = int(np.floor(w / 2)) + 1 + int(np.ceil(-target_ny / 2))
        end_y = int(np.floor(w / 2)) + int(np.ceil(target_ny / 2))

        # 边界检查
        start_x = max(0, start_x)
        end_x = min(h, end_x)
        start_y = max(0, start_y)
        end_y = min(w, end_y)

        # 创建输出图像
        result = np.zeros((target_nx, target_ny), dtype=image.dtype)

        # 计算实际裁剪区域
        actual_nx = end_x - start_x
        actual_ny = end_y - start_y

        if actual_nx > 0 and actual_ny > 0:
            # 计算在结果中的位置
            result_start_x = max(0, (target_nx - actual_nx) // 2)
            result_start_y = max(0, (target_ny - actual_ny) // 2)

            result[result_start_x:result_start_x + actual_nx,
            result_start_y:result_start_y + actual_ny] = \
                image[start_x:end_x, start_y:end_y]

        return result

    def _create_radial_acs_mask(self, nx, ny, nt, ncalib):
        """创建径向采样的ACS mask"""
        # Matlab中径向采样的ACS是 zpad(ones(ncalib,ncalib,nt),[nx,ny,nt])
        acs_core = np.ones((ncalib, ncalib, nt))
        acs = np.zeros((nx, ny, nt))

        # 计算中心位置
        start_x = (nx - ncalib) // 2
        end_x = start_x + ncalib
        start_y = (ny - ncalib) // 2
        end_y = start_y + ncalib

        acs[start_x:end_x, start_y:end_y, :] = acs_core
        return acs

class CmrxRecon25MaskFunc(MaskFunc):
    def __init__(
        self,
        num_low_frequencies: Sequence[int],
        num_adj_slices: int,
        seed: Optional[int] = None
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """

        self.uniform_mask = FixedLowEquiSpacedMaskFunc(num_low_frequencies, [8,16,24], allow_any_combination=True, seed=seed )
        # self.kt_uniform_mask = FixedLowEquiSpacedMaskFunc(num_low_frequencies, [8,16,24], allow_any_combination=True, seed=seed )
        # self.kt_random_mask = FixedLowRandomMaskFunc(num_low_frequencies, [8,16,24], allow_any_combination=True, seed=seed )
        self.kt_gaussian_mask = FixedLowGaussianMaskFunc(num_low_frequencies, [8, 16, 24], allow_any_combination=True, seed=seed, alpha=0.28, sigma_factor=5.0)
        self.kt_radial_mask = FixedLowRadialMaskFunc(num_low_frequencies, [8, 16, 24], allow_any_combination=True, seed=seed, angle4next=180.0, crop_corner=True)
        # mask_dict is set according to cmrxrecon24 challenge settings
        self.mask_dict = {'uniform':[8,16,24],
                           # 'kt_uniform':[8,16,24],
                           # 'kt_random':[8,16,24],
                           'kt_radial':[8,16,24],
                           'kt_gaussian':[8,16,24]}
        self.masks_pool = list(self.mask_dict.keys())

        self.rng = np.random.RandomState(seed)

        self.num_adj_slices = num_adj_slices
        self.start_adj, self.end_adj = -(num_adj_slices//2), num_adj_slices//2+1

    def choose_mask(self):
        '''
        choose from FixedLowEquiSpacedMaskFunc, FixedLowRandomMaskFunc and radial
        '''
        mask_type = self.rng.choice(self.masks_pool)
        return mask_type

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        seed: Optional[Union[int, Tuple[int, ...]]] = None,
        slice_idx: Optional[int] = None,
        num_t: Optional[int] = None,
        num_slc: Optional[int] = None,
        mask_type: Optional[str] = None,
        acc: Optional[str] = None
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks). If no
                offset is given, then one is selected randomly.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number of
            center frequency lines.
        """
        if len(shape) < 3:
            raise ValueError("Shape should have 3 or more dimensions")
        self.seed = seed
        with temp_seed(self.rng, seed):
            # mask_type = self.choose_mask()
            mask, num_low_frequencies = self.sample_mask(mask_type, shape, offset, slice_idx, num_t, num_slc, acc)

        return mask, num_low_frequencies, mask_type

    def sample_mask(self,mask_type, shape,offset=None,  slice_idx=None,num_t=None,num_slc=None, acc=None):
        
        if mask_type=='uniform':
            mask, num_low_frequencies = self.uniform_mask.sample_uniform_mask(shape, offset, self.rng, acc) #, self.seed)
        # elif mask_type=='kt_uniform':
        #     mask, num_low_frequencies = self.kt_uniform_mask.sample_kt_mask(shape, offset, self.num_adj_slices, slice_idx, num_t,num_slc, self.rng, self.seed)
        # elif mask_type=='kt_random':
        #     mask, num_low_frequencies = self.kt_random_mask.sample_kt_mask(shape, offset, self.num_adj_slices, slice_idx, num_t,num_slc, self.rng)
        elif mask_type == 'kt_gaussian':
            mask, num_low_frequencies = self.kt_gaussian_mask.sample_kt_mask(shape, offset, self.num_adj_slices, slice_idx, num_t, num_slc, self.rng, self.seed, acc)
        elif mask_type == 'kt_radial':
            mask, num_low_frequencies = self.kt_radial_mask.sample_kt_mask(shape, offset, self.num_adj_slices, slice_idx, num_t, num_slc, self.rng, self.seed, acc)
        else:
            raise ValueError(f"{mask_type} not supported")

        return mask.float(), num_low_frequencies

class CmrxRecon25TestValMaskFunc(CmrxRecon25MaskFunc):
    """
    Sample data 

    """
    def __init__(
        self,
        num_low_frequencies: Sequence[int],
        num_adj_slices: int,
        seed: Optional[int] = None,
        test_mask_type: str = 'uniform',
        test_acc: int = 10
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """

        self.uniform_mask = FixedLowEquiSpacedMaskFunc(num_low_frequencies, [test_acc], allow_any_combination=True, seed=seed )
        self.kt_uniform_mask = FixedLowEquiSpacedMaskFunc(num_low_frequencies, [test_acc], allow_any_combination=True, seed=seed )
        self.kt_random_mask = FixedLowRandomMaskFunc(num_low_frequencies, [test_acc], allow_any_combination=True, seed=seed )
        self.kt_gaussian_mask = FixedLowGaussianMaskFunc(num_low_frequencies, [test_acc], allow_any_combination=True, seed=seed, alpha=0.28, sigma_factor=5.0)
        self.kt_radial_mask = FixedLowRadialMaskFunc(num_low_frequencies, [test_acc], allow_any_combination=True, seed=seed, angle4next=180.0, crop_corner=True)

        # mask_dict is set according to test config
        self.mask_dict = {test_mask_type:[test_acc]}
        self.masks_pool = list(self.mask_dict.keys())

        self.rng = np.random.RandomState(seed)

        self.num_adj_slices = num_adj_slices
        self.start_adj, self.end_adj = -(num_adj_slices//2), num_adj_slices//2+1