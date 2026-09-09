"""
Dataset classes for fastMRI, Calgary-Campinas, CMRxRecon datasets
"""
import logging
import os
import pickle
import random
import xml.etree.ElementTree as etree
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)
import h5py
import numpy as np
import torch
import torch.utils

from mri_utils.utils import load_shape, load_kdata, load_mask

#########################################################################################################
# Common functions
#########################################################################################################


class RawDataSample(NamedTuple):
    """
    A container for raw data samples.
    """
    fname: Path
    slice_ind: int
    metadata: Dict[str, Any]
    mask_type: str
    acc: str


class BalanceSampler:
    def __init__(self, ratio_dict={'a':1, 'b':2}):
        self.ratio_dict = ratio_dict
        
    def __call__(self, raw_sample: List[RawDataSample]):
        # create dict, keys with empty list
        dict_list = {key: [] for key in self.ratio_dict.keys()}
        
        # for key, value in self.ratio_dict.items():
        for raw_i in raw_sample:
            for key in dict_list.keys():
                if key in str(raw_i.fname):
                    dict_list[key].append(raw_i)
                    break
        # combine to final list multiply with ratio 
        final_list = []
        for key, value in self.ratio_dict.items():
            final_list += dict_list[key] * value

        return final_list
            

class FuncFilterString:
    """
    A callable class to filter samples based on a string in their 'fname'.

    Args:
        filter_str (str): The string to filter by. Defaults to None (allow all samples).
    """

    def __init__(self, filter_str: Optional[Union[str, List[str]]] = None, logic: str = 'or'):
        """
        Initializes the filter string.
        """
        assert logic in ['or', 'and'], f"Invalid logic: {logic}"
        self.filter_str = filter_str
        self.logic = logic

    def __call__(self, raw_sample: RawDataSample) -> bool:
        """
        Filters the raw_sample based on the filter_str.

        Args:
            raw_sample (dict): A dictionary containing metadata for the raw sample.

        Returns:
            bool: True if the filter_str is in raw_sample["fname"], or if filter_str is None.
        """
        if isinstance(raw_sample, RawDataSample):
            fname = str(raw_sample.fname)
        elif isinstance(raw_sample, str):
            fname = raw_sample
        else:
            assert False, f"Invalid raw_sample type: {type(raw_sample)}"
        
        if self.filter_str is None:
            return True
        elif isinstance(self.filter_str, str):
            return self.filter_str in fname
        elif isinstance(self.filter_str, list):
            if self.logic == 'or':
                return any(filter_str in fname for filter_str in self.filter_str)
            elif self.logic == 'and':
                return all(filter_str in fname for filter_str in self.filter_str)
        else:
            raise ValueError(f"Invalid filter_str: {self.filter_str}")
        
class CombinedSliceDataset(torch.utils.data.Dataset):
    """
    A container for combining slice datasets.
    """

    def __init__(
        self,
        slice_dataset: torch.utils.data.Dataset,
        roots: Sequence[Path],
        transforms: Optional[Sequence[Optional[Callable]]] = None,
        sample_rates: Optional[Sequence[Optional[float]]] = None,
        volume_sample_rates: Optional[Sequence[Optional[float]]] = None,
        use_dataset_cache: bool = False,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        num_adj_slices: int = 5,
        data_balancer: Optional[Callable] = None,
    ):
        """
        Args:
            roots: Paths to the datasets.
            transforms: Optional; A sequence of callable objects that
                preprocesses the raw data into appropriate form. The transform
                function should take 'kspace', 'target', 'attributes',
                'filename', and 'slice' as inputs. 'target' may be null for
                test data.
            sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the slices should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            volume_sample_rates: Optional; A sequence of floats between 0 and 1.
                This controls what fraction of the volumes should be loaded.
                When creating subsampled datasets either set sample_rates
                (sample by slices) or volume_sample_rates (sample by volumes)
                but not both.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """
        if sample_rates is not None and volume_sample_rates is not None:
            raise ValueError(
                "either set sample_rates (sample by slices) or volume_sample_rates (sample by volumes) but not both"
            )
        if transforms is None:
            transforms = [None] * len(roots)
        if sample_rates is None:
            sample_rates = [None] * len(roots)
        if volume_sample_rates is None:
            volume_sample_rates = [None] * len(roots)
        if not (
            len(roots)
            == len(transforms)
            == len(sample_rates)
            == len(volume_sample_rates)
        ):
            raise ValueError(
                "Lengths of roots, transforms, sample_rates do not match"
            )

        self.datasets = []
        self.raw_samples: List[RawDataSample] = []

        for i, root_i in enumerate(roots):
            self.datasets.append(
                slice_dataset(
                    root=root_i,
                    transform=transforms[i],
                    sample_rate=sample_rates[i],
                    volume_sample_rate=volume_sample_rates[i],
                    use_dataset_cache=use_dataset_cache,
                    dataset_cache_file=dataset_cache_file,
                    num_cols=num_cols,
                    raw_sample_filter=raw_sample_filter,
                    data_balancer = data_balancer,
                    num_adj_slices=num_adj_slices
                )
            )

            self.raw_samples = self.raw_samples + self.datasets[-1].raw_samples

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)

    def __getitem__(self, i):
        for dataset in self.datasets:
            if i < len(dataset):
                return dataset[i]
            else:
                i = i - len(dataset)

#########################################################################################################
# CMRxRecon dataset
#########################################################################################################


class CmrxReconSliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for the CMRxRecon 2023 & 2024 challenge.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        data_balancer: Optional[Callable] = None,
        num_adj_slices: int = 5,
    ):
        """
        Args:
            root: Path to the dataset.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """
        self.root = root
        if 'train' in str(root):
            self._split = 'train'
        elif 'val' in str(root):
            self._split = 'val'
        else:
            self._split = 'test'

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform

        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd in SliceDataset"
        # max temporal slice number is 12
        assert num_adj_slices <= 11, "Number of adjacent slices must be less than 11 in CMRxRecon SliceDataset"
        self.num_adj_slices = num_adj_slices
        self.start_adj, self.end_adj = -(self.num_adj_slices//2), self.num_adj_slices//2+1

        self.recons_key = "reconstruction_rss"
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:

            files = list(Path(root).iterdir())

            for fname in sorted(files):
                with h5py.File(fname, 'r') as hf:
                    # print('load debug: ', fname, hf.keys())
                    num_slices = hf["kspace"].shape[0]*hf["kspace"].shape[1]
                    metadata = {**hf.attrs}
                new_raw_samples = []
                if root.name == 'train':
                    for slice_ind in range(num_slices):
                        for mask_type in ['uniform', 'kt_gaussian', 'kt_radial']:
                            if mask_type == 'kt_radial':
                                accs = random.sample(['8', '16', '24'], 1)
                                for acc in accs:
                                    raw_sample = RawDataSample(fname, slice_ind, metadata, mask_type, acc)
                            else:
                                raw_sample = RawDataSample(fname, slice_ind, metadata, mask_type, acc=None)
                            if self.raw_sample_filter(raw_sample):
                                new_raw_samples.append(raw_sample)
                elif root.name == 'val':
                    for slice_ind in range(num_slices):
                        for mask_type in ['uniform', 'kt_gaussian', 'kt_radial']:
                            for acc in ['8', '16', '24']:
                                raw_sample = RawDataSample(fname, slice_ind, metadata, mask_type, acc)
                                if self.raw_sample_filter(raw_sample):
                                    new_raw_samples.append(raw_sample)


                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(
                    "Saving dataset cache to %s.", self.dataset_cache_file)
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(
                "Using dataset cache from %s.", self.dataset_cache_file)
            self.raw_samples = dataset_cache[root]

        if 'train' in str(root) and data_balancer is not None:
            self.raw_samples = data_balancer(self.raw_samples)

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(
                list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]

    def _get_ti_adj_idx_list(self, ti, num_t_in_volume):
        '''
        get the circular adjacent indices of the temporal axis for the given ti.
        '''
        start_lim, end_lim = -(num_t_in_volume//2), (num_t_in_volume//2+1)
        start, end = max(self.start_adj, start_lim), min(self.end_adj, end_lim)
        # Generate initial list of indices
        ti_idx_list = [(i + ti) % num_t_in_volume for i in range(start, end)]
        # duplicate padding if necessary
        replication_prefix = max(start_lim-self.start_adj, 0) * ti_idx_list[0:1]
        replication_suffix = max(self.end_adj-end_lim, 0) * ti_idx_list[-1:]

        return replication_prefix + ti_idx_list + replication_suffix

    def __len__(self):
        return len(self.raw_samples)

    def __getitem__(self, i: int):
        fname, data_slice, metadata, mask_type, acc = self.raw_samples[i]
        kspace = []
        with h5py.File(str(fname), 'r') as hf:
            kspace_volume = hf["kspace"]
            attrs = dict(hf.attrs)
            num_t = attrs['shape'][0]
            num_slices = attrs['shape'][1]
            ti = data_slice//num_slices
            zi = data_slice - ti*num_slices

            mask = np.asarray(hf["mask"]) if "mask" in hf else None
            target = hf[self.recons_key][ti,zi] if self.recons_key in hf else None

            ti_idx_list = self._get_ti_adj_idx_list(ti, num_t)

            for idx in ti_idx_list:
                kspace.append(kspace_volume[idx, zi])
            kspace = np.concatenate(kspace, axis=0)
            
        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, data_slice, num_t, mask_type, acc)
        else:
            sample = self.transform(kspace, mask, target, attrs, fname.name, data_slice, num_t, num_slices, mask_type, acc)
        return sample


class CmrxReconInferenceSliceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        raw_sample_filter: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        num_adj_slices: int = 5
    ):
        self.root = root
        # get all the kspace mat files from root, under folder or its subfolders
        volume_paths = root.glob('**/*.mat')

        if '2023' in str(self.root):
            self.year = 2023 
        elif '2024' in str(self.root):
            self.year = 2024
        else:
            raise ValueError('Invalid dataset root')
        #
        if self.year == 2023:
            # filter out files contains '_mask.mat'
            self.volume_paths = [str(path) for path in volume_paths if '_mask.mat' not in str(path)]
            
        elif self.year == 2024:
            self.volume_paths = [str(path) for path in volume_paths if '_mask_' not in str(path)]
        
        self.volume_paths = [pp for pp in self.volume_paths if raw_sample_filter(pp)]
        print('number of inference paths: ', len(self.volume_paths))
            

        self.transform = transform
        
        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd."
        assert num_adj_slices <= 11, "Number of adjacent slices must be <= 11."
        self.num_adj_slices = num_adj_slices
        self.start_adj = -(num_adj_slices // 2)
        self.end_adj = num_adj_slices // 2 + 1
        self.volume_shape_dict = self._get_volume_shape_info()
        # add the fisrt element in each dict
        self.len_dataset = sum([v[0]*v[1] for v in self.volume_shape_dict.values()])
        
        
        self.current_volume = None
        self.current_file_index = -1
        self.current_num_slices = None
        self.slices_offset = 0  # Track the starting index of the slices in the current volume

        # New attributes
        self.index_to_volume_idx = {}
        self.index_to_slice_idx = {}
        self.volume_start_indices = []
        self.volume_indices = []  # Add this line

        self.current_volume = None
        self.current_volume_index = None

        self._build_index_mappings()

    def _build_index_mappings(self):
        global_idx = 0
        for volume_idx, path in enumerate(self.volume_paths):
            shape = self.volume_shape_dict[path]
            num_slices = shape[0] * shape[1]
            self.volume_start_indices.append(global_idx)

            volume_indices = []
            for slice_idx in range(num_slices):
                self.index_to_volume_idx[global_idx] = volume_idx
                self.index_to_slice_idx[global_idx] = slice_idx
                volume_indices.append(global_idx)
                global_idx += 1
            self.volume_indices.append(volume_indices)

        self.len_dataset = global_idx  # Update dataset length
        
    def _get_volume_shape_info(self):
        shape_dict = {} #defaultdict(dict)
        for path in self.volume_paths:
            shape_dict[path]=load_shape(path)
        return shape_dict
 
    def _get_ti_adj_idx_list(self, ti, num_t_in_volume):
        """
        Get circular adjacent indices for temporal axis.
        """
        start_lim, end_lim = -(num_t_in_volume // 2), (num_t_in_volume // 2 + 1)
        start, end = max(self.start_adj, start_lim), min(self.end_adj, end_lim)
        ti_idx_list = [(i + ti) % num_t_in_volume for i in range(start, end)]

        replication_prefix = max(start_lim - self.start_adj, 0) * ti_idx_list[0:1]
        replication_suffix = max(self.end_adj - end_lim, 0) * ti_idx_list[-1:]

        return replication_prefix + ti_idx_list + replication_suffix
    
    def _load_volume(self, path):
        """
        Load the k-space volume and mask for the given path.
        Modify this function based on your `load_kdata` and `load_mask` functions.
        """
        kspace_volume = load_kdata(path)
        kspace_volume = kspace_volume[None] if len(kspace_volume.shape) != 5 else kspace_volume # blackblood has no time dimension
        kspace_volume = kspace_volume.transpose(0, 1, 2, 4, 3)
        
        if self.year==2023:
            mask_path = path.replace('.mat', '_mask.mat')
            mask = load_mask(mask_path).T[0:1]
            mask=mask[None,:,:,None]
        elif self.year==2024:
            mask_path = path.replace('UnderSample_Task', 'Mask_Task').replace('_kus_', '_mask_')
            if 'UnderSample_Task1' in path:
                mask = load_mask(mask_path).T[0:1]
                mask=mask[None,:,:,None]
            else:
                mask = load_mask(mask_path).transpose(0,2,1)
                mask=mask[:,:,:,None]

        attrs = {
            'encoding_size': [kspace_volume.shape[3], kspace_volume.shape[4], 1],
            'padding_left': 0,
            'padding_right': kspace_volume.shape[-1],
            'recon_size': [kspace_volume.shape[3], kspace_volume.shape[4], 1],
        }
        return kspace_volume, mask, attrs
    
    def _load_next_volume(self):
        """Loads the next volume in the dataset."""
        self.current_file_index += 1
        if self.current_file_index < len(self.volume_paths):
            self.current_path = self.volume_paths[self.current_file_index]
            self.current_volume, self.mask, self.attrs = self._load_volume(self.current_path)  # Shape: (D, H, W)
            self.current_num_t = self.current_volume.shape[0]
            self.current_num_z = self.current_volume.shape[1]
            self.current_num_slices = self.current_num_t * self.current_num_z
            self.slices_offset += self.current_num_slices  # Update offset
        else:
            self.current_volume = None
            self.current_num_slices = None
            
    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        volume_idx = self.index_to_volume_idx[idx]
        slice_idx = self.index_to_slice_idx[idx]

        # Load the volume if not already loaded
        if self.current_volume_index != volume_idx:
            self._load_volume_by_index(volume_idx)

        # Compute temporal and spatial indices
        ti = slice_idx // self.current_num_z
        zi = slice_idx % self.current_num_z

        # Get temporal indices
        ti_idx_list = self._get_ti_adj_idx_list(ti, self.current_num_t)

        # Gather k-space data for adjacent slices
        nc = self.current_volume.shape[2]
        kspace = [self.current_volume[idx, zi] for idx in ti_idx_list]
        kspace = np.concatenate(kspace, axis=0)
        
        _path = self.current_path.replace(str(self.root)+'/', '')
        # gather mask data for adjacent slices
        if self.year==2023 or (self.year==2024 and 'UnderSample_Task1' in _path): 
            mask = self.mask
        else:
            mask = [self.mask[idx] for idx in ti_idx_list]
            mask = np.stack(mask, axis=0)
            mask = mask.repeat(nc, axis=0)

        # Prepare the sample
        if self.transform is None:
            sample = (kspace, mask, None, self.attrs, _path, slice_idx, self.current_num_t, self.current_num_z)
        else:
            sample = self.transform(kspace, mask, None, self.attrs, _path, slice_idx, self.current_num_t, self.current_num_z)

        return sample

    def _load_volume_by_index(self, volume_idx):
        self.current_volume_index = volume_idx
        self.current_path = self.volume_paths[volume_idx]
        self.current_volume, self.mask, self.attrs = self._load_volume(self.current_path)
        self.current_num_t = self.current_volume.shape[0]
        self.current_num_z = self.current_volume.shape[1]


        

#########################################################################################################
# Calgary-Campinas dataset
#########################################################################################################

class CalgaryCampinasSliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset for the Calgary-Campinas dataset.
    """

    def __init__(
        self,
        root: Union[str, Path, os.PathLike],
        transform: Optional[Callable] = None,
        use_dataset_cache: bool = False,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        dataset_cache_file: Union[str, Path, os.PathLike] = "dataset_cache.pkl",
        num_cols: Optional[Tuple[int]] = None,
        raw_sample_filter: Optional[Callable] = None,
        data_balancer: Optional[Callable] = None,
        num_adj_slices: int = 5
    ):
        """
        Args:
            root: Path to the dataset.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                'kspace', 'target', 'attributes', 'filename', and 'slice' as
                inputs. 'target' may be null for test data.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the slices should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls what fraction
                of the volumes should be loaded. Defaults to 1 if no value is given.
                When creating a sampled dataset either set sample_rate (sample by slices)
                or volume_sample_rate (sample by volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_cols: Optional; If provided, only slices with the desired
                number of columns will be considered.
            raw_sample_filter: Optional; A callable object that takes an raw_sample
                metadata as input and returns a boolean indicating whether the
                raw_sample should be included in the dataset.
        """
        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd in SliceDataset"
        self.num_adj_slices = num_adj_slices
        self.start_adj, self.end_adj = -(self.num_adj_slices//2), self.num_adj_slices//2+1

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        
        self.data_balancer = data_balancer

        assert num_adj_slices % 2 == 1, "Number of adjacent slices must be odd in SliceDataset"
        self.num_adj_slices = num_adj_slices

        self.recons_key = "reconstruction_rss"
        self.raw_samples = []
        if raw_sample_filter is None:
            self.raw_sample_filter = lambda raw_sample: True
        else:
            self.raw_sample_filter = raw_sample_filter

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())

            for fname in sorted(files):
                with h5py.File(fname, 'r') as hf:
                    num_slices = hf["kspace"].shape[0]
                    metadata = {**hf.attrs}
                new_raw_samples = []

                # * for validation set, only use the middle slices
                if '/val' in str(root):
                    slice_range = range(50, num_slices-50)
                else:
                    slice_range = range(0, num_slices)

                for slice_ind in slice_range:  # range(num_slices):
                    raw_sample = RawDataSample(fname, slice_ind, metadata)
                    if self.raw_sample_filter(raw_sample):
                        new_raw_samples.append(raw_sample)
                self.raw_samples += new_raw_samples

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.raw_samples
                logging.info(
                    "Using dataset cache to %s.", self.dataset_cache_file)
                with open(self.dataset_cache_file, "wb") as cache_f:
                    pickle.dump(dataset_cache, cache_f)
        else:
            logging.info(
                "Using dataset cache from %s.", self.dataset_cache_file)
            self.raw_samples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.raw_samples)
            num_raw_samples = round(len(self.raw_samples) * sample_rate)
            self.raw_samples = self.raw_samples[:num_raw_samples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(
                list(set([f[0].stem for f in self.raw_samples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.raw_samples = [
                raw_sample
                for raw_sample in self.raw_samples
                if raw_sample[0].stem in sampled_vols
            ]

        if num_cols:
            self.raw_samples = [
                ex
                for ex in self.raw_samples
                if ex[2]["encoding_size"][1] in num_cols  # type: ignore
            ]
        print('debug dataset: ', len(self.raw_samples))

    def __len__(self):
        return len(self.raw_samples)

    def _get_frames_indices(self, data_slice, num_slices):
        z_list = [min(max(i+data_slice, 0), num_slices-1)
                  for i in range(self.start_adj, self.end_adj)]
        return z_list

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.raw_samples[i]

        kspace = []
        with h5py.File(fname, "r") as hf:
            num_slices = hf["kspace"].shape[0]
            slice_idx_list = self._get_frames_indices(dataslice, num_slices)
            for slice_idx in slice_idx_list:
                kspace.append(hf["kspace"][slice_idx])
            kspace = np.concatenate(kspace, axis=0)

            mask = np.asarray(hf["mask"]) if "mask" in hf else None

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if self.transform is None:
            sample = (kspace, mask, target, attrs, fname.name, dataslice)
        else:
            sample = self.transform(
                kspace, mask, target, attrs, fname.name, dataslice)

        return sample


