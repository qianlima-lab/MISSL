
from .negative_samplers import negative_sampler_factory

from abc import *


class AbstractDataloader(metaclass=ABCMeta):
    def __init__(self, dataset, val_negative_sampler_code, val_negative_sample_size, dataAug=False):
        save_folder = dataset._get_preprocessed_folder_path()
        dataset = dataset.load_dataset()
        if dataAug:
            self.tUser = dataset['tUser']
            self.vUser = dataset['vUser']
            self.train = dataset['train']
            self.val = dataset['val']
            self.train_b = dataset['train_b']
            self.val_b = dataset['val_b']
            self.train_l = dataset['train_l']
            self.val_l = dataset['val_l']
            self.train_pos_item = dataset['train_pos_item']
            self.train_pos_beh = dataset['train_pos_beh']
            self.val_pos_item = dataset['val_pos_item']
            self.val_pos_beh = dataset['val_pos_beh']

            self.Train = dataset['Train']
            self.Val = dataset['Val']

            self.umap = dataset['umap']
            self.smap = dataset['smap']
            self.bmap = dataset['bmap']
            self.user_count = len(self.umap)
            self.item_count = len(self.smap)
            self.behavior_count = len(self.bmap)

            val_negative_sampler = negative_sampler_factory(val_negative_sampler_code, self.Train, self.Val,
                                                            self.user_count, self.item_count,
                                                            val_negative_sample_size,
                                                            save_folder)
            self.val_negative_samples = val_negative_sampler.get_negative_samples()
        else:
            self.train = dataset['train']
            self.val = dataset['val']
            self.train_b = dataset['train_b']
            self.val_b = dataset['val_b']
            self.val_num = dataset['val_num']
            self.umap = dataset['umap']
            self.smap = dataset['smap']
            self.bmap = dataset['bmap']
            self.user_count = len(self.umap)
            self.item_count = len(self.smap)
            self.behavior_count = len(self.bmap)

            val_negative_sampler = negative_sampler_factory(val_negative_sampler_code, self.train, self.val,
                                                             self.user_count, self.item_count,
                                                             val_negative_sample_size,
                                                             save_folder)
            self.val_negative_samples = val_negative_sampler.get_negative_samples()


    @abstractmethod
    def get_train_loader(self):
        pass

    @abstractmethod
    def get_val_loader(self):
        pass