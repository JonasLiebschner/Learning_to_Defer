from abc import ABC, abstractmethod

class BasicDataset(ABC):
 
    @abstractmethod
    def getExpert(self, id):
        pass

    @abstractmethod
    def getData(self):
        pass

    @abstractmethod
    def getDataForLabelers(self, labelerIds):
        pass


class ImageContainer(ABC):
        
    @abstractmethod
    def loadImages(self):
        pass
        
    @abstractmethod       
    def get_image_from_id(self, idx):
        """
        Returns the image from index idx
        """
        pass
            
    @abstractmethod
    def get_image_from_name(self, fname):
        pass
            
    @abstractmethod
    def get_images_from_name(self, fnames):
        pass
            
    @abstractmethod
    def get_images_from_name_np(self, fnames):
        pass


class K_Fold_Dataloader(ABC):

    @abstractmethod
    def get_data_loader_for_fold(self, fold_idx):
        pass

    @abstractmethod
    def get_dataset_for_folder(self, fold_idx):
        pass

    @abstractmethod
    def create_Dataloader_for_Fold(self, idx):
        pass

    @abstractmethod
    def buildDataloaders(self):
        pass

    @abstractmethod
    def getFullDataloader(self):
        pass

    @abstractmethod
    def get_ImageContainer(self):
        pass

    @abstractmethod
    def getData(self):
        pass

class Dataset(ABC):
    """
    """

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def getImage(self, idx):
        """
        Returns the image from index idx
        """
        pass

    @abstractmethod  
    def transformImage(self, image):
        """
        Transforms the image
        """
        pass

    @abstractmethod
    def getTransformedImage(self, image, image_id):
        """
        Transforms the image
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass

    """
    Functions for Verma active learning
    """
    @abstractmethod
    def getAllImagesNP(self):
        """
        Returns all images from the Dataset
        """
        pass

    @abstractmethod
    def getAllImages(self):
        """
        Returns all images from the Dataset
        """
        pass

    @abstractmethod
    def getAllTargets(self):
        """
        Returns all targets
        """
        pass

    @abstractmethod
    def getAllFilenames(self):
        """
        Returns all filenames
        """
        pass

    @abstractmethod
    def getAllIndices(self):
        pass


class DataManager(ABC):
    """
    Class to contain and manage all data for all experiments
    
    This class contains all functions to get data and dataloaders for every step of the experiment
    
    It also implements a k fold
    """
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    @abstractmethod
    def createData(self):
        pass
        
    @abstractmethod   
    def getKFoldDataloader(self, seed):
        pass

    @abstractmethod
    def getSSLDataset(self, seed):
        pass

    @abstractmethod
    def getBasicDataset(self):
        pass


class SSLDataset(ABC):
   
    def set_seed(self, seed, fold=None, text=None):
        if fold is not None and text is not None:
            s = text + f" + {seed} + {fold}"
            seed = int(hashlib.sha256(s.encode('utf-8')).hexdigest(), 16) % 10**8
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


    @abstractmethod
    def sampleIndices(self, n, k, data, experten, seed = None, sample_equal=False, fold=None):
        pass

    @abstractmethod
    def createLabeledIndices(self, labelerIds, n_L, k, seed=0, sample_equal=False):
        """
        Creates the labeled indices for all folds for every expert
        
        n_L - number of labeled images
        k - number of shared labeled images
        seed - random seed to init the random function
        """
        pass

    @abstractmethod
    def getLabeledIndices(self, labelerId, fold_idx):
        pass

    @abstractmethod
    def addNewLabels(self, filenames, fold_idx, labelerId):
        """
        Add new indeces for labeled images for ssl in combination with active learning

        filenames contains the names of the new labeled images
        fold_idx is the current fold to spezify where the indices should be set
        """
        pass

    @abstractmethod
    def getDatasetsForExpert(self, labelerId, fold_idx):
        pass

    @abstractmethod
    def getTrainDataset(self, labelerId, fold_idx):
        pass

    @abstractmethod
    def getValDataset(self, labelerId, fold_idx):
        pass

    @abstractmethod
    def getTestDataset(self, labelerId, fold_idx):
        pass
        
    @abstractmethod
    def get_train_loader_interface(self, expert, batch_size, mu, n_iters_per_epoch, L, method='comatch', imsize=(128, 128), fold_idx=0, pin_memory=False):
        pass

    @abstractmethod
    def get_val_loader_interface(self, expert, batch_size, num_workers, pin_memory=True, imsize=(128, 128), fold_idx=0):
        pass

    @abstractmethod
    def get_test_loader_interface(self, expert, batch_size, num_workers, pin_memory=True, imsize=(128, 128), fold_idx=0):
        pass

    @abstractmethod
    def getLabeledFilenames(self, labelerId, fold_idx):
        pass
        
    """
    Functions to get the whole dataset as dataloaders
    """
    @abstractmethod
    def get_data_loader_for_fold(self, fold_idx):
        pass

    @abstractmethod
    def get_dataset_for_folder(self, fold_idx):
        pass

    @abstractmethod
    def create_Dataloader_for_Fold(self, idx):
        pass

    @abstractmethod
    def buildDataloaders(self):
        pass

    @abstractmethod
    def getFullDataloader(self):
        pass

    @abstractmethod
    def get_ImageContainer(self):
        pass

    @abstractmethod
    def getData(self):
        pass

###Bis hier hin bearbeitet

class SSL_Dataset(ABC):
    """Class representing the NIH dataset

    :param data: Images
    :param labels: Labels
    :param mode: Mode
    :param imsize: Image size

    :ivar data: Images
    :ivar labels: Labels
    :ivar mode: Mode
    """
    
    @abstractmethod
    def loadImages(self):
        """
        Load all images
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass