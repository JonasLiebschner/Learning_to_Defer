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
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
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
###Bis hier hin bearbeitet