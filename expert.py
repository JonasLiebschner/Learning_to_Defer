import numpy as np
import pandas as pd

class Expert:
    def __init__(self, dataset, labeler_id, modus="perfect", param=None, nLabels=800, prob=0.5):
        self.labelerId = labeler_id
        self.dataset = dataset
        self.data = dataset.getData()[["Image ID", str(self.labelerId)]]
        self.nLabels = nLabels
        self.param = param
        self.prob = prob
        self.modus = modus

        if self.modus == "perfect":
            self.predictions = self.data

    def predict(self, img, target, fnames):
        """
        img: the input image
        target: the GT label
        fname: filename (id for the image)
        """
        return np.array([self.predictions[self.predictions["Image ID"] == image_id][str(self.labelerId)].values for image_id in fnames]).ravel()

class NihExpert:
    """A class used to represent an Expert on NIH ChestX-ray data.

    Parameters
    ----------
    labeler_id : int
        the Reader ID to specify which radiologist the expert object represents
    target : str
        the target to make predictions for

    Attributes
    ----------
    labeler_id : int
        the Reader ID to specify which radiologist the expert object represents
    target : str
        the target to make predictions for
    image_id_to_prediction : dict of {int : str}
        a dictionary that maps the image id to the prediction the radiologist made for the specified target

    Methods
    -------
    predict(image_ids)
        makes a prediction for the given image ids
    """

    def __init__(self, labeler_id: int, target: str, PATH, numLabels=800, prob=0.5):
        self.labelerId = labeler_id
        self.target = target
        self.maxLabels = numLabels
        self.prob = prob
        
        self.resetPredictionCount()

        individual_labels = pd.read_csv(PATH + "labels.csv")

        expert_labels = individual_labels[individual_labels["Reader ID"] == self.labelerId][
            ["Image ID", self.target + "_Expert_Label", self.target + "_GT_Label"]]
        expert_labels = expert_labels.fillna(0)

        self.image_id_to_prediction = pd.Series(expert_labels[self.target + "_Expert_Label"].values,
                                                index=expert_labels["Image ID"]).to_dict()

    def predict(self, image_ids):
        """Returns the experts predictions for the given image ids. Works only for image ids that are labeled by the expert

        Parameters
        ----------
        image_ids : list of int
            the image ids to get the radiologists predictions for

        Returns
        -------
        list of int
            returns a list of 0 or 1 that represent the radiologists prediction for the specified target
        """
        return [self.image_id_to_prediction[image_id] for image_id in image_ids]

    def predict_unlabeled_data(self, image_ids):
        """Returns the experts predictions for the given image ids. Works for all image ids, returns -1 if not labeled by expert

        Parameters
        ----------
        image_ids : list of int
            the image ids to get the radiologists predictions for

        Returns
        -------
        list of int
            returns a list of 0 or 1 that represent the radiologists prediction for the specified target, or -1 if no prediction
        """
        return [self.image_id_to_prediction[image_id] if image_id in self.image_id_to_prediction else -1 for image_id in image_ids]
    
    def predictNew(self, image_ids):
        """
        Returns the expert prediction for the first n predictions
        For every other prediction is predicts with the probability (random guessing)
        """
        length = len(image_ids)
        if (self.predictions + length) <= self.maxLabels:
            self.predictions += length
            return [self.image_id_to_prediction[image_id] for image_id in image_ids]
        else:
            temp_predictions = [self.image_id_to_prediction[image_id] for image_id in image_ids[:(self.maxLabels - self.predictions)]]
            self.predictions = self.maxLabels
            for image_id in image_ids[(self.maxLabels - self.predictions):]:
                if np.random.uniform(0,1) > self.prob:
                    temp_predictions.append(self.image_id_to_prediction[image_id])
                else:
                    temp_predictions.append(np.random.randint(2, size=1))
    
    def resetPredictionCount(self):
        self.predictions = 0