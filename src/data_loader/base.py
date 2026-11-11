from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Optional
import random

@dataclass
class Example:
    id: str
    prompt: str
    source_info: str
    response: str
    labels: list
    split: Optional[str] = None
    source_id: Optional[str] = None
    task_type: Optional[str] = None
    model: Optional[str] = None
    quality: Optional[str] = None
    question: Optional[str] = None  # additional field to store the 'question' subfield of the source info in QA type tasks

class DataLoader(ABC):
    # Steps 1., 2. and 3. are dataset specific methods and therefore are defined in their respective files

    @abstractmethod
    def read_data(self, folder_path):
        """
        1. data reader:
                    Iterates the dataset line by line and returns each row/entry.
        """
        pass

    def join_files(self, data):
        """
        2. files join:
                If there are more than 1 file, links the dataset files' entries together else returns the dataset as is
        """
        return data

    @abstractmethod
    def convert_into_example(self, data):
        """
        3. example conversion:
                      Transforms the data into a shape,
                       ready to be inputted on the detector or evaluator.
                       This shape is constant and same for every type of detector, evaluator or dataset
                       and is called an example.
                       Returns a list of examples.
        """
        pass


    def filter(self, examples, split_filter=None,
                                        model_filter=None,
                                        task_type_filter=None,
                                        quality_filter=None):
        """
        4. examples filtering:
                     Removes examples so that only necessary ones are present for the project's specific goal.
                     It filters on split, model, task_type and quality, which are optional fields of the example,
                     because not every dataset provides them.
                     A filter left as None is ignored.
        """
        filtered_examples = []

        for example in examples:
            if (split_filter is None or split_filter == example.split) and \
               (model_filter is None or model_filter == example.model) and \
               (task_type_filter is None or task_type_filter == example.task_type) and \
               (quality_filter is None or quality_filter == example.quality):
                filtered_examples.append(example)
        return filtered_examples

    def load_data(self, folder_path, split=None, model=None, task_type=None, quality=None, n=None, seed=19):
        # Runs all 4 steps in their respective order
        data = self.read_data(folder_path)
        linked_data = self.join_files(data)
        examples = self.convert_into_example(linked_data)
        structured_data = self.filter(examples, split, model, task_type, quality)

        #The seed is given a default value so that the same n examples are picked in every run,
        # keeping results comparable and reproducible
        if n and n < len(structured_data): structured_data = random.Random(seed).sample(structured_data, n)
        # TODO: add explanatory comment

        return structured_data  # the dataset transformed into containing example shaped data

