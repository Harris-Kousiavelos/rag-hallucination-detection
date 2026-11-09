from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class Example:
    id: str
    prompt: str
    source_info: str
    response: str
    labels: list
    split: str = None
    source_id: str = None
    task_type: str = None
    model: str = None
    quality: str = None

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

    def filter(self, examples, split_filter=None, model_filter=None, task_type_filter=None):
        """
        4. examples filtering:
                     Removes examples so that only necessary ones are present for the project's specific goal.
                     It filters on split, model and task_type, which are optional fields of the example,
                     because not every dataset provides them.
                     A filter left as None is ignored.
        """
        filtered_examples = []

        for example in examples:
            if (split_filter is None or split_filter == example.split) and \
               (model_filter is None or model_filter == example.model) and \
               (task_type_filter is None or task_type_filter ==example.task_type):
                filtered_examples.append(example)
        return filtered_examples

    def load_data(self, folder_path, split=None, model=None, task_type=None):
        # Runs all 4 steps in their respective order
        data = self.read_data(folder_path)
        linked_data = self.join_files(data)
        examples = self.convert_into_example(linked_data)
        structured_data = self.filter(examples, split, model, task_type)

        return structured_data  # the dataset transformed into containing example shaped data

