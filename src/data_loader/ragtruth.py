from src.data_loader.base import DataLoader
from src.data_loader.base import Example
import json
import os

class RAGTruthLoader(DataLoader):

    def read_data(self, folder_path):
        """
         1. reader:
                    Iterates the dataset line by line and returns each row/entry as a dictionary, grouped in one list per file.
                    The dataset consists of two logically linked distinct files:
                    source_info.jsonl & response.jsonl.
        """

        data = {
            'source': [],
            'response': []
        }

        # Read the actual dataset files by using the folder parameter
        with open(os.path.join(folder_path, 'source_info.jsonl')) as source_dataset:
            with open(os.path.join(folder_path, 'response.jsonl')) as response_dataset:
                data['source'].extend(json.loads(line) for line in source_dataset)
                data['response'].extend(json.loads(line) for line in response_dataset)

        return data


    def join_files(self, data):
        """
        2. join:
                 The two dataset files have to be joined to get the full view of their actual logic context.
                 Each prompt (from source_info.jsonl) links to 6 answers (from response.jsonl)
                 via the attribute 'source_id'.
        """

        # the sources dict contains the source ids as keys having the entire source file entry as the value
        sources = {source_line['source_id']: source_line for source_line in data['source']}

        # the linked data list contains each response dict joined with its respective source entry (source first)
        linked_data = [sources[response_line['source_id']] | response_line for response_line in data['response']]

        return linked_data


    def convert_into_example(self, data):
        """
        3. example conversion:
                               Transforms the joined dictionaries into an example.
                               The source_info field is transformed into text.
        """
        examples = []

        for entry in data:
            source_info_str = entry['source_info']  # in order not to mutate the parameter-dictionary
            qa_question = None

            # In case the task type is a QA or Data2txt, and thus has a dictionary as the source_info's value,
            # it needs to be transformed to match the general Example signature defined in base.py
            if entry['task_type'] == 'QA':
                source_info_str = entry['source_info']['passages']
                qa_question = entry['source_info']['question']
            elif entry['task_type'] == 'Data2txt':
                source_info_str = str(entry['source_info'])

            # Create example from the linked data dictionaries
            example = Example(entry['id'],
                              entry['prompt'],
                              source_info_str,
                              entry['response'],
                              entry['labels'],
                              entry['split'],
                              entry['source_id'],
                              entry['task_type'],
                              entry['model'],
                              entry['quality'],
                              qa_question
                              )
            examples.append(example)
        return examples

