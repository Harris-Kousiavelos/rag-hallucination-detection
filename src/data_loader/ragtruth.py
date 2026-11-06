"""
 1. reader:
            Iterates the dataset line by line and returns each row/entry as a dictionary.
            The dataset consists of two logically linked distinct files:
            source_info.jsonl & response.jsonl.
"""

"""
2. join: 
         The two dataset files have to be joined to get the full view of their actual logic context.
         Each prompt (from source_info.jsonl) links to 6 answers (from response.jsonl)
         via the attribute 'source_id'.
"""

"""
3. example conversion:
                       Transforms the joined dictionaries into an example
"""