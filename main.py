from src.data_loader import ragtruth as rag_loader
# test runs
if __name__ == '__main__':
    l = rag_loader.RAGTruthLoader()
    dataset = l.read_data('data/raw')
    linked_dataset = l.join_files(dataset)
