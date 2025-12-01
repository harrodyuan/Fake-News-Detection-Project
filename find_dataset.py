from huggingface_hub import list_datasets

def search_hf():
    print("Searching for datasets...")
    datasets = list_datasets(search="Pulk17")
    found = False
    for d in datasets:
        print(d.id)
        if "Fake-News-Detection" in d.id:
            found = True
    
    if not found:
        print("Searching generally for 'Fake News Detection'...")
        datasets = list_datasets(search="Fake News Detection", limit=10)
        for d in datasets:
            print(d.id)

if __name__ == "__main__":
    search_hf()

