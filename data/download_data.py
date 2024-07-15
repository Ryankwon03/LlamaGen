from datasets import load_dataset
#access_token = "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"




dataset = load_dataset('ILSVRC/imagenet-1k', 
                       split='train', 
                       trust_remote_code=True,
                       cache_dir='E:/.cache') #data stored here

print(dataset.cache_files)
print(dataset[0])