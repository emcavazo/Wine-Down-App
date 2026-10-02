#Making wine_embeddings.npy smaller for import to github

import numpy as np

input = "wine_embeddings.npy"

emb = np.load(input)
print(f"Loaded in:",emb.shape, emb.dtype) #this is just to check the shape and type of the embeddings

if emb.dtype == np.float16:
    print("Already float16, no need to convert")
else:
    emb16 = emb.astype(np.float16)
    np.save(input,emb16)
    print(f"Converted to float16, new size: {emb.nbytes / 1024**2:.2f} MB")



