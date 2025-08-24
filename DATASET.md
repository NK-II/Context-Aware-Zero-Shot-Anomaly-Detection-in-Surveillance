# Dataset Setup

We use the **UCA(UCF Crime Annotation) Dataset** available on Kaggle.

## Steps:
1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/vigneshwar472/ucaucf-crime-annotation-dataset).
2. Create the directory named `videos/` & include all the normal vdeos from the dataset and only abnormal/anomalous videos that matches the ones that are in the val/test text data. (You can test or validate on other anomalous videos as well. Just make sure to the video's text desctiption is placed inside the val/test text file)
3. Extract the videos into the `videos/` directory.

   Example structure:

   ```
   videos/
     ├── Raw videos

   ```
3. Use the provided split files under `data/`:
   - `train_list.txt`
   - `val_list.txt`
   - `test_list.txt`
4. Ensure that the video paths in the text files match your `videos/` directory.

The `data/` directory is kept in the repository since it only contains metadata (small text files), not raw videos.
