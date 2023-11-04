import pandas as pd
from tqdm import tqdm
from datasets import Dataset, DatasetDict


# I am using double space separator to parse data correctly.
df = pd.read_csv(
    filepath_or_buffer="../data/raw/filtered.tsv", sep='	', header=0)

# Dropping unnecessary column with row indices because they are already included in pandas DataFrame.
df = df.drop('Unnamed: 0', axis=1)

# Amend the spelling mistakes...
df = df.rename(columns={'lenght_diff': 'length_diff'})

# Swap pairs with trn_tox > ref_tox.
for i, row in tqdm(df.iterrows()):
    if row['trn_tox'] > row['ref_tox']:

        # Swap toxicities
        copy_ref_tox = row['ref_tox']
        df.at[i, 'ref_tox'] = row['trn_tox']
        df.at[i, 'trn_tox'] = copy_ref_tox

        # Swap texts
        copy_ref_text = row['reference']
        df.at[i, 'reference'] = row['translation']
        df.at[i, 'translation'] = copy_ref_text

df = df.drop(df.loc[df['trn_tox'] > 0.4].index)

df = df.loc[df['similarity'] > 0.75]

# Split df into train, validation, test,
# and convert to Hugging Face Dataset to work with pretrained model.
train_size = int(len(df) * 0.75)
validation_size = int(len(df) * 0.05)
test_size = int(len(df) * 0.2)

train_df = df[:train_size]
validation_df = df[train_size:train_size + validation_size]
test_df = df[train_size + validation_size:]

paranmt_ds = DatasetDict()

paranmt_ds['train'] = Dataset.from_pandas(train_df)
paranmt_ds['validation'] = Dataset.from_pandas(validation_df)
paranmt_ds['test'] = Dataset.from_pandas(test_df)

# Crop the dataset.
cropped_datasets = paranmt_ds
cropped_datasets['train'] = paranmt_ds['train'].select(range(10000))
cropped_datasets['validation'] = paranmt_ds['validation'].select(range(1000))
cropped_datasets['test'] = paranmt_ds['test'].select(range(1000))

# Save datasets.
# Save full preprocessed Dataframe.
df.to_csv("..\data\interim\para-nmt-preprocessed.csv")

# Save preprocessed and cropped Hugging Face dataset.
cropped_datasets.save_to_disk("..\data\interim\para-nmt-preprocessed-cropped")
