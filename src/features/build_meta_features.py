import os
import pathlib 
import pandas as pd

from sklearn.preprocessing import LabelEncoder

here = pathlib.Path(__file__).parent.absolute()
data_path = os.path.join(here, '..', '..', 'data')

def encode(infile, outfile):
    meta = pd.read_csv(os.path.join(data_path, 'meta', infile), sep='\t')
    # Generate dataframe of categorical encoded targets
    meta_trainable = pd.DataFrame(index=meta.index)

    le = LabelEncoder()
    for col in 'Class', 'State', 'Type', 'Subtype':
        meta_trainable.loc[:, col] = le.fit_transform(meta.loc[:, col])

    print(meta_trainable)
    meta_trainable.to_csv(os.path.join(data_path, 'processed', outfile), sep=',', index=False)

if __name__ == "__main__":
    for infile in ['meta_organoid.tsv', 'meta_primary.tsv']:
        encode(
            infile=infile,
            outfile=f'{infile}_labels.csv',
        )