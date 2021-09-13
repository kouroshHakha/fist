import tempfile
from pathlib import Path

from spirl.utils.download import download_file_from_google_drive, unzip

ID = '1OEkPI-z7THYt0T-DbEe2UiQG1z2lQ0LA'

with tempfile.TemporaryDirectory() as tmp_dir:
    zip_file = str(Path(tmp_dir) / 'checkpoints.zip')
    print('Downloading the checkpoints ...')
    download_file_from_google_drive(ID,  zip_file)
    print(f'Unzipping {zip_file} to current directory ...')
    unzip(zip_file, '.')
    print('Download and Extraction complete.')