import tempfile
from pathlib import Path

from spirl.utils.download import download_file_from_google_drive, unzip

FILE_IDS = {
    'kitchen': '1RE4XwTmZQ7xMKBU4ZXtCwBhvxZQKjXdN',
    'pointmaze': '1ySROlRBABWpK0CXP9e8B3pCpeDlbvuwh',
    'antmaze': '1-Zsd0HZYJ6XzcSJTaQfMXnz58IbSMepF'
}

data_dir = Path('./data')
data_dir.mkdir(exist_ok=True)

for name in FILE_IDS:
    with tempfile.TemporaryDirectory() as tmp_dir:
        zip_file = str(Path(tmp_dir) / f'{name}.zip')
        print('Downloading the dataset ...')
        download_file_from_google_drive(FILE_IDS[name],  zip_file)
        print(f'Unzipping {zip_file} to {str(data_dir)} ...')
        unzip(zip_file, str(data_dir))
    print('Download and Extraction complete.')