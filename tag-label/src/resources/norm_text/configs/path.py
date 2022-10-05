import os
from pathlib import Path

root_path = str(Path(os.path.dirname(os.path.abspath(__file__))).parent)

def join_root_path(path, is_folder=False):
    join_path = os.path.join(root_path, path)
    if is_folder and not os.path.exists(join_path):
        os.makedirs(join_path)
    return join_path

VIETNAMESE_STOPWORDS = join_root_path("vietnamese-stopword.txt")
CUSTOMER_NAMES = join_root_path("customer_name.txt")