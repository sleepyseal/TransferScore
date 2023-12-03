 
from typing import Optional
import os,sys
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class sem(ImageList):
    image_list = {
        "source": "image_list/source.txt",
        "target": "image_list/target.txt",
    }
    CLASSES = ['0','1','2','3']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])
        # print(data_list_file)
        # sys.exit(0)
        super(sem, self).__init__(root, sem.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())