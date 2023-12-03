 
from typing import Optional
import os
from .imagelist import ImageList
from ._util import download as download_data, check_exits


class robust(ImageList):
    image_list = {
        "cifar_origin": "image_list/cifar_origin.txt",
        "cifar_corruption": "image_list/cifar_corruption.txt",
    }
    CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    def __init__(self, root: str, task: str, download: Optional[bool] = True, **kwargs):
        assert task in self.image_list
        data_list_file = os.path.join(root, self.image_list[task])

        super(robust, self).__init__(root, robust.CLASSES, data_list_file=data_list_file, **kwargs)

    @classmethod
    def domains(cls):
        return list(cls.image_list.keys())