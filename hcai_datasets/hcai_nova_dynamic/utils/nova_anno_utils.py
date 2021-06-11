import tensorflow_datasets as tfds
import hcai_datasets.hcai_nova_dynamic.utils.nova_types as nt
import numpy as np

from abc import ABC, abstractmethod
from hcai_datasets.hcai_nova_dynamic.utils.nova_utils import merge_role_key, split_role_key

class Annotation(ABC):

    def __init__(self, role: str = '', scheme: str = '', is_valid: bool = True):
        self.role = role
        self.scheme = scheme
        self.is_valid = is_valid

        # Gets set when load_annotation is called
        self.data = None

    @abstractmethod
    def get_tf_info(self):
        """
        Returns the labels for this annotation to create the DatasetInfo for tensorflow
        """
        raise NotImplementedError

    @abstractmethod
    def set_annotation_from_mongo_doc(self, session):
        """
        Returns the labels for this annotation to create the DatasetInfo for tensorflow
        """
        raise NotImplementedError

    @abstractmethod
    def get_label_for_frame(self, start, end):
        """
        Returns the label for this frame
        """
        raise NotImplementedError

class DiscreteAnnotation(Annotation):

    REST = 'REST'

    def __init__(self, labels= {}, add_rest_class=False, **kwargs):
        super().__init__(**kwargs)
        self.type = nt.AnnoTypes.DISCRETE
        self.labels = { x['id'] : x['name'] if x['isValid'] else '' for x in sorted(labels, key=lambda k: k['id']) }
        self.add_rest_class = add_rest_class
        if self.add_rest_class:
            self.labels[max(self.labels.keys()) + 1] = DiscreteAnnotation.REST

    def get_tf_info(self):
        return (merge_role_key(self.role, self.scheme), tfds.features.ClassLabel(names=list(self.labels.values())))

    def set_annotation_from_mongo_doc(self, mongo_doc, time_to_ms=False):
        self.data = mongo_doc
        if time_to_ms:
            for d in self.data:
                d['from'] = int(d['from'] * 1000)
                d['to'] = int(d['to'] * 1000)

    def get_label_for_frame(self, start, end):

        # If we don't have any data we return the garbage label
        if self.data == -1:
            return -1

        else:
            # Finding all annos that overlap with the frame
            def is_overlapping(af, at, ff, ft):

                # anno is larger than frame
                altf = af <= ff and at >= ft

                # anno overlaps frame start
                aofs = at >= ff and at <= ft

                # anno overlaps frame end
                aofe = af >= ff and af <= ft

                return altf or aofs or aofe

            annos_for_sample = list(filter(lambda x: is_overlapping(x['from'], x['to'], start, end), self.data))

            # No label matches
            if not annos_for_sample:
                if self.add_rest_class:
                    return len(self.labels.values()) -1
                else:
                    return -1

            majority_sample_idx = np.argmax(
                list(map(lambda x: min(end, x['to']) - max(start, x['from']), annos_for_sample)))

            return annos_for_sample[majority_sample_idx]['id']


class ContinousAnnotation(Annotation):

    def __init__(self, sr=0, min=0, max= 0, **kwargs):
        super().__init__(**kwargs)
        self.type = nt.AnnoTypes.CONTINOUS
        self.sr = sr
        self.min = min
        self.max = max


class FreeAnnotation(Annotation):

    def __init__(self, **kwargs):
        self.type = nt.AnnoTypes.FREE
        super().__init__(**kwargs)


class PolygonAnnotaion(Annotation):

    def __init__(self, **kwargs):
        self.type = nt.AnnoTypes.FREE
        super().__init__(**kwargs)


