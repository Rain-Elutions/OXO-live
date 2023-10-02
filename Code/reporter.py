from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, List


def seconds_to_ms(epoch:int) -> int:
    """Function to convert an epoch in s (standard format) to ms (our server format)"""
    return int(1000 * epoch)

# TODO: make this into an abc with 3 abstract properties
class AdditionalInfoChecker(ABC):
    """ 
    Interface for a reporting check
    """
    @abstractmethod
    def report(self, epoch: int) -> list:
        """Function to internal info to a single live deployment line"""

    def add_value(self, epoch: int, val: Any) -> None:
        """Function to add a value to a timestamp to the value field"""
        
        self.value[epoch] = self.convert_type(val)


# NOTE: the classes here need to have a defaultdict bec in the main script we need to assume 
# that things are working until they aren't (or vice versa) 

class LiveConnectionChecker(AdditionalInfoChecker):
    """
    Class to hold object id, property id and a default value for a live connection check
    """
    convert_type = int
    def __init__(self, object_id:int , property_id:int) -> None:
        self.object_id = object_id
        self.property_id = property_id

        self.value = defaultdict(lambda: 1) # NOTE: 1 means there is a live connection

    def report(self, epoch: int) -> list:
        return [self.object_id, self.property_id, int(self.value[epoch]), seconds_to_ms(epoch), 0]


class MissingDataChecker(AdditionalInfoChecker):
    """
    Class to hold object id, property id and a default value for a the number of missing variables
    """
    convert_type = int

    def __init__(self, object_id:int, property_id:int) -> None:
        self.object_id = object_id
        self.property_id = property_id

        self.value = defaultdict(lambda: 0)  # NOTE: 0 means there is no missing data

    def report(self, epoch: int) -> list:
        return [self.object_id, self.property_id, self.value[epoch], seconds_to_ms(epoch), 0]

# NOTE: do we need a checker for the missing tags themselves??? If we do consider modifying the class above ...

class NewDirectiveChecker(AdditionalInfoChecker):
    """
    New directive checker
    """
    convert_type = int

    def __init__(self, object_id:int, property_id:int) -> None:
        self.object_id = object_id
        self.property_id = property_id

        self.value = defaultdict(lambda: 0) # NOTE: this means there is not a new directive

    def report(self, epoch: int) -> list:
        return [self.object_id, self.property_id, self.value[epoch], seconds_to_ms(epoch), 0]


class PredictedKPIChecker(AdditionalInfoChecker):
    """
    KPI Checker
    """
    convert_type = float

    def __init__(self, object_id:int, property_id:int) -> None:
        self.object_id = object_id
        self.property_id = property_id

        self.value = defaultdict(lambda: 0) # NOTE: this means there is not a new directive

    def report(self, epoch: int) -> list:
        return [self.object_id, self.property_id, self.value[epoch], seconds_to_ms(epoch), 0]


class OptimizedKPIChecker(AdditionalInfoChecker):
    """
    Optimized KPI Checker
    """
    convert_type = float

    def __init__(self, object_id:int, property_id:int) -> None:
        self.object_id = object_id
        self.property_id = property_id

        self.value = defaultdict(lambda: 0) # NOTE: this means there is not a new directive

    def report(self, epoch: int) -> list:
        return [self.object_id, self.property_id, self.value[epoch], seconds_to_ms(epoch), 0]

class NPSpecificConsChecker(AdditionalInfoChecker):
    """
    NP Specific Cons Checker
    """
    convert_type = float

    def __init__(self, object_id:int, property_id:int) -> None:
        self.object_id = object_id
        self.property_id = property_id

        self.value = defaultdict(lambda: 0) # NOTE: this means there is not a new directive

    def report(self, epoch: int) -> list:
        return [self.object_id, self.property_id, self.value[epoch], seconds_to_ms(epoch), 0]

class NGSpecificConsChecker(AdditionalInfoChecker):
    """
    NP Specific Cons Checker
    """
    convert_type = float

    def __init__(self, object_id:int, property_id:int) -> None:
        self.object_id = object_id
        self.property_id = property_id

        self.value = defaultdict(lambda: 0) # NOTE: this means there is not a new directive

    def report(self, epoch: int) -> list:
        return [self.object_id, self.property_id, self.value[epoch], seconds_to_ms(epoch), 0]


@dataclass
class AdditionalInfoReporter:
    """
    Class to hold information about the live connection, missing data check, and the new directive boolean
    """
    reporters: field(default_factory=list)

    def report(self, epoch:int) -> List[List[int]]:
        """Function to append check information to the data"""
        additional_data = [None] * len(self.reporters)

        for i, reporter in enumerate(self.reporters):
            additional_data[i] = reporter.report(epoch)
        
        return additional_data


def create_additonal_info_container(
    epochs: List[int],
    *checkers
    ) -> dict:
    """
    Function to create a dictionary which holds information on the additional information needed for that file
    NOTE: this will hold information in the live deployment format so it can written out easily

    Parameters
    ----------
    epochs : List[int]
        list of timestamps in the epoch format
    checkers: List[AdditionalInfoChecker]
        list of additional checkers

    Returns
    -------
    info_container : Dict
        container with additional information to add for each epoch    
    """
    info_container = {}

    for epoch in epochs:
        # initialize the additional live deployment info for each combination of object and property ids
        # add the timestamp to the dictionary
        info_container[epoch] = [checker.report(epoch) for checker in checkers]

    return info_container



