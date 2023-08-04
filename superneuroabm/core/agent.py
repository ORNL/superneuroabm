from __future__ import annotations
from typing import Any, Callable, List, Dict, Optional, Union
from collections import OrderedDict
from copy import copy
import warnings
from multiprocessing import shared_memory

import numpy as np
from numba import cuda

from superneuroabm.core.util import (
    compress_tensor,
    convert_to_equal_side_tensor,
)


class Breed:
    def __init__(self, name: str) -> None:
        # self._properties is a dict with keys as property name and
        #   values are properties.
        #   properties themselves are list of type and default value.
        self._properties = OrderedDict()
        self._prop2pos = {}
        self._name: str = name
        self._step_funcs: Dict[int, Callable] = {}
        self._breedidx = -1
        self._num_properties = 0
        self._prop2maxdims: Dict[str, List[int]] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def properties(self) -> Dict[str, Any]:
        return self._properties

    @property
    def step_funcs(self) -> Dict[int, Callable]:
        return self._step_funcs

    def register_property(
        self,
        name: str,
        default: Union[int, float, List] = np.nan,
        max_dims: Optional[List[int]] = None,
    ) -> None:
        self._properties[name] = default
        self._prop2pos[name] = self._num_properties
        self._num_properties += 1
        self._prop2maxdims[name] = max_dims

    def register_step_func(self, step_func: Callable, priority: int = 0):
        """
        What the agent is supposed to do during a simulation step.

        """
        self._step_funcs[priority] = step_func


class AgentFactory:
    def __init__(self) -> None:
        self._breeds: Dict[str, Breed] = OrderedDict()
        self._num_breeds = 0
        self._num_agents = 0
        self._property_name_2_agent_data_tensor = OrderedDict({"breed": []})
        self._property_name_2_defaults = OrderedDict({"breed": 0})
        self._property_name_2_max_dims = OrderedDict({"breed": []})

    @property
    def breeds(self) -> List[Breed]:
        """
        Returns the breeds registered in the model

        :return: A list of currently registered breeds.

        """
        return self._breeds.values()

    @property
    def num_agents(self) -> int:
        """
        Returns number of agents. Agents are not removed if they are killed at the
            moment.

        """
        return self._num_agents

    def register_breed(self, breed: Breed) -> None:
        """
        Registered agent breed in the model so that agents can be created under
            this definition.

        :param breed: Breed definition of agent

        """
        breed._breedidx = self._num_breeds
        self._num_breeds += 1
        self._breeds[breed.name] = breed
        for property_name, default in breed.properties.items():
            self._property_name_2_agent_data_tensor[property_name] = []
            self._property_name_2_defaults[property_name] = default
            self._property_name_2_max_dims[
                property_name
            ] = breed._prop2maxdims[property_name]

    def create_agent(self, breed: Breed, **kwargs) -> int:
        """
        Creates and agent of the given breed initialized with the properties given in
            **kwargs.

        :param breed: Breed definition of agent
        :param **kwargs: named arguments of agent properties. Names much match properties
            already registered in breed.
        :return: Agent ID

        """
        if breed.name not in self._breeds:
            raise ValueError(f"Fatal: unregistered breed {breed.name}")
        property_names = self._property_name_2_agent_data_tensor.keys()
        for property_name in property_names:
            if property_name == "breed":
                breed = self._breeds[breed.name]
                self._property_name_2_agent_data_tensor[property_name].append(
                    breed._breedidx
                )
            else:
                default_value = copy(
                    self._property_name_2_defaults[property_name]
                )
                self._property_name_2_agent_data_tensor[property_name].append(
                    kwargs.get(property_name, default_value)
                )

        self._num_agents += 1
        return self._num_agents - 1

    def get_agent_property_value(
        self, property_name: str, agent_id: int
    ) -> Any:
        """
        Returns the value of the specified property_name of the agent with
            agent_id

        :param property_name: str name of property as registered in the breed.
        :param agent_id: Agent's id as returned by create_agent
        :return: value of property_name property for agent of agent_id
        """
        return self._property_name_2_agent_data_tensor[property_name][agent_id]

    def set_agent_property_value(
        self,
        property_name: str,
        agent_id: int,
        value: Any,
        dims: Optional[List[int]] = None,
    ) -> None:
        """
        Sets the property of property_name for the agent with agent_id with
            value.
        :param property_name: str name of property as registered in the breed.
        :param agent_id: Agent's id as returned by create_agent
        :param value: New value for property
        :param dims: Optional dimensions of property_value as List of ints. Not
            required, but improves performance if provied. E.g. [0] if value is a
            single number like 0, [4, 2] if it is a multi-dimensional list/array with
            dimensions 4 and 2.
        """

        if property_name not in self._property_name_2_agent_data_tensor:
            raise ValueError(f"{property_name} not a property of any breed")
        self._property_name_2_agent_data_tensor[property_name][
            agent_id
        ] = value
        if dims:
            if self._property_name_2_max_dims[property_name]:
                if len(dims) != len(
                    self._property_name_2_max_dims[property_name]
                ):
                    raise ValueError(
                        f"Dimensions {dims} do not match exsiting number of dimensions of property {property_name} which is {len(dims)} for other agents"
                    )
                max_dims = self._property_name_2_max_dims.get(
                    property_name, dims
                )
                for i, dim in enumerate(dims):
                    max_dims[i] = max(max_dims[i], dim)
                self._property_name_2_max_dims[property_name] = max_dims
            else:
                self._property_name_2_max_dims[property_name] = dims
        else:
            warnings.warn(
                "Performance reduced with no dimension specification for agent property"
            )

    def get_agents_with(self, query: Callable) -> Dict[int, List[Any]]:
        """
        Returns an Dict, key: agent_id value: List of properties, of the agents that satisfy
            the query. Query must be a callable that returns a boolean and accepts **kwargs
            where arguments may with breed property names may be accepted and used to form
            query logic.

        :param query: Callable that takes agent data as dict and returns List of agent data
        :return: Dict of agent_id: List of properties

        """
        matching_agents = {}
        property_names = self._property_name_2_agent_data_tensor.keys()
        for agent_id in range(self._num_agents):
            agent_properties = {
                property_name: self._property_name_2_agent_data_tensor[
                    property_name
                ][agent_id]
                for property_name in property_names
            }
            if query(**agent_properties):
                matching_agents[agent_id] = agent_properties
        return matching_agents

    def generate_agent_data_tensors(
        self, use_cuda=True
    ) -> Union[List[List[Any]], Dict[str, List[Any]]]:
        converted_agent_data_tensors = []
        dtype = np.float64
        for property_name in self._property_name_2_agent_data_tensor.keys():
            adt = self._property_name_2_agent_data_tensor[property_name]
            max_dims = self._property_name_2_max_dims.get(property_name, None)
            if max_dims != None:
                max_dims = [self.num_agents] + max_dims
            adt = convert_to_equal_side_tensor(adt, max_dims)
            if use_cuda:
                adt = cuda.to_device(adt)
            else:
                # use shared memory if not cuda
                d_size = np.dtype(dtype).itemsize * np.prod(adt.shape)
                shm = shared_memory.SharedMemory(
                    create=True,
                    size=d_size,
                    name=f"npshared{property_name}",
                )
                dst = np.ndarray(shape=adt.shape, dtype=dtype, buffer=shm.buf)
                dst[:] = adt[:]
            converted_agent_data_tensors.append(adt)
        return converted_agent_data_tensors

    def update_agents_properties(
        self, equal_side_agent_data_tensors=List[List[Any]], use_cuda=True
    ) -> None:
        property_names = list(self._property_name_2_agent_data_tensor.keys())
        for i, property_name in enumerate(property_names):
            if use_cuda:
                dt = equal_side_agent_data_tensors[i]
            else:
                dt = equal_side_agent_data_tensors[i]
                # Free shared memory
                shm = shared_memory.SharedMemory(
                    name=f"npshared{property_name}"
                )
                shm.close()
                shm.unlink()
            self._property_name_2_agent_data_tensor[
                property_names[i]
            ] = compress_tensor(dt)
