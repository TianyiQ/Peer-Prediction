import json
from typing import Tuple, Iterable, Dict, Hashable, Any
import os
import re
from time import strftime, localtime

try:
    from jsonstreamer import JSONStreamer
    brute_force = False
except:
    print("jsonstreamer not found, switching to brute force. Please install it from pip, or if installation failed, install the apt package libyajl2-dev and try again.")
    brute_force = True


def read_json_memory_efficient(path: str) -> Iterable[Dict[Hashable, Any]]:
    """
    Read a json file with the format List[Dict[Hashable, Any]] (e.g. List[Dict[str,List[str]]]).
    Instead of reading the entire file into memory, it reads chunk by chunk (with no regard to line breaks) and returns an iterator.
    It does not require that each Dict rests on a single line, nor that only one Dict rests on one line.
    
    Args:
        path (str): Path to the json file.
        
    Returns:
        Iterable[Dict[Hashable, Any]]: An iterator over the elements in the json file.
    """
    
    if brute_force:
        with open(path, "r") as in_file:
            json_content = json.load(in_file)
            for element in json_content:
                yield element
        return
    
    stack = []
    last_key = None

    def catch_all_events(event_name, *args):
        
        nonlocal stack, last_key
        if "doc" in event_name:
            return

        if "start" in event_name:
            obj = {} if event_name == "object_start" else []

            if len(stack) and last_key is not None:
                assert type(stack[-1]) == dict
                stack[-1][last_key] = obj
                last_key = None
            elif len(stack):
                assert type(stack[-1]) == list
                stack[-1].append(obj)

            stack.append(obj)
        elif "end" in event_name:
            if len(stack) != 1:
                assert type(stack[-1]) == (list if "array" in event_name else dict)
                stack.pop()
        elif event_name == "key":
            assert last_key is None and (type(stack[-1]) == dict)
            last_key = args[0]
        elif event_name == "value":
            assert last_key is not None and (type(stack[-1]) == dict)
            stack[-1][last_key] = args[0]
            last_key = None
        else:
            assert event_name == "element" and (type(stack[-1]) == list)
            stack[-1].append(args[0])

    streamer = JSONStreamer()
    streamer.add_catch_all_listener(catch_all_events)

    max_chars = 500000
    with open(path, "r") as in_file:
        while True:
            s = in_file.read(max_chars)
            if not s:
                break

            streamer.consume(s)
            assert len(stack) and (type(stack[0]) == list)
            if len(stack[0]):
                for i in range(len(stack[0]) - 1):
                    yield stack[0][i]

                stack[0] = [stack[0][-1]]

    streamer.close()
    for element in stack[0]:
        yield element


class JsonListReader:

    def __init__(self, json_path: str):
        """
        Read a json list from a file, line by line. Memory-efficient, and can handle arbitrarily large files.
        This class should be used as a context manager.
        
        Example:
        with JsonListReader('./test.json') as reader:
            for element in reader:
                print(element)
        """

        self.path = json_path

    def __enter__(self):
        return read_json_memory_efficient(self.path)

    def __exit__(self, type, value, traceback):
        pass


class JsonListWriter:

    def __init__(self, json_path: str):
        """
        Write a json list into a file, line by line. Memory-efficient, and can handle arbitrarily large files.
        This class should be used as a context manager.
        
        Example:
        with JsonListWriter('./test.json') as writer:
            writer.append({'key': value})
        """

        self.path = json_path
        self.file_obj = open(self.path, "w")
        self.file_obj.write("[")
        self.is_first = True

    def __enter__(self):
        return self

    def append(self, element: Any, flush: bool = False):
        """Append an element at the end of the json list."""

        self.file_obj.write("\n" if self.is_first else ",\n")
        self.is_first = False
        self.file_obj.write(json.dumps(element))
        if flush:
            self.file_obj.flush()

    def __exit__(self, type, value, traceback):
        self.file_obj.write("\n]")
        self.file_obj.close()