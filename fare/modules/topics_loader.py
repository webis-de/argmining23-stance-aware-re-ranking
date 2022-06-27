from typing import Optional, Tuple
from xml.etree.ElementTree import ElementTree, Element, parse

from pandas import DataFrame

from fare.config import CONFIG
from fare.logging import logger


def _parse_objects(xml: Element) -> Tuple[str, str]:
    objects = xml.text.split(",")
    if len(objects) != 2:
        raise RuntimeError(
            f"Expected exactly 2 comparative objects "
            f"but got {len(objects)}."
        )
    object_a, object_b = [obj.strip() for obj in objects]
    return object_a, object_b


def _parse_topic(xml: Element) -> dict:
    number = int(xml.findtext("number").strip())
    title = xml.findtext("title").strip()

    objects_element = xml.find("objects")
    objects: Optional[Tuple[str, str]]
    if objects_element is None:
        logger.warning(
            f"No objects were found for topic '{title}'. "
            "You may need to re-download the latest topic file."
        )
        objects = None
    else:
        objects = _parse_objects(objects_element)
    object_first, object_second = objects

    description_element = xml.find("description")
    if description_element is not None:
        description = description_element.text.strip()
    else:
        logger.warning(f"No description was given for topic '{title}'.")
        description = ""

    narrative_element = xml.find("narrative")
    if narrative_element is not None:
        narrative = narrative_element.text.strip()
    else:
        logger.warning(f"No narrative was given for topic '{title}'.")
        narrative = ""

    return {
        "qid": number,
        "query": title,
        "object_first": object_first,
        "object_second": object_second,
        "description": description,
        "narrative": narrative
    }


def _parse_topics(tree: ElementTree) -> DataFrame:
    root = tree.getroot()
    assert root.tag == "topics"
    return DataFrame(
        data=[_parse_topic(child) for child in root],
        columns=[
            "qid",
            "query",
            "description",
            "narrative",
            "object_first",
            "object_second",
        ],
    ).astype({"qid": "str"})


def parse_topics() -> DataFrame:
    xml: ElementTree = parse(CONFIG.topics_file_path)
    return _parse_topics(xml)
