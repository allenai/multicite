"""

Shared utility functions

"""


from typing import Tuple

import re


def instance_id_to_paper_id_and_intent(instance_id: str) -> Tuple[str, str]:
    match = re.match(r'(.+)__(@.+@)__[0-9]+', instance_id)
    return match.group(1), match.group(2)


def sent_id_to_pos(sent_id: str) -> int:
    match = re.match(r'.+-C001-([0-9]+)', sent_id)
    return int(match.group(1)) - 1      # in our multicite dataset, sent_ids counts from 1, not 0
