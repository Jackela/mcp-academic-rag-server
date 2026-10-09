# Narrow boundary used by this project, verified against pdfkit 1.0.0 api.py.
# Source: https://github.com/JazzCore/python-pdfkit/blob/1.0.0/pdfkit/api.py
from typing import IO, Any, Mapping, Optional, Sequence, Union

def from_file(
    input: Union[str, Sequence[str], IO[str]],
    output_path: Optional[str] = ...,
    options: Optional[Mapping[str, Any]] = ...,
) -> Union[bool, bytes]: ...
