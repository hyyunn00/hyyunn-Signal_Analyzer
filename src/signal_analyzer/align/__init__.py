"""Registration-input preparation for brains needing atlas registration.

Never touches the detection mask -- see prepare_registration_input's
docstring and the architecture plan's confirmed facts.
"""
from .convert import prepare_registration_input

__all__ = ["prepare_registration_input"]
