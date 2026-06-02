"""Re-export of the package-level obs module so submodules can log without
introducing a circular import."""
from .. import obs

event = obs.event
timed = obs.timed
