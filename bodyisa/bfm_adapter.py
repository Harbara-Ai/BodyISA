"""Historical BFMAdapter import; a single G1Adapter implementation."""
from .adapters.g1.adapter import G1Adapter, ReferenceMotion, make_env, MINIMAL, DEFAULT_JOINT_POS, POLICY_JOINT_NAMES
BFMAdapter = G1Adapter
