# =============================================================================
# Adapted from MedAgent-Pro (https://github.com/jinlab-imvr/MedAgent-Pro)
# Original work: Wang et al., "MedAgent-Pro: Towards Evidence-Based
#   Multi-Modal Medical Diagnosis via Reasoning Agentic Workflow",
#   ICLR 2026.
# Upstream license: no license file. Used with attribution; see
#   baselines/MedAgent-Pro/ATTRIBUTION.md for details.
# Modifications by the DermAgent authors for dermatology benchmarks.
# =============================================================================

from .GPT_Decider import GPT_Decider
from .Janus_Decider import Janus_Decider
from .BioMedClip_Decider import BioMedClip_Decider
from .Pro_Decider import Pro_Decider
from .Qwen_Decider import Qwen_Decider
from .InternVL_Decider import InternVL_Decider
from .Gemma_Decider import Gemma_Decider
from .MultiClass_Decider import MultiClass_Decider