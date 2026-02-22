"""Request / response schemas for the prediction API."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Bundle label lookup ──────────────────────────────────────────────
BUNDLE_NAMES: dict[int, str] = {
    0: "Auto_Comprehensive",
    1: "Auto_Liability_Basic",
    2: "Basic_Health",
    3: "Family_Comprehensive",
    4: "Health_Dental_Vision",
    5: "Home_Premium",
    6: "Home_Standard",
    7: "Premium_Health_Life",
    8: "Renter_Basic",
    9: "Renter_Premium",
}


# ═══════════════════════════════════════════════════════════════════════
#  Shared customer-fields mixin
# ═══════════════════════════════════════════════════════════════════════
class CustomerFields(BaseModel):
    """Raw customer features — same column names as the training CSV."""

    User_ID: Optional[str] = None
    Policy_Cancelled_Post_Purchase: Optional[int] = 0
    Policy_Start_Year: Optional[int] = None
    Policy_Start_Month: Optional[str] = None
    Policy_Start_Week: Optional[int] = None
    Policy_Start_Day: Optional[int] = None
    Grace_Period_Extensions: Optional[int] = 0
    Previous_Policy_Duration_Months: Optional[int] = 0
    Adult_Dependents: Optional[int] = 0
    Child_Dependents: Optional[float] = 0
    Infant_Dependents: Optional[int] = 0
    Region_Code: Optional[str] = None
    Existing_Policyholder: Optional[int] = 0
    Previous_Claims_Filed: Optional[int] = 0
    Years_Without_Claims: Optional[int] = 0
    Policy_Amendments_Count: Optional[int] = 0
    Broker_ID: Optional[float] = None
    Employer_ID: Optional[float] = None
    Underwriting_Processing_Days: Optional[int] = 0
    Vehicles_on_Policy: Optional[int] = 0
    Custom_Riders_Requested: Optional[int] = 0
    Broker_Agency_Type: Optional[str] = None
    Deductible_Tier: Optional[str] = None
    Acquisition_Channel: Optional[str] = None
    Payment_Schedule: Optional[str] = None
    Employment_Status: Optional[str] = None
    Estimated_Annual_Income: Optional[float] = 0.0
    Days_Since_Quote: Optional[int] = 0


# ═══════════════════════════════════════════════════════════════════════
#  /predict
# ═══════════════════════════════════════════════════════════════════════
class PredictRequest(CustomerFields):
    pass


class PredictResponse(BaseModel):
    predicted_bundle: int = Field(..., ge=0, le=9)
    bundle_name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: List[float]


# ═══════════════════════════════════════════════════════════════════════
#  /health
# ═══════════════════════════════════════════════════════════════════════
class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    agents_ready: bool = False


# ═══════════════════════════════════════════════════════════════════════
#  /explain
# ═══════════════════════════════════════════════════════════════════════
class ExplainRequest(CustomerFields):
    pass


class InfluenceItem(BaseModel):
    feature: str
    label: str
    impact: float
    direction: str


class UncertaintyInfo(BaseModel):
    entropy: float
    level: str
    explanation: str


class NearestBundleItem(BaseModel):
    bundle: int
    bundle_name: str
    similarity_level: str
    similarity_context: str


class NarrativeStep(BaseModel):
    step: int
    agent: str
    title: str
    summary: str


class NarrativeInfo(BaseModel):
    decision_timeline: List[NarrativeStep]


class BusinessInsight(BaseModel):
    customer_segment: str
    retention_risk: str
    upsell_potential: str


class AgentVote(BaseModel):
    agent: str
    top_bundle: int
    top_bundle_name: str
    confidence: float
    probabilities: List[float]


class AgentConsensus(BaseModel):
    top_bundle: int
    top_bundle_name: str
    confidence: float
    agreement: int
    total_agents: int
    probabilities: List[float]


class ExplainResponse(BaseModel):
    predicted_bundle: int = Field(..., ge=0, le=9)
    confidence: float
    probabilities: List[float]
    uncertainty: UncertaintyInfo
    top_influences: List[InfluenceItem]
    nearest_bundles: List[NearestBundleItem]
    narrative: NarrativeInfo
    agent_votes: List[AgentVote]
    agent_consensus: AgentConsensus
    upsell_suggestions: List[str]
    business_insight: BusinessInsight


# ═══════════════════════════════════════════════════════════════════════
#  /simulate
# ═══════════════════════════════════════════════════════════════════════
class SimulateRequest(BaseModel):
    base: CustomerFields
    changes: Dict[str, Any]


class SimulatePrediction(BaseModel):
    predicted_bundle: int
    confidence: float
    probabilities: List[float]


class ProbabilityShift(BaseModel):
    class_: int = Field(alias="class")
    from_: float = Field(alias="from")
    to: float
    delta: float

    model_config = {"populate_by_name": True}


class SimulateDelta(BaseModel):
    bundle_changed: bool
    top_probability_shift: ProbabilityShift


class SimulateResponse(BaseModel):
    base: SimulatePrediction
    modified: SimulatePrediction
    delta: SimulateDelta
    upsell_suggestions: List[str]
