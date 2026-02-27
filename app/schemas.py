from pydantic import BaseModel, Field, field_validator
from typing import Literal

class ShopperFeatures(BaseModel):
    Administrative: float
    Administrative_Duration: float
    Informational: float
    Informational_Duration: float
    ProductRelated: float
    ProductRelated_Duration: float
    BounceRates: float
    ExitRates: float
    PageValues: float
    SpecialDay: float
    Month: Literal["Jan","Feb","Mar","Apr","May","June","Jul","Aug","Sep","Oct","Nov","Dec"]
    OperatingSystems: int
    Browser: int
    Region: int
    TrafficType: int
    VisitorType: Literal["Returning_Visitor","New_Visitor","Other"]
    Weekend: bool

    @field_validator("Administrative", "Administrative_Duration", "Informational",
               "Informational_Duration", "ProductRelated", "ProductRelated_Duration",
               "BounceRates", "ExitRates", "PageValues", "SpecialDay",
               "OperatingSystems", "Browser", "Region", "TrafficType")
    def check_non_negative(cls, v):
        if v < 0:
            raise ValueError("Numeric features must be non-negative")
        return v


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = No purchase, 1 = Purchase")
    probability: float = Field(..., description="Probability of purchase (0-1)")

