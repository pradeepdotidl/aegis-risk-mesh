import strawberry
from typing import List

# Strawberry requires its own type definitions, mapping closely to our Pydantic models
@strawberry.type
class GQLRiskFeature:
    feature_name: str
    value: float
    source: str

@strawberry.type
class GQLRiskAssessment:
    entity_name: str
    overall_risk_score: float
    key_drivers: List[str]
    
# Mock database query for demonstration
def get_historical_assessments(entity: str) -> List[GQLRiskAssessment]:
    # In a real app, this queries your PostgreSQL/Chroma DB
    return [
        GQLRiskAssessment(
            entity_name=entity, 
            overall_risk_score=85.5, 
            key_drivers=["Market Volatility Index"]
        )
    ]

@strawberry.type
class Query:
    @strawberry.field
    def historical_risk(self, entity_name: str) -> List[GQLRiskAssessment]:
        """Query past risk assessments by entity name."""
        print("inside graphql")
        return get_historical_assessments(entity_name)

schema = strawberry.Schema(query=Query)