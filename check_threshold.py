from kite import Kite

ai = Kite(config={
    "semantic_router_threshold": 0.25
})

print(f"Config threshold: {ai.config.get('semantic_router_threshold')}")
print(f"Router threshold: {ai.semantic_router.confidence_threshold}")
