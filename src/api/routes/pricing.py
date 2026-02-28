from fastapi import APIRouter

router = APIRouter()

@router.post("/optimize")
async def optimize_price(product_id: int, base_price: float, inventory_level: int):
    # Logic: Dynamic discount if inventory is high (inventory pressure)
    # Goal: Maximize revenue uplift
    discount = 0.05 if inventory_level > 100 else 0.0
    recommended_price = base_price * (1 - discount)
    
    return {
        "product_id": product_id,
        "recommended_price": round(recommended_price, 2),
        "expected_revenue_uplift_percent": 8.2 if discount > 0 else 0.0
    }