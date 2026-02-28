class Ranker:
    def __init__(self, w_relevance=0.6, w_margin=0.3, w_inventory=0.1):
        self.w_relevance = w_relevance
        self.w_margin = w_margin
        self.w_inventory = w_inventory
        self.max_stock_baseline = 500  # Based on seed_db logic

    def rank(self, candidates):
        """
        Applies business logic: final_score = (0.6*rel) + (0.3*margin) + (0.1*stock)
        """
        scored_items = []

        for item in candidates:
            # 1. Convert pgvector distance to similarity (relevance)
            relevance = 1 - item.distance

            # 2. Normalize Margin (Assuming $15 is a 'high' margin in your data)
            normalized_margin = min(item.margin / 15.0, 1.0)

            # 3. Calculate Inventory Pressure (Prefer items with higher stock)
            inventory_factor = min(item.stock / self.max_stock_baseline, 1.0)

            # 4. Final Weighted Calculation
            final_score = (
                (self.w_relevance * relevance) +
                (self.w_margin * normalized_margin) +
                (self.w_inventory * inventory_factor)
            )

            scored_items.append({
                "product_id": item.product_id,
                "product_name": item.product_name,
                "score": round(final_score, 4),
                "price": item.price,
                "margin": item.margin,
                "stock": item.stock
            })

        # Sort descending by the new business-aware score
        return sorted(scored_items, key=lambda x: x['score'], reverse=True)