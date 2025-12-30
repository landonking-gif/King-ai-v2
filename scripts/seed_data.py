"""
Demo Data Seeder - Populates the database with initial sample businesses.
Run with: py -3 scripts/seed_data.py
"""

import asyncio
from uuid import uuid4
from src.database.connection import init_db, get_db
from src.database.models import BusinessUnit, BusinessStatus

DEMO_BUSINESSES = [
    {
        "name": "PetPal Pet Supplies",
        "type": "dropshipping",
        "status": BusinessStatus.OPERATION,
        "revenue": 12500.50,
        "expenses": 7800.20,
        "kpis": {"conversion_rate": 0.024, "active_ads": 12}
    },
    {
        "name": "CodeDoc AI SaaS",
        "type": "saas",
        "status": BusinessStatus.VALIDATION,
        "revenue": 2400.00,
        "expenses": 150.00,
        "kpis": {"users": 450, "churn": 0.05}
    },
    {
        "name": "GreenHome Eco-Shop",
        "type": "dropshipping",
        "status": BusinessStatus.SETUP,
        "revenue": 0.00,
        "expenses": 1200.00,
        "kpis": {"products_listed": 150}
    },
    {
        "name": "MarketInsight Analytics",
        "type": "service",
        "status": BusinessStatus.DISCOVERY,
        "revenue": 0.00,
        "expenses": 0.00,
        "kpis": {}
    }
]

async def seed():
    print("Initializing database for seeding...")
    await init_db()
    
    async with get_db() as db:
        print(f"Purging existing data...")
        # Optional: Add logic to clear tables if needed
        # await db.execute("DELETE FROM business_units")
        
        print(f"Seeding {len(DEMO_BUSINESSES)} demo businesses...")
        for biz in DEMO_BUSINESSES:
            unit = BusinessUnit(
                id=str(uuid4()),
                name=biz["name"],
                type=biz["type"],
                status=biz["status"],
                total_revenue=biz["revenue"],
                total_expenses=biz["expenses"],
                kpis=biz["kpis"]
            )
            db.add(unit)
        
        await db.commit()
    
    print("\n✅ Seeding complete!")

if __name__ == "__main__":
    asyncio.run(seed())
