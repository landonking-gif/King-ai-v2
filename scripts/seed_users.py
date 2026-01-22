"""
User Seeder - Populates the database with initial users for authentication.
Run with: python scripts/seed_users.py
"""

import asyncio
import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'control-panel'))

from main import User, engine, Base, get_password_hash
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select

async def seed_users():
    async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    users = [
        {"username": "admin", "password": "admin123", "role": "Admin"},
        {"username": "operator", "password": "op123", "role": "Operator"},
        {"username": "analyst", "password": "an123", "role": "Analyst"},
        {"username": "auditor", "password": "au123", "role": "Auditor"},
        {"username": "developer", "password": "dev123", "role": "Developer"},
    ]

    async with async_session() as session:
        for user_data in users:
            existing = await session.execute(
                select(User).where(User.username == user_data["username"])
            )
            if not existing.scalar_one_or_none():
                hashed = get_password_hash(user_data["password"])
                user = User(
                    username=user_data["username"],
                    password_hash=hashed,
                    role=user_data["role"]
                )
                session.add(user)
        await session.commit()

    print("Users seeded successfully.")

if __name__ == "__main__":
    asyncio.run(seed_users())