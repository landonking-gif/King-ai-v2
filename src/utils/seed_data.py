"""
Development Seed Data.
Realistic test data for development and testing.
"""

import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Type, TypeVar
from enum import Enum
import json

from src.utils.structured_logging import get_logger

logger = get_logger("seed_data")


T = TypeVar("T")


class SeedCategory(str, Enum):
    """Categories of seed data."""
    USERS = "users"
    PRODUCTS = "products"
    ORDERS = "orders"
    CUSTOMERS = "customers"
    BUSINESSES = "businesses"
    ANALYTICS = "analytics"
    AGENTS = "agents"
    APPROVALS = "approvals"


@dataclass
class SeedConfig:
    """Configuration for seed data generation."""
    num_users: int = 10
    num_products: int = 50
    num_customers: int = 100
    num_orders: int = 500
    num_businesses: int = 3
    date_range_days: int = 90
    random_seed: Optional[int] = None


# Sample data pools
FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael", "Linda",
    "William", "Barbara", "David", "Elizabeth", "Richard", "Susan", "Joseph", "Jessica",
    "Thomas", "Sarah", "Charles", "Karen", "Christopher", "Nancy", "Daniel", "Lisa",
    "Matthew", "Betty", "Anthony", "Margaret", "Mark", "Sandra", "Donald", "Ashley",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson",
    "Thomas", "Taylor", "Moore", "Jackson", "Martin", "Lee", "Perez", "Thompson",
    "White", "Harris", "Sanchez", "Clark", "Ramirez", "Lewis", "Robinson", "Walker",
]

PRODUCT_ADJECTIVES = [
    "Premium", "Professional", "Ultra", "Advanced", "Smart", "Classic", "Elite",
    "Essential", "Pro", "Plus", "Max", "Deluxe", "Custom", "Signature", "Limited",
]

PRODUCT_TYPES = [
    "Widget", "Gadget", "Device", "Tool", "Kit", "System", "Module", "Pack",
    "Set", "Bundle", "Solution", "Platform", "Suite", "Series", "Collection",
]

PRODUCT_CATEGORIES = [
    "Electronics", "Home & Garden", "Sports & Outdoors", "Health & Beauty",
    "Clothing & Accessories", "Books & Media", "Toys & Games", "Automotive",
    "Office Supplies", "Pet Supplies", "Food & Beverages", "Industrial",
]

COMPANY_SUFFIXES = [
    "Inc", "LLC", "Corp", "Co", "Ltd", "Group", "Industries", "Enterprises",
    "Solutions", "Technologies", "Systems", "Services", "Partners", "Holdings",
]

CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"),
    ("Houston", "TX"), ("Phoenix", "AZ"), ("Philadelphia", "PA"),
    ("San Antonio", "TX"), ("San Diego", "CA"), ("Dallas", "TX"),
    ("San Jose", "CA"), ("Austin", "TX"), ("Jacksonville", "FL"),
    ("Fort Worth", "TX"), ("Columbus", "OH"), ("San Francisco", "CA"),
    ("Charlotte", "NC"), ("Indianapolis", "IN"), ("Seattle", "WA"),
]


class RandomGenerator:
    """Generate random data."""
    
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
    
    def choice(self, items: List[T]) -> T:
        return self.rng.choice(items)
    
    def choices(self, items: List[T], k: int) -> List[T]:
        return self.rng.choices(items, k=k)
    
    def sample(self, items: List[T], k: int) -> List[T]:
        return self.rng.sample(items, min(k, len(items)))
    
    def randint(self, a: int, b: int) -> int:
        return self.rng.randint(a, b)
    
    def uniform(self, a: float, b: float) -> float:
        return self.rng.uniform(a, b)
    
    def gauss(self, mu: float, sigma: float) -> float:
        return max(0, self.rng.gauss(mu, sigma))
    
    def boolean(self, probability: float = 0.5) -> bool:
        return self.rng.random() < probability
    
    def uuid(self) -> str:
        return str(uuid.UUID(int=self.rng.getrandbits(128)))[:8]
    
    def email(self, name: str) -> str:
        domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "email.com"]
        clean_name = name.lower().replace(" ", ".")
        return f"{clean_name}{self.randint(1, 99)}@{self.choice(domains)}"
    
    def phone(self) -> str:
        return f"+1{self.randint(200, 999)}{self.randint(100, 999)}{self.randint(1000, 9999)}"
    
    def date_between(self, start: datetime, end: datetime) -> datetime:
        delta = end - start
        random_days = self.rng.random() * delta.days
        return start + timedelta(days=random_days)
    
    def past_date(self, days: int = 90) -> datetime:
        return self.date_between(
            datetime.utcnow() - timedelta(days=days),
            datetime.utcnow(),
        )


class SeedDataGenerator:
    """
    Generate realistic seed data for development.
    
    Features:
    - Users and customers
    - Products and inventory
    - Orders and transactions
    - Business entities
    - Analytics data
    """
    
    def __init__(self, config: SeedConfig = None):
        self.config = config or SeedConfig()
        self.rng = RandomGenerator(self.config.random_seed)
        
        # Generated data storage
        self.data: Dict[SeedCategory, List[Dict]] = {
            cat: [] for cat in SeedCategory
        }
    
    def generate_all(self) -> Dict[SeedCategory, List[Dict]]:
        """Generate all seed data."""
        logger.info("Generating seed data...")
        
        # Generate in order of dependencies
        self.generate_users()
        self.generate_businesses()
        self.generate_products()
        self.generate_customers()
        self.generate_orders()
        self.generate_analytics()
        self.generate_agents()
        self.generate_approvals()
        
        logger.info(
            "Seed data generated",
            extra={
                "users": len(self.data[SeedCategory.USERS]),
                "products": len(self.data[SeedCategory.PRODUCTS]),
                "customers": len(self.data[SeedCategory.CUSTOMERS]),
                "orders": len(self.data[SeedCategory.ORDERS]),
            },
        )
        
        return self.data
    
    def generate_users(self) -> List[Dict]:
        """Generate user accounts."""
        users = []
        
        for i in range(self.config.num_users):
            first_name = self.rng.choice(FIRST_NAMES)
            last_name = self.rng.choice(LAST_NAMES)
            
            user = {
                "id": self.rng.uuid(),
                "username": f"{first_name.lower()}.{last_name.lower()}{self.rng.randint(1, 99)}",
                "email": self.rng.email(f"{first_name} {last_name}"),
                "first_name": first_name,
                "last_name": last_name,
                "role": self.rng.choice(["admin", "user", "operator", "analyst"]),
                "is_active": self.rng.boolean(0.9),
                "created_at": self.rng.past_date(365).isoformat(),
                "last_login": self.rng.past_date(30).isoformat(),
            }
            users.append(user)
        
        self.data[SeedCategory.USERS] = users
        return users
    
    def generate_businesses(self) -> List[Dict]:
        """Generate business entities."""
        businesses = []
        
        business_types = ["E-commerce", "SaaS", "Marketplace", "Subscription", "Agency"]
        
        for i in range(self.config.num_businesses):
            name_word = self.rng.choice([
                "Tech", "Cloud", "Digital", "Smart", "Next", "Meta", "Cyber", "Data",
                "AI", "Quantum", "Future", "Alpha", "Prime", "Core", "Edge",
            ])
            
            business = {
                "id": self.rng.uuid(),
                "name": f"{name_word} {self.rng.choice(COMPANY_SUFFIXES)}",
                "type": self.rng.choice(business_types),
                "status": self.rng.choice(["active", "active", "active", "pending", "inactive"]),
                "revenue_ytd": round(self.rng.uniform(50000, 5000000), 2),
                "monthly_revenue": round(self.rng.uniform(5000, 500000), 2),
                "customer_count": self.rng.randint(100, 10000),
                "employee_count": self.rng.randint(1, 50),
                "founded_date": self.rng.past_date(1000).isoformat(),
                "industry": self.rng.choice([
                    "Technology", "Retail", "Healthcare", "Finance", "Education",
                    "Entertainment", "Manufacturing", "Real Estate", "Food & Beverage",
                ]),
                "settings": {
                    "timezone": "UTC",
                    "currency": "USD",
                    "auto_approve_threshold": self.rng.randint(100, 1000),
                },
            }
            businesses.append(business)
        
        self.data[SeedCategory.BUSINESSES] = businesses
        return businesses
    
    def generate_products(self) -> List[Dict]:
        """Generate product catalog."""
        products = []
        
        for i in range(self.config.num_products):
            base_price = round(self.rng.uniform(9.99, 499.99), 2)
            
            product = {
                "id": self.rng.uuid(),
                "sku": f"SKU-{self.rng.randint(10000, 99999)}",
                "name": f"{self.rng.choice(PRODUCT_ADJECTIVES)} {self.rng.choice(PRODUCT_TYPES)}",
                "category": self.rng.choice(PRODUCT_CATEGORIES),
                "description": f"High-quality {self.rng.choice(PRODUCT_TYPES).lower()} for all your needs.",
                "price": base_price,
                "cost": round(base_price * self.rng.uniform(0.3, 0.6), 2),
                "inventory": self.rng.randint(0, 500),
                "reorder_point": self.rng.randint(10, 50),
                "status": self.rng.choice(["active", "active", "active", "inactive", "discontinued"]),
                "weight_kg": round(self.rng.uniform(0.1, 10.0), 2),
                "created_at": self.rng.past_date(365).isoformat(),
                "tags": self.rng.sample(
                    ["bestseller", "new", "sale", "featured", "limited", "trending"],
                    k=self.rng.randint(0, 3),
                ),
            }
            products.append(product)
        
        self.data[SeedCategory.PRODUCTS] = products
        return products
    
    def generate_customers(self) -> List[Dict]:
        """Generate customer data."""
        customers = []
        
        for i in range(self.config.num_customers):
            first_name = self.rng.choice(FIRST_NAMES)
            last_name = self.rng.choice(LAST_NAMES)
            city, state = self.rng.choice(CITIES)
            
            customer = {
                "id": self.rng.uuid(),
                "email": self.rng.email(f"{first_name} {last_name}"),
                "first_name": first_name,
                "last_name": last_name,
                "phone": self.rng.phone(),
                "address": {
                    "street": f"{self.rng.randint(100, 9999)} {self.rng.choice(['Main', 'Oak', 'Maple', 'Pine', 'Cedar'])} St",
                    "city": city,
                    "state": state,
                    "zip": f"{self.rng.randint(10000, 99999)}",
                    "country": "US",
                },
                "total_orders": self.rng.randint(1, 50),
                "total_spent": round(self.rng.uniform(50, 5000), 2),
                "first_order_date": self.rng.past_date(365).isoformat(),
                "last_order_date": self.rng.past_date(90).isoformat(),
                "segment": self.rng.choice(["new", "active", "vip", "at_risk", "churned"]),
                "tags": self.rng.sample(
                    ["wholesale", "retail", "subscription", "loyalty_member"],
                    k=self.rng.randint(0, 2),
                ),
            }
            customers.append(customer)
        
        self.data[SeedCategory.CUSTOMERS] = customers
        return customers
    
    def generate_orders(self) -> List[Dict]:
        """Generate order history."""
        orders = []
        products = self.data[SeedCategory.PRODUCTS]
        customers = self.data[SeedCategory.CUSTOMERS]
        
        if not products or not customers:
            return orders
        
        statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
        status_weights = [0.05, 0.1, 0.15, 0.65, 0.05]
        
        for i in range(self.config.num_orders):
            customer = self.rng.choice(customers)
            order_date = self.rng.past_date(self.config.date_range_days)
            
            # Generate order items
            num_items = self.rng.randint(1, 5)
            order_products = self.rng.sample(products, num_items)
            
            items = []
            subtotal = 0
            
            for product in order_products:
                quantity = self.rng.randint(1, 3)
                price = product["price"]
                line_total = price * quantity
                
                items.append({
                    "product_id": product["id"],
                    "product_name": product["name"],
                    "sku": product["sku"],
                    "quantity": quantity,
                    "price": price,
                    "total": round(line_total, 2),
                })
                subtotal += line_total
            
            # Calculate totals
            tax_rate = 0.08
            shipping = round(self.rng.uniform(5, 25), 2) if subtotal < 100 else 0
            tax = round(subtotal * tax_rate, 2)
            total = round(subtotal + tax + shipping, 2)
            
            # Select status with weights
            status = self.rng.choices(statuses, weights=status_weights, k=1)[0]
            
            order = {
                "id": self.rng.uuid(),
                "order_number": f"ORD-{self.rng.randint(100000, 999999)}",
                "customer_id": customer["id"],
                "customer_email": customer["email"],
                "status": status,
                "items": items,
                "subtotal": round(subtotal, 2),
                "tax": tax,
                "shipping": shipping,
                "total": total,
                "payment_method": self.rng.choice(["credit_card", "paypal", "stripe", "apple_pay"]),
                "shipping_address": customer["address"],
                "created_at": order_date.isoformat(),
                "updated_at": (order_date + timedelta(hours=self.rng.randint(1, 48))).isoformat(),
            }
            orders.append(order)
        
        # Sort by date
        orders.sort(key=lambda x: x["created_at"])
        
        self.data[SeedCategory.ORDERS] = orders
        return orders
    
    def generate_analytics(self) -> List[Dict]:
        """Generate analytics data points."""
        analytics = []
        
        # Generate daily metrics
        for day_offset in range(self.config.date_range_days):
            date = datetime.utcnow() - timedelta(days=day_offset)
            
            # Add some seasonality (weekends lower)
            day_factor = 0.7 if date.weekday() >= 5 else 1.0
            
            metric = {
                "date": date.strftime("%Y-%m-%d"),
                "page_views": int(self.rng.gauss(5000, 1000) * day_factor),
                "unique_visitors": int(self.rng.gauss(2000, 400) * day_factor),
                "sessions": int(self.rng.gauss(3000, 600) * day_factor),
                "bounce_rate": round(self.rng.uniform(0.35, 0.55), 2),
                "avg_session_duration": round(self.rng.uniform(120, 300), 1),
                "conversion_rate": round(self.rng.uniform(0.02, 0.05), 4),
                "revenue": round(self.rng.gauss(5000, 1500) * day_factor, 2),
                "orders": int(self.rng.gauss(50, 15) * day_factor),
                "avg_order_value": round(self.rng.uniform(80, 150), 2),
            }
            analytics.append(metric)
        
        analytics.reverse()  # Chronological order
        
        self.data[SeedCategory.ANALYTICS] = analytics
        return analytics
    
    def generate_agents(self) -> List[Dict]:
        """Generate agent data."""
        agents = [
            {
                "id": "research_agent",
                "name": "Research Agent",
                "type": "research",
                "status": "active",
                "tasks_completed": self.rng.randint(100, 1000),
                "success_rate": round(self.rng.uniform(0.92, 0.99), 2),
                "avg_response_time_ms": self.rng.randint(500, 2000),
            },
            {
                "id": "finance_agent",
                "name": "Finance Agent",
                "type": "finance",
                "status": "active",
                "tasks_completed": self.rng.randint(100, 1000),
                "success_rate": round(self.rng.uniform(0.95, 0.99), 2),
                "avg_response_time_ms": self.rng.randint(800, 3000),
            },
            {
                "id": "legal_agent",
                "name": "Legal Agent",
                "type": "legal",
                "status": "active",
                "tasks_completed": self.rng.randint(50, 500),
                "success_rate": round(self.rng.uniform(0.97, 0.99), 2),
                "avg_response_time_ms": self.rng.randint(1000, 5000),
            },
            {
                "id": "content_agent",
                "name": "Content Agent",
                "type": "content",
                "status": "active",
                "tasks_completed": self.rng.randint(200, 2000),
                "success_rate": round(self.rng.uniform(0.90, 0.98), 2),
                "avg_response_time_ms": self.rng.randint(2000, 8000),
            },
            {
                "id": "commerce_agent",
                "name": "Commerce Agent",
                "type": "commerce",
                "status": "active",
                "tasks_completed": self.rng.randint(300, 3000),
                "success_rate": round(self.rng.uniform(0.93, 0.99), 2),
                "avg_response_time_ms": self.rng.randint(400, 1500),
            },
        ]
        
        self.data[SeedCategory.AGENTS] = agents
        return agents
    
    def generate_approvals(self) -> List[Dict]:
        """Generate approval requests."""
        approvals = []
        
        action_types = [
            ("financial", "Budget increase request", 5000),
            ("legal", "Contract review", 2500),
            ("marketing", "Campaign launch", 3000),
            ("hiring", "New position approval", 8000),
            ("product", "Feature development", 4000),
        ]
        
        for i in range(20):
            action_type, description, base_amount = self.rng.choice(action_types)
            amount = round(base_amount * self.rng.uniform(0.5, 2.0), 2)
            created = self.rng.past_date(30)
            
            approval = {
                "id": self.rng.uuid(),
                "action_type": action_type,
                "description": description,
                "amount": amount,
                "status": self.rng.choice(["pending", "approved", "rejected", "escalated"]),
                "risk_level": self.rng.choice(["low", "medium", "high"]),
                "requester": self.rng.choice(self.data[SeedCategory.AGENTS])["id"],
                "approver": self.rng.choice(self.data[SeedCategory.USERS])["id"] if self.data[SeedCategory.USERS] else None,
                "created_at": created.isoformat(),
                "updated_at": (created + timedelta(hours=self.rng.randint(1, 48))).isoformat(),
            }
            approvals.append(approval)
        
        self.data[SeedCategory.APPROVALS] = approvals
        return approvals
    
    def to_json(self, pretty: bool = True) -> str:
        """Export data as JSON."""
        indent = 2 if pretty else None
        return json.dumps(
            {cat.value: data for cat, data in self.data.items()},
            indent=indent,
            default=str,
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save data to JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())
        logger.info(f"Saved seed data to {filepath}")


# Factory function
def generate_seed_data(config: SeedConfig = None) -> Dict[SeedCategory, List[Dict]]:
    """Generate seed data with the given configuration."""
    generator = SeedDataGenerator(config)
    return generator.generate_all()


# Global generator
_generator: Optional[SeedDataGenerator] = None


def get_seed_generator(config: SeedConfig = None) -> SeedDataGenerator:
    """Get or create the global seed data generator."""
    global _generator
    if _generator is None or config is not None:
        _generator = SeedDataGenerator(config)
    return _generator
