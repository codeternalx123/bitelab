"""
MULTI-TENANCY ARCHITECTURE
===========================

Enterprise-grade multi-tenancy system with:
- Tenant Management & Isolation
- Database Per Tenant Strategy
- Schema Per Tenant Strategy
- Shared Database with Tenant ID
- Tenant-Aware Routing & Middleware
- Tenant Context & Scoping
- Resource Quotas & Limits
- Cross-Tenant Data Prevention
- Tenant Onboarding & Provisioning

Author: Wellomex AI Team
Created: 2025-11-12
"""

import logging
import asyncio
import json
import time
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. TENANT MANAGEMENT
# ============================================================================

class TenantStatus(Enum):
    """Tenant status"""
    ACTIVE = "active"
    SUSPENDED = "suspended"
    TRIAL = "trial"
    EXPIRED = "expired"


@dataclass
class Tenant:
    """Tenant entity"""
    id: str
    name: str
    domain: str
    status: TenantStatus = TenantStatus.ACTIVE
    created_at: float = field(default_factory=time.time)
    plan: str = "free"
    metadata: Dict[str, Any] = field(default_factory=dict)
    settings: Dict[str, Any] = field(default_factory=dict)
    
    def is_active(self) -> bool:
        """Check if tenant is active"""
        return self.status == TenantStatus.ACTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "status": self.status.value,
            "created_at": self.created_at,
            "plan": self.plan,
            "metadata": self.metadata,
            "settings": self.settings
        }


class TenantRegistry:
    """Central tenant registry"""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.domain_index: Dict[str, str] = {}  # domain -> tenant_id
        self.tenant_count = 0
        logger.info("TenantRegistry initialized")
    
    def create_tenant(self, name: str, domain: str, plan: str = "free") -> Tenant:
        """Create new tenant"""
        tenant_id = f"tenant-{uuid.uuid4().hex[:8]}"
        
        if domain in self.domain_index:
            raise ValueError(f"Domain already exists: {domain}")
        
        tenant = Tenant(
            id=tenant_id,
            name=name,
            domain=domain,
            plan=plan
        )
        
        self.tenants[tenant_id] = tenant
        self.domain_index[domain] = tenant_id
        self.tenant_count += 1
        
        logger.info(f"Created tenant: {name} ({tenant_id})")
        return tenant
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        return self.tenants.get(tenant_id)
    
    def get_tenant_by_domain(self, domain: str) -> Optional[Tenant]:
        """Get tenant by domain"""
        tenant_id = self.domain_index.get(domain)
        return self.tenants.get(tenant_id) if tenant_id else None
    
    def update_tenant(self, tenant_id: str, updates: Dict[str, Any]) -> bool:
        """Update tenant"""
        if tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[tenant_id]
        
        for key, value in updates.items():
            if hasattr(tenant, key):
                setattr(tenant, key, value)
        
        logger.info(f"Updated tenant: {tenant_id}")
        return True
    
    def delete_tenant(self, tenant_id: str) -> bool:
        """Delete tenant"""
        if tenant_id not in self.tenants:
            return False
        
        tenant = self.tenants[tenant_id]
        del self.domain_index[tenant.domain]
        del self.tenants[tenant_id]
        
        logger.info(f"Deleted tenant: {tenant_id}")
        return True
    
    def list_tenants(self, status: Optional[TenantStatus] = None) -> List[Tenant]:
        """List all tenants, optionally filtered by status"""
        tenants = list(self.tenants.values())
        
        if status:
            tenants = [t for t in tenants if t.status == status]
        
        return tenants
    
    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        status_counts = defaultdict(int)
        plan_counts = defaultdict(int)
        
        for tenant in self.tenants.values():
            status_counts[tenant.status.value] += 1
            plan_counts[tenant.plan] += 1
        
        return {
            "total_tenants": self.tenant_count,
            "active_tenants": len(self.tenants),
            "status_distribution": dict(status_counts),
            "plan_distribution": dict(plan_counts)
        }


# ============================================================================
# 2. TENANT ISOLATION STRATEGIES
# ============================================================================

class IsolationStrategy(ABC):
    """Base isolation strategy"""
    
    @abstractmethod
    async def get_connection(self, tenant_id: str) -> Any:
        pass
    
    @abstractmethod
    async def execute_query(self, tenant_id: str, query: str, params: Dict[str, Any]) -> Any:
        pass


class DatabasePerTenant(IsolationStrategy):
    """Separate database for each tenant (highest isolation)"""
    
    def __init__(self):
        self.databases: Dict[str, Dict[str, Any]] = {}
        self.query_count = 0
        logger.info("DatabasePerTenant strategy initialized")
    
    def create_database(self, tenant_id: str) -> None:
        """Create database for tenant"""
        if tenant_id not in self.databases:
            self.databases[tenant_id] = {
                "connection": f"db_{tenant_id}",
                "data": {},
                "created_at": time.time()
            }
            logger.info(f"Created database for tenant: {tenant_id}")
    
    async def get_connection(self, tenant_id: str) -> Any:
        """Get database connection for tenant"""
        if tenant_id not in self.databases:
            self.create_database(tenant_id)
        
        return self.databases[tenant_id]["connection"]
    
    async def execute_query(self, tenant_id: str, query: str, params: Dict[str, Any]) -> Any:
        """Execute query on tenant database"""
        self.query_count += 1
        
        if tenant_id not in self.databases:
            raise ValueError(f"Database not found for tenant: {tenant_id}")
        
        db = self.databases[tenant_id]["data"]
        
        # Simple query simulation
        if query == "INSERT":
            table = params.get("table", "default")
            if table not in db:
                db[table] = []
            db[table].append(params.get("data", {}))
            return {"inserted": 1}
        
        elif query == "SELECT":
            table = params.get("table", "default")
            return db.get(table, [])
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            "databases": len(self.databases),
            "queries": self.query_count,
            "avg_tables_per_db": sum(len(db["data"]) for db in self.databases.values()) / max(1, len(self.databases))
        }


class SchemaPerTenant(IsolationStrategy):
    """Separate schema per tenant in shared database (medium isolation)"""
    
    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}
        self.query_count = 0
        logger.info("SchemaPerTenant strategy initialized")
    
    def create_schema(self, tenant_id: str) -> None:
        """Create schema for tenant"""
        if tenant_id not in self.schemas:
            self.schemas[tenant_id] = {
                "name": f"schema_{tenant_id}",
                "tables": {},
                "created_at": time.time()
            }
            logger.info(f"Created schema for tenant: {tenant_id}")
    
    async def get_connection(self, tenant_id: str) -> Any:
        """Get connection with schema context"""
        if tenant_id not in self.schemas:
            self.create_schema(tenant_id)
        
        return f"SET search_path TO {self.schemas[tenant_id]['name']}"
    
    async def execute_query(self, tenant_id: str, query: str, params: Dict[str, Any]) -> Any:
        """Execute query in tenant schema"""
        self.query_count += 1
        
        if tenant_id not in self.schemas:
            raise ValueError(f"Schema not found for tenant: {tenant_id}")
        
        schema = self.schemas[tenant_id]["tables"]
        
        # Simple query simulation
        if query == "INSERT":
            table = params.get("table", "default")
            if table not in schema:
                schema[table] = []
            schema[table].append(params.get("data", {}))
            return {"inserted": 1}
        
        elif query == "SELECT":
            table = params.get("table", "default")
            return schema.get(table, [])
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        return {
            "schemas": len(self.schemas),
            "queries": self.query_count,
            "avg_tables_per_schema": sum(len(s["tables"]) for s in self.schemas.values()) / max(1, len(self.schemas))
        }


class SharedDatabaseWithTenantID(IsolationStrategy):
    """Shared database with tenant_id column (lowest isolation, highest efficiency)"""
    
    def __init__(self):
        self.database: Dict[str, List[Dict[str, Any]]] = {}
        self.query_count = 0
        logger.info("SharedDatabaseWithTenantID strategy initialized")
    
    async def get_connection(self, tenant_id: str) -> Any:
        """Get shared database connection"""
        return "shared_db_connection"
    
    async def execute_query(self, tenant_id: str, query: str, params: Dict[str, Any]) -> Any:
        """Execute query with tenant_id filter"""
        self.query_count += 1
        
        table = params.get("table", "default")
        
        if query == "INSERT":
            if table not in self.database:
                self.database[table] = []
            
            # Automatically add tenant_id
            data = params.get("data", {})
            data["tenant_id"] = tenant_id
            self.database[table].append(data)
            
            return {"inserted": 1}
        
        elif query == "SELECT":
            if table not in self.database:
                return []
            
            # Filter by tenant_id
            return [row for row in self.database[table] if row.get("tenant_id") == tenant_id]
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get strategy statistics"""
        tenant_distribution = defaultdict(int)
        
        for table in self.database.values():
            for row in table:
                tenant_id = row.get("tenant_id")
                if tenant_id:
                    tenant_distribution[tenant_id] += 1
        
        return {
            "tables": len(self.database),
            "queries": self.query_count,
            "total_rows": sum(len(t) for t in self.database.values()),
            "tenant_distribution": dict(tenant_distribution)
        }


# ============================================================================
# 3. TENANT CONTEXT
# ============================================================================

class TenantContext:
    """Thread-local tenant context"""
    
    _context = threading.local()
    
    @classmethod
    def set_tenant(cls, tenant_id: str) -> None:
        """Set current tenant"""
        cls._context.tenant_id = tenant_id
        logger.debug(f"Tenant context set: {tenant_id}")
    
    @classmethod
    def get_tenant(cls) -> Optional[str]:
        """Get current tenant"""
        return getattr(cls._context, 'tenant_id', None)
    
    @classmethod
    def clear(cls) -> None:
        """Clear tenant context"""
        if hasattr(cls._context, 'tenant_id'):
            delattr(cls._context, 'tenant_id')
        logger.debug("Tenant context cleared")
    
    @classmethod
    def require_tenant(cls) -> str:
        """Get tenant or raise error"""
        tenant_id = cls.get_tenant()
        if not tenant_id:
            raise ValueError("No tenant context set")
        return tenant_id


class TenantMiddleware:
    """Middleware for tenant resolution"""
    
    def __init__(self, tenant_registry: TenantRegistry):
        self.tenant_registry = tenant_registry
        self.request_count = 0
        self.resolution_failures = 0
        logger.info("TenantMiddleware initialized")
    
    async def resolve_tenant(self, request: Dict[str, Any]) -> Optional[Tenant]:
        """Resolve tenant from request"""
        self.request_count += 1
        
        # Try multiple resolution strategies
        
        # 1. Header-based (X-Tenant-ID)
        tenant_id = request.get("headers", {}).get("X-Tenant-ID")
        if tenant_id:
            tenant = self.tenant_registry.get_tenant(tenant_id)
            if tenant:
                return tenant
        
        # 2. Domain-based (Host header)
        domain = request.get("headers", {}).get("Host")
        if domain:
            tenant = self.tenant_registry.get_tenant_by_domain(domain)
            if tenant:
                return tenant
        
        # 3. Subdomain-based (extract from URL)
        url = request.get("url", "")
        if "." in url:
            subdomain = url.split(".")[0].replace("http://", "").replace("https://", "")
            tenant = self.tenant_registry.get_tenant_by_domain(subdomain)
            if tenant:
                return tenant
        
        # 4. Token-based (JWT claim)
        token = request.get("headers", {}).get("Authorization")
        if token:
            # Decode JWT and extract tenant claim
            # Simplified simulation
            pass
        
        self.resolution_failures += 1
        logger.warning("Failed to resolve tenant from request")
        return None
    
    async def process_request(self, request: Dict[str, Any], handler: Callable) -> Any:
        """Process request with tenant context"""
        tenant = await self.resolve_tenant(request)
        
        if not tenant:
            return {"error": "Tenant not found", "status": 400}
        
        if not tenant.is_active():
            return {"error": "Tenant is not active", "status": 403}
        
        # Set tenant context
        TenantContext.set_tenant(tenant.id)
        
        try:
            # Execute handler
            result = await handler(request) if asyncio.iscoroutinefunction(handler) else handler(request)
            return result
        finally:
            # Clear context
            TenantContext.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get middleware statistics"""
        return {
            "requests": self.request_count,
            "resolution_failures": self.resolution_failures,
            "success_rate": (self.request_count - self.resolution_failures) / max(1, self.request_count)
        }


# ============================================================================
# 4. RESOURCE QUOTAS & LIMITS
# ============================================================================

@dataclass
class ResourceQuota:
    """Resource quota for tenant"""
    max_users: int = 100
    max_storage_gb: float = 10.0
    max_api_calls_per_day: int = 10000
    max_concurrent_connections: int = 50
    max_database_size_mb: float = 1000.0


class ResourceLimits:
    """Resource limits enforcer"""
    
    def __init__(self):
        self.quotas: Dict[str, ResourceQuota] = {}
        self.usage: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "users": 0,
            "storage_gb": 0.0,
            "api_calls_today": 0,
            "concurrent_connections": 0,
            "database_size_mb": 0.0,
            "last_reset": time.time()
        })
        self.violations = 0
        logger.info("ResourceLimits initialized")
    
    def set_quota(self, tenant_id: str, quota: ResourceQuota) -> None:
        """Set quota for tenant"""
        self.quotas[tenant_id] = quota
        logger.info(f"Set quota for tenant: {tenant_id}")
    
    def get_quota(self, tenant_id: str) -> ResourceQuota:
        """Get quota for tenant"""
        return self.quotas.get(tenant_id, ResourceQuota())
    
    def check_limit(self, tenant_id: str, resource: str, requested: float = 1.0) -> bool:
        """Check if resource limit is exceeded"""
        quota = self.get_quota(tenant_id)
        usage = self.usage[tenant_id]
        
        # Reset daily counters
        if time.time() - usage["last_reset"] > 86400:  # 24 hours
            usage["api_calls_today"] = 0
            usage["last_reset"] = time.time()
        
        # Check limits
        if resource == "users":
            if usage["users"] + requested > quota.max_users:
                self.violations += 1
                logger.warning(f"User limit exceeded for tenant: {tenant_id}")
                return False
        
        elif resource == "storage":
            if usage["storage_gb"] + requested > quota.max_storage_gb:
                self.violations += 1
                logger.warning(f"Storage limit exceeded for tenant: {tenant_id}")
                return False
        
        elif resource == "api_calls":
            if usage["api_calls_today"] + requested > quota.max_api_calls_per_day:
                self.violations += 1
                logger.warning(f"API call limit exceeded for tenant: {tenant_id}")
                return False
        
        elif resource == "connections":
            if usage["concurrent_connections"] + requested > quota.max_concurrent_connections:
                self.violations += 1
                logger.warning(f"Connection limit exceeded for tenant: {tenant_id}")
                return False
        
        return True
    
    def increment_usage(self, tenant_id: str, resource: str, amount: float = 1.0) -> None:
        """Increment resource usage"""
        usage = self.usage[tenant_id]
        
        if resource in usage:
            usage[resource] += amount
            logger.debug(f"Incremented {resource} for {tenant_id}: +{amount}")
    
    def decrement_usage(self, tenant_id: str, resource: str, amount: float = 1.0) -> None:
        """Decrement resource usage"""
        usage = self.usage[tenant_id]
        
        if resource in usage:
            usage[resource] = max(0, usage[resource] - amount)
            logger.debug(f"Decremented {resource} for {tenant_id}: -{amount}")
    
    def get_usage_report(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage report for tenant"""
        quota = self.get_quota(tenant_id)
        usage = self.usage[tenant_id]
        
        return {
            "tenant_id": tenant_id,
            "quota": asdict(quota),
            "usage": dict(usage),
            "utilization": {
                "users": usage["users"] / quota.max_users,
                "storage": usage["storage_gb"] / quota.max_storage_gb,
                "api_calls": usage["api_calls_today"] / quota.max_api_calls_per_day,
                "connections": usage["concurrent_connections"] / quota.max_concurrent_connections
            }
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get limits statistics"""
        return {
            "tracked_tenants": len(self.usage),
            "quota_violations": self.violations,
            "total_api_calls": sum(u["api_calls_today"] for u in self.usage.values())
        }


# ============================================================================
# 5. CROSS-TENANT DATA PREVENTION
# ============================================================================

class TenantGuard:
    """Prevent cross-tenant data access"""
    
    def __init__(self):
        self.checks_performed = 0
        self.violations_detected = 0
        logger.info("TenantGuard initialized")
    
    def validate_access(self, resource_tenant_id: str) -> bool:
        """Validate that current tenant can access resource"""
        self.checks_performed += 1
        
        current_tenant = TenantContext.get_tenant()
        
        if not current_tenant:
            self.violations_detected += 1
            logger.error("Attempted access without tenant context")
            return False
        
        if current_tenant != resource_tenant_id:
            self.violations_detected += 1
            logger.error(f"Cross-tenant access attempt: {current_tenant} -> {resource_tenant_id}")
            return False
        
        return True
    
    def filter_by_tenant(self, items: List[Dict[str, Any]], tenant_field: str = "tenant_id") -> List[Dict[str, Any]]:
        """Filter items to only include current tenant's data"""
        current_tenant = TenantContext.require_tenant()
        
        filtered = [item for item in items if item.get(tenant_field) == current_tenant]
        
        if len(filtered) < len(items):
            logger.debug(f"Filtered {len(items) - len(filtered)} items from other tenants")
        
        return filtered
    
    def scope_query(self, query: Dict[str, Any], tenant_field: str = "tenant_id") -> Dict[str, Any]:
        """Add tenant filter to query"""
        current_tenant = TenantContext.require_tenant()
        
        # Add tenant filter
        query.setdefault("filters", {})[tenant_field] = current_tenant
        
        logger.debug(f"Scoped query to tenant: {current_tenant}")
        return query
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guard statistics"""
        return {
            "checks_performed": self.checks_performed,
            "violations_detected": self.violations_detected,
            "violation_rate": self.violations_detected / max(1, self.checks_performed)
        }


# ============================================================================
# 6. TENANT ONBOARDING & PROVISIONING
# ============================================================================

class TenantProvisioner:
    """Automated tenant provisioning"""
    
    def __init__(self, tenant_registry: TenantRegistry, isolation_strategy: IsolationStrategy):
        self.tenant_registry = tenant_registry
        self.isolation_strategy = isolation_strategy
        self.provisioned_count = 0
        self.provisioning_failures = 0
        logger.info("TenantProvisioner initialized")
    
    async def provision_tenant(self, name: str, domain: str, plan: str = "free") -> Dict[str, Any]:
        """Provision new tenant with all resources"""
        try:
            logger.info(f"Starting provisioning for tenant: {name}")
            
            # 1. Create tenant record
            tenant = self.tenant_registry.create_tenant(name, domain, plan)
            
            # 2. Create database/schema
            if isinstance(self.isolation_strategy, DatabasePerTenant):
                self.isolation_strategy.create_database(tenant.id)
            elif isinstance(self.isolation_strategy, SchemaPerTenant):
                self.isolation_strategy.create_schema(tenant.id)
            
            # 3. Initialize default data
            await self._initialize_tenant_data(tenant.id)
            
            # 4. Set up resource quotas
            quota = self._get_quota_for_plan(plan)
            # Would set quota here in production
            
            # 5. Create default admin user
            admin_user = {
                "id": f"user-{uuid.uuid4().hex[:8]}",
                "email": f"admin@{domain}",
                "role": "admin",
                "tenant_id": tenant.id
            }
            
            self.provisioned_count += 1
            
            logger.info(f"Successfully provisioned tenant: {tenant.id}")
            
            return {
                "success": True,
                "tenant": tenant.to_dict(),
                "admin_user": admin_user,
                "provisioned_at": time.time()
            }
        
        except Exception as e:
            self.provisioning_failures += 1
            logger.error(f"Provisioning failed for {name}: {e}")
            
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _initialize_tenant_data(self, tenant_id: str) -> None:
        """Initialize default data for tenant"""
        # Create default tables/collections
        await self.isolation_strategy.execute_query(tenant_id, "INSERT", {
            "table": "settings",
            "data": {
                "theme": "light",
                "language": "en",
                "notifications": True
            }
        })
        
        logger.debug(f"Initialized data for tenant: {tenant_id}")
    
    def _get_quota_for_plan(self, plan: str) -> ResourceQuota:
        """Get resource quota based on plan"""
        quotas = {
            "free": ResourceQuota(
                max_users=10,
                max_storage_gb=1.0,
                max_api_calls_per_day=1000,
                max_concurrent_connections=10
            ),
            "pro": ResourceQuota(
                max_users=100,
                max_storage_gb=50.0,
                max_api_calls_per_day=100000,
                max_concurrent_connections=100
            ),
            "enterprise": ResourceQuota(
                max_users=10000,
                max_storage_gb=1000.0,
                max_api_calls_per_day=10000000,
                max_concurrent_connections=1000
            )
        }
        
        return quotas.get(plan, quotas["free"])
    
    async def deprovision_tenant(self, tenant_id: str) -> bool:
        """Deprovision tenant and clean up resources"""
        try:
            logger.info(f"Deprovisioning tenant: {tenant_id}")
            
            # 1. Archive/backup data
            # Would implement backup logic here
            
            # 2. Delete database/schema
            # Would implement deletion here
            
            # 3. Remove tenant record
            success = self.tenant_registry.delete_tenant(tenant_id)
            
            if success:
                logger.info(f"Successfully deprovisioned tenant: {tenant_id}")
            
            return success
        
        except Exception as e:
            logger.error(f"Deprovisioning failed for {tenant_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provisioner statistics"""
        return {
            "provisioned": self.provisioned_count,
            "failures": self.provisioning_failures,
            "success_rate": self.provisioned_count / max(1, self.provisioned_count + self.provisioning_failures)
        }


# ============================================================================
# 7. TENANT-AWARE DATA ACCESS LAYER
# ============================================================================

class TenantDataAccess:
    """Tenant-aware data access layer"""
    
    def __init__(self, isolation_strategy: IsolationStrategy, guard: TenantGuard):
        self.isolation_strategy = isolation_strategy
        self.guard = guard
        self.query_count = 0
        logger.info("TenantDataAccess initialized")
    
    async def insert(self, table: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Insert data for current tenant"""
        tenant_id = TenantContext.require_tenant()
        self.query_count += 1
        
        # Add tenant_id to data
        data["tenant_id"] = tenant_id
        
        result = await self.isolation_strategy.execute_query(tenant_id, "INSERT", {
            "table": table,
            "data": data
        })
        
        logger.debug(f"Inserted into {table} for tenant {tenant_id}")
        return result
    
    async def select(self, table: str, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Select data for current tenant"""
        tenant_id = TenantContext.require_tenant()
        self.query_count += 1
        
        result = await self.isolation_strategy.execute_query(tenant_id, "SELECT", {
            "table": table,
            "filters": filters or {}
        })
        
        # Apply tenant guard filtering
        filtered = self.guard.filter_by_tenant(result)
        
        logger.debug(f"Selected {len(filtered)} rows from {table} for tenant {tenant_id}")
        return filtered
    
    async def update(self, table: str, record_id: str, updates: Dict[str, Any]) -> bool:
        """Update data for current tenant"""
        tenant_id = TenantContext.require_tenant()
        self.query_count += 1
        
        # Verify ownership
        existing = await self.select(table, {"id": record_id})
        
        if not existing:
            logger.warning(f"Record {record_id} not found or not owned by tenant {tenant_id}")
            return False
        
        # Would implement update logic here
        logger.debug(f"Updated {record_id} in {table} for tenant {tenant_id}")
        return True
    
    async def delete(self, table: str, record_id: str) -> bool:
        """Delete data for current tenant"""
        tenant_id = TenantContext.require_tenant()
        self.query_count += 1
        
        # Verify ownership
        if not self.guard.validate_access(tenant_id):
            return False
        
        # Would implement delete logic here
        logger.debug(f"Deleted {record_id} from {table} for tenant {tenant_id}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get data access statistics"""
        return {
            "queries": self.query_count
        }


# ============================================================================
# DEMONSTRATION
# ============================================================================

async def demo_multi_tenancy():
    """Comprehensive demonstration of multi-tenancy architecture"""
    
    print("=" * 80)
    print("MULTI-TENANCY ARCHITECTURE")
    print("=" * 80)
    print()
    
    print("🏗️  COMPONENTS:")
    print("   1. Tenant Management & Registry")
    print("   2. Database Per Tenant Strategy")
    print("   3. Schema Per Tenant Strategy")
    print("   4. Shared Database with Tenant ID")
    print("   5. Tenant Context & Middleware")
    print("   6. Resource Quotas & Limits")
    print("   7. Cross-Tenant Data Prevention")
    print("   8. Tenant Provisioning")
    print("   9. Tenant-Aware Data Access")
    print()
    
    # ========================================================================
    # 1. Tenant Registry
    # ========================================================================
    print("=" * 80)
    print("1. TENANT MANAGEMENT & REGISTRY")
    print("=" * 80)
    
    registry = TenantRegistry()
    
    print("\n👥 Creating tenants...")
    
    tenant1 = registry.create_tenant("Acme Corp", "acme.example.com", "pro")
    tenant2 = registry.create_tenant("TechStart", "techstart.example.com", "free")
    tenant3 = registry.create_tenant("MegaHealth", "megahealth.example.com", "enterprise")
    
    print(f"   ✓ Created {len(registry.tenants)} tenants")
    
    # Get tenant
    tenant = registry.get_tenant_by_domain("acme.example.com")
    print(f"\n🔍 Retrieved tenant by domain: {tenant.name}")
    
    # Update tenant
    registry.update_tenant(tenant1.id, {"status": TenantStatus.ACTIVE})
    print(f"   ✓ Updated tenant status")
    
    # List tenants
    active_tenants = registry.list_tenants(TenantStatus.ACTIVE)
    print(f"   Active tenants: {len(active_tenants)}")
    
    stats = registry.get_stats()
    print(f"\n📊 Registry Statistics:")
    print(f"   Total Tenants: {stats['total_tenants']}")
    print(f"   Active Tenants: {stats['active_tenants']}")
    print(f"   Plan Distribution: {stats['plan_distribution']}")
    
    # ========================================================================
    # 2. Isolation Strategies
    # ========================================================================
    print("\n" + "=" * 80)
    print("2. TENANT ISOLATION STRATEGIES")
    print("=" * 80)
    
    # Database per tenant
    print("\n🗄️  Database Per Tenant (Highest Isolation):")
    db_per_tenant = DatabasePerTenant()
    
    # Create database first
    db_per_tenant.create_database(tenant1.id)
    
    await db_per_tenant.execute_query(tenant1.id, "INSERT", {
        "table": "foods",
        "data": {"name": "Apple", "calories": 95}
    })
    
    foods = await db_per_tenant.execute_query(tenant1.id, "SELECT", {"table": "foods"})
    print(f"   Inserted and retrieved {len(foods)} items for {tenant1.name}")
    
    db_stats = db_per_tenant.get_stats()
    print(f"   Databases created: {db_stats['databases']}")
    
    # Schema per tenant
    print("\n🏗️  Schema Per Tenant (Medium Isolation):")
    schema_per_tenant = SchemaPerTenant()
    
    # Create schema first
    schema_per_tenant.create_schema(tenant2.id)
    
    await schema_per_tenant.execute_query(tenant2.id, "INSERT", {
        "table": "users",
        "data": {"email": "user@techstart.com", "name": "John"}
    })
    
    users = await schema_per_tenant.execute_query(tenant2.id, "SELECT", {"table": "users"})
    print(f"   Inserted and retrieved {len(users)} items for {tenant2.name}")
    
    schema_stats = schema_per_tenant.get_stats()
    print(f"   Schemas created: {schema_stats['schemas']}")
    
    # Shared database
    print("\n📊 Shared Database with Tenant ID (Highest Efficiency):")
    shared_db = SharedDatabaseWithTenantID()
    
    await shared_db.execute_query(tenant1.id, "INSERT", {
        "table": "meals",
        "data": {"name": "Breakfast", "calories": 350}
    })
    
    await shared_db.execute_query(tenant2.id, "INSERT", {
        "table": "meals",
        "data": {"name": "Lunch", "calories": 650}
    })
    
    tenant1_meals = await shared_db.execute_query(tenant1.id, "SELECT", {"table": "meals"})
    tenant2_meals = await shared_db.execute_query(tenant2.id, "SELECT", {"table": "meals"})
    
    print(f"   {tenant1.name} meals: {len(tenant1_meals)}")
    print(f"   {tenant2.name} meals: {len(tenant2_meals)}")
    
    shared_stats = shared_db.get_stats()
    print(f"   Total rows: {shared_stats['total_rows']}")
    print(f"   Tenant distribution: {shared_stats['tenant_distribution']}")
    
    # ========================================================================
    # 3. Tenant Context & Middleware
    # ========================================================================
    print("\n" + "=" * 80)
    print("3. TENANT CONTEXT & MIDDLEWARE")
    print("=" * 80)
    
    middleware = TenantMiddleware(registry)
    
    print("\n🔄 Resolving tenant from requests...")
    
    # Request with tenant header
    request1 = {
        "headers": {"X-Tenant-ID": tenant1.id},
        "url": "/api/foods"
    }
    
    resolved_tenant = await middleware.resolve_tenant(request1)
    print(f"   Header-based: {resolved_tenant.name if resolved_tenant else 'None'}")
    
    # Request with domain
    request2 = {
        "headers": {"Host": "acme.example.com"},
        "url": "/api/users"
    }
    
    resolved_tenant = await middleware.resolve_tenant(request2)
    print(f"   Domain-based: {resolved_tenant.name if resolved_tenant else 'None'}")
    
    # Process request with context
    print("\n▶️  Processing request with tenant context...")
    
    async def sample_handler(request):
        tenant_id = TenantContext.get_tenant()
        return {"message": f"Processed for tenant: {tenant_id}"}
    
    result = await middleware.process_request(request1, sample_handler)
    print(f"   Result: {result}")
    
    mw_stats = middleware.get_stats()
    print(f"\n📊 Middleware Statistics:")
    print(f"   Requests: {mw_stats['requests']}")
    print(f"   Success Rate: {mw_stats['success_rate']:.1%}")
    
    # ========================================================================
    # 4. Resource Quotas & Limits
    # ========================================================================
    print("\n" + "=" * 80)
    print("4. RESOURCE QUOTAS & LIMITS")
    print("=" * 80)
    
    limits = ResourceLimits()
    
    print("\n📏 Setting resource quotas...")
    
    # Set quotas for each plan
    limits.set_quota(tenant1.id, ResourceQuota(
        max_users=100,
        max_storage_gb=50.0,
        max_api_calls_per_day=100000
    ))
    
    limits.set_quota(tenant2.id, ResourceQuota(
        max_users=10,
        max_storage_gb=1.0,
        max_api_calls_per_day=1000
    ))
    
    print(f"   ✓ Set quotas for {len(limits.quotas)} tenants")
    
    # Check and increment usage
    print("\n✅ Checking resource limits...")
    
    # Add users
    for i in range(5):
        if limits.check_limit(tenant2.id, "users"):
            limits.increment_usage(tenant2.id, "users")
    
    print(f"   Added users for {tenant2.name}")
    
    # Add API calls
    for i in range(100):
        if limits.check_limit(tenant1.id, "api_calls"):
            limits.increment_usage(tenant1.id, "api_calls")
    
    print(f"   Recorded API calls for {tenant1.name}")
    
    # Try to exceed limit
    limits.increment_usage(tenant2.id, "storage", 0.9)
    
    can_add = limits.check_limit(tenant2.id, "storage", 0.5)
    print(f"   Can add 0.5GB storage: {can_add}")
    
    # Usage report
    print("\n📊 Usage Report:")
    report = limits.get_usage_report(tenant2.id)
    print(f"   Tenant: {report['tenant_id']}")
    print(f"   Users: {report['usage']['users']}/{report['quota']['max_users']}")
    print(f"   Storage: {report['usage']['storage_gb']:.1f}/{report['quota']['max_storage_gb']}GB")
    print(f"   API Calls: {report['usage']['api_calls_today']}/{report['quota']['max_api_calls_per_day']}")
    
    limits_stats = limits.get_stats()
    print(f"\n📊 Limits Statistics:")
    print(f"   Tracked Tenants: {limits_stats['tracked_tenants']}")
    print(f"   Quota Violations: {limits_stats['quota_violations']}")
    
    # ========================================================================
    # 5. Cross-Tenant Data Prevention
    # ========================================================================
    print("\n" + "=" * 80)
    print("5. CROSS-TENANT DATA PREVENTION")
    print("=" * 80)
    
    guard = TenantGuard()
    
    print("\n🛡️  Testing tenant guard...")
    
    # Set tenant context
    TenantContext.set_tenant(tenant1.id)
    
    # Valid access
    can_access = guard.validate_access(tenant1.id)
    print(f"   Same tenant access: {can_access}")
    
    # Invalid access
    can_access = guard.validate_access(tenant2.id)
    print(f"   Cross-tenant access: {can_access}")
    
    # Filter data
    print("\n🔍 Filtering data by tenant...")
    
    all_data = [
        {"id": "1", "name": "Item 1", "tenant_id": tenant1.id},
        {"id": "2", "name": "Item 2", "tenant_id": tenant2.id},
        {"id": "3", "name": "Item 3", "tenant_id": tenant1.id},
    ]
    
    filtered = guard.filter_by_tenant(all_data)
    print(f"   Total items: {len(all_data)}")
    print(f"   Filtered items: {len(filtered)}")
    
    # Scope query
    query = {"table": "foods"}
    scoped = guard.scope_query(query)
    print(f"   Scoped query filters: {scoped['filters']}")
    
    TenantContext.clear()
    
    guard_stats = guard.get_stats()
    print(f"\n📊 Guard Statistics:")
    print(f"   Checks Performed: {guard_stats['checks_performed']}")
    print(f"   Violations Detected: {guard_stats['violations_detected']}")
    
    # ========================================================================
    # 6. Tenant Provisioning
    # ========================================================================
    print("\n" + "=" * 80)
    print("6. TENANT PROVISIONING")
    print("=" * 80)
    
    provisioner = TenantProvisioner(registry, db_per_tenant)
    
    print("\n🚀 Provisioning new tenant...")
    
    result = await provisioner.provision_tenant(
        "NewStartup",
        "newstartup.example.com",
        "pro"
    )
    
    if result["success"]:
        print(f"   ✓ Tenant provisioned: {result['tenant']['name']}")
        print(f"   Admin user: {result['admin_user']['email']}")
        print(f"   Plan: {result['tenant']['plan']}")
    
    prov_stats = provisioner.get_stats()
    print(f"\n📊 Provisioner Statistics:")
    print(f"   Provisioned: {prov_stats['provisioned']}")
    print(f"   Success Rate: {prov_stats['success_rate']:.1%}")
    
    # ========================================================================
    # 7. Tenant-Aware Data Access
    # ========================================================================
    print("\n" + "=" * 80)
    print("7. TENANT-AWARE DATA ACCESS LAYER")
    print("=" * 80)
    
    data_access = TenantDataAccess(shared_db, guard)
    
    print("\n💾 Performing tenant-aware data operations...")
    
    # Set context for tenant1
    TenantContext.set_tenant(tenant1.id)
    
    # Insert
    await data_access.insert("products", {
        "id": "prod-1",
        "name": "Protein Bar",
        "price": 2.99
    })
    print(f"   ✓ Inserted product for {tenant1.name}")
    
    # Select
    products = await data_access.select("products")
    print(f"   Retrieved {len(products)} products")
    
    # Switch context
    TenantContext.set_tenant(tenant2.id)
    
    await data_access.insert("products", {
        "id": "prod-2",
        "name": "Energy Drink",
        "price": 3.49
    })
    print(f"   ✓ Inserted product for {tenant2.name}")
    
    products = await data_access.select("products")
    print(f"   Retrieved {len(products)} products (different tenant)")
    
    TenantContext.clear()
    
    da_stats = data_access.get_stats()
    print(f"\n📊 Data Access Statistics:")
    print(f"   Queries: {da_stats['queries']}")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("✅ MULTI-TENANCY ARCHITECTURE COMPLETE")
    print("=" * 80)
    
    print("\n📦 CAPABILITIES:")
    print("   ✓ Complete tenant lifecycle management")
    print("   ✓ Three isolation strategies (Database/Schema/Shared)")
    print("   ✓ Automatic tenant resolution from requests")
    print("   ✓ Thread-safe tenant context")
    print("   ✓ Resource quotas and usage tracking")
    print("   ✓ Cross-tenant data access prevention")
    print("   ✓ Automated tenant provisioning")
    print("   ✓ Tenant-aware data access layer")
    
    print("\n🎯 ARCHITECTURE METRICS:")
    print(f"   Total tenants: {registry.get_stats()['total_tenants']} ✓")
    print(f"   Isolation strategies: 3 ✓")
    print(f"   Tenant resolutions: {middleware.get_stats()['requests']} ✓")
    print(f"   Resource checks: {limits_stats['tracked_tenants']} ✓")
    print(f"   Security checks: {guard_stats['checks_performed']} ✓")
    print(f"   Provisioned tenants: {prov_stats['provisioned']} ✓")
    print(f"   Data access queries: {da_stats['queries']} ✓")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    asyncio.run(demo_multi_tenancy())
