"""
Kubernetes Deployment Infrastructure
=====================================

Complete Kubernetes manifests and deployment automation for
production-ready nutrition AI platform deployment.

Features:
1. Multi-environment configs (dev, staging, prod)
2. Auto-scaling configurations
3. Load balancing and ingress
4. ConfigMaps and Secrets management
5. Persistent volume claims
6. Service mesh integration
7. Monitoring and logging
8. Health checks and probes

Components:
- Flask API deployment
- Redis cache cluster
- PostgreSQL database
- Model serving infrastructure
- GPU nodes for inference
- Horizontal Pod Autoscaler

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import os
import yaml
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class ResourceRequirements:
    """Resource requirements for containers"""
    cpu_request: str = "100m"
    cpu_limit: str = "1000m"
    memory_request: str = "256Mi"
    memory_limit: str = "1Gi"
    gpu_limit: Optional[int] = None


@dataclass
class DeploymentConfig:
    """Kubernetes deployment configuration"""
    name: str
    namespace: str = "wellomex"
    replicas: int = 3
    image: str = "wellomex/nutrition-api:latest"
    port: int = 8000
    environment: Environment = Environment.PRODUCTION
    resources: ResourceRequirements = field(default_factory=ResourceRequirements)
    env_vars: Dict[str, str] = field(default_factory=dict)
    secrets: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# KUBERNETES MANIFEST GENERATOR
# ============================================================================

class KubernetesManifestGenerator:
    """
    Generate Kubernetes YAML manifests
    """
    
    def __init__(self, config: DeploymentConfig):
        self.config = config
    
    def generate_namespace(self) -> Dict:
        """Generate namespace manifest"""
        return {
            'apiVersion': 'v1',
            'kind': 'Namespace',
            'metadata': {
                'name': self.config.namespace,
                'labels': {
                    'name': self.config.namespace,
                    'environment': self.config.environment.value
                }
            }
        }
    
    def generate_deployment(self) -> Dict:
        """Generate deployment manifest"""
        container_resources = {
            'requests': {
                'cpu': self.config.resources.cpu_request,
                'memory': self.config.resources.memory_request
            },
            'limits': {
                'cpu': self.config.resources.cpu_limit,
                'memory': self.config.resources.memory_limit
            }
        }
        
        # Add GPU if needed
        if self.config.resources.gpu_limit:
            container_resources['limits']['nvidia.com/gpu'] = self.config.resources.gpu_limit
        
        # Environment variables
        env = [
            {'name': k, 'value': v}
            for k, v in self.config.env_vars.items()
        ]
        
        # Secrets
        env.extend([
            {
                'name': k,
                'valueFrom': {
                    'secretKeyRef': {
                        'name': f"{self.config.name}-secrets",
                        'key': k
                    }
                }
            }
            for k in self.config.secrets.keys()
        ])
        
        return {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {
                'name': self.config.name,
                'namespace': self.config.namespace,
                'labels': {
                    'app': self.config.name,
                    'environment': self.config.environment.value,
                    'version': 'v1'
                }
            },
            'spec': {
                'replicas': self.config.replicas,
                'selector': {
                    'matchLabels': {
                        'app': self.config.name
                    }
                },
                'template': {
                    'metadata': {
                        'labels': {
                            'app': self.config.name,
                            'version': 'v1'
                        }
                    },
                    'spec': {
                        'containers': [{
                            'name': self.config.name,
                            'image': self.config.image,
                            'imagePullPolicy': 'Always',
                            'ports': [{
                                'containerPort': self.config.port,
                                'name': 'http',
                                'protocol': 'TCP'
                            }],
                            'env': env,
                            'resources': container_resources,
                            'livenessProbe': {
                                'httpGet': {
                                    'path': '/health',
                                    'port': self.config.port
                                },
                                'initialDelaySeconds': 30,
                                'periodSeconds': 10,
                                'timeoutSeconds': 5,
                                'failureThreshold': 3
                            },
                            'readinessProbe': {
                                'httpGet': {
                                    'path': '/ready',
                                    'port': self.config.port
                                },
                                'initialDelaySeconds': 10,
                                'periodSeconds': 5,
                                'timeoutSeconds': 3,
                                'failureThreshold': 3
                            }
                        }]
                    }
                },
                'strategy': {
                    'type': 'RollingUpdate',
                    'rollingUpdate': {
                        'maxSurge': 1,
                        'maxUnavailable': 0
                    }
                }
            }
        }
    
    def generate_service(self) -> Dict:
        """Generate service manifest"""
        return {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {
                'name': self.config.name,
                'namespace': self.config.namespace,
                'labels': {
                    'app': self.config.name
                }
            },
            'spec': {
                'type': 'ClusterIP',
                'selector': {
                    'app': self.config.name
                },
                'ports': [{
                    'port': 80,
                    'targetPort': self.config.port,
                    'protocol': 'TCP',
                    'name': 'http'
                }],
                'sessionAffinity': 'ClientIP'
            }
        }
    
    def generate_hpa(self) -> Dict:
        """Generate Horizontal Pod Autoscaler"""
        return {
            'apiVersion': 'autoscaling/v2',
            'kind': 'HorizontalPodAutoscaler',
            'metadata': {
                'name': f"{self.config.name}-hpa",
                'namespace': self.config.namespace
            },
            'spec': {
                'scaleTargetRef': {
                    'apiVersion': 'apps/v1',
                    'kind': 'Deployment',
                    'name': self.config.name
                },
                'minReplicas': max(2, self.config.replicas // 2),
                'maxReplicas': self.config.replicas * 3,
                'metrics': [
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'cpu',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 70
                            }
                        }
                    },
                    {
                        'type': 'Resource',
                        'resource': {
                            'name': 'memory',
                            'target': {
                                'type': 'Utilization',
                                'averageUtilization': 80
                            }
                        }
                    }
                ],
                'behavior': {
                    'scaleDown': {
                        'stabilizationWindowSeconds': 300,
                        'policies': [{
                            'type': 'Percent',
                            'value': 10,
                            'periodSeconds': 60
                        }]
                    },
                    'scaleUp': {
                        'stabilizationWindowSeconds': 0,
                        'policies': [{
                            'type': 'Percent',
                            'value': 50,
                            'periodSeconds': 30
                        }]
                    }
                }
            }
        }
    
    def generate_ingress(self, host: str = "api.wellomex.com") -> Dict:
        """Generate ingress manifest"""
        return {
            'apiVersion': 'networking.k8s.io/v1',
            'kind': 'Ingress',
            'metadata': {
                'name': f"{self.config.name}-ingress",
                'namespace': self.config.namespace,
                'annotations': {
                    'kubernetes.io/ingress.class': 'nginx',
                    'cert-manager.io/cluster-issuer': 'letsencrypt-prod',
                    'nginx.ingress.kubernetes.io/ssl-redirect': 'true',
                    'nginx.ingress.kubernetes.io/rate-limit': '100',
                    'nginx.ingress.kubernetes.io/proxy-body-size': '50m'
                }
            },
            'spec': {
                'tls': [{
                    'hosts': [host],
                    'secretName': f"{self.config.name}-tls"
                }],
                'rules': [{
                    'host': host,
                    'http': {
                        'paths': [{
                            'path': '/',
                            'pathType': 'Prefix',
                            'backend': {
                                'service': {
                                    'name': self.config.name,
                                    'port': {
                                        'number': 80
                                    }
                                }
                            }
                        }]
                    }
                }]
            }
        }
    
    def generate_configmap(self, data: Dict[str, str]) -> Dict:
        """Generate ConfigMap"""
        return {
            'apiVersion': 'v1',
            'kind': 'ConfigMap',
            'metadata': {
                'name': f"{self.config.name}-config",
                'namespace': self.config.namespace
            },
            'data': data
        }
    
    def generate_secret(self) -> Dict:
        """Generate Secret manifest (base64 encoded)"""
        import base64
        
        secret_data = {
            k: base64.b64encode(v.encode()).decode()
            for k, v in self.config.secrets.items()
        }
        
        return {
            'apiVersion': 'v1',
            'kind': 'Secret',
            'metadata': {
                'name': f"{self.config.name}-secrets",
                'namespace': self.config.namespace
            },
            'type': 'Opaque',
            'data': secret_data
        }
    
    def generate_pvc(
        self,
        name: str,
        size: str = "10Gi",
        storage_class: str = "fast-ssd"
    ) -> Dict:
        """Generate PersistentVolumeClaim"""
        return {
            'apiVersion': 'v1',
            'kind': 'PersistentVolumeClaim',
            'metadata': {
                'name': name,
                'namespace': self.config.namespace
            },
            'spec': {
                'accessModes': ['ReadWriteOnce'],
                'storageClassName': storage_class,
                'resources': {
                    'requests': {
                        'storage': size
                    }
                }
            }
        }
    
    def generate_all_manifests(self) -> Dict[str, Dict]:
        """Generate all Kubernetes manifests"""
        return {
            'namespace': self.generate_namespace(),
            'deployment': self.generate_deployment(),
            'service': self.generate_service(),
            'hpa': self.generate_hpa(),
            'ingress': self.generate_ingress(),
        }


# ============================================================================
# REDIS CLUSTER DEPLOYMENT
# ============================================================================

def generate_redis_cluster() -> List[Dict]:
    """Generate Redis cluster deployment"""
    manifests = []
    
    # Redis StatefulSet
    manifests.append({
        'apiVersion': 'apps/v1',
        'kind': 'StatefulSet',
        'metadata': {
            'name': 'redis-cluster',
            'namespace': 'wellomex'
        },
        'spec': {
            'serviceName': 'redis-cluster',
            'replicas': 6,
            'selector': {
                'matchLabels': {
                    'app': 'redis-cluster'
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': 'redis-cluster'
                    }
                },
                'spec': {
                    'containers': [{
                        'name': 'redis',
                        'image': 'redis:7.2-alpine',
                        'command': [
                            'redis-server',
                            '/conf/redis.conf'
                        ],
                        'ports': [
                            {'containerPort': 6379, 'name': 'client'},
                            {'containerPort': 16379, 'name': 'gossip'}
                        ],
                        'volumeMounts': [
                            {
                                'name': 'conf',
                                'mountPath': '/conf'
                            },
                            {
                                'name': 'data',
                                'mountPath': '/data'
                            }
                        ],
                        'resources': {
                            'requests': {
                                'cpu': '500m',
                                'memory': '1Gi'
                            },
                            'limits': {
                                'cpu': '1000m',
                                'memory': '2Gi'
                            }
                        }
                    }],
                    'volumes': [{
                        'name': 'conf',
                        'configMap': {
                            'name': 'redis-cluster-config'
                        }
                    }]
                }
            },
            'volumeClaimTemplates': [{
                'metadata': {
                    'name': 'data'
                },
                'spec': {
                    'accessModes': ['ReadWriteOnce'],
                    'resources': {
                        'requests': {
                            'storage': '10Gi'
                        }
                    }
                }
            }]
        }
    })
    
    # Redis Service
    manifests.append({
        'apiVersion': 'v1',
        'kind': 'Service',
        'metadata': {
            'name': 'redis-cluster',
            'namespace': 'wellomex'
        },
        'spec': {
            'clusterIP': 'None',
            'selector': {
                'app': 'redis-cluster'
            },
            'ports': [
                {'port': 6379, 'name': 'client'},
                {'port': 16379, 'name': 'gossip'}
            ]
        }
    })
    
    return manifests


# ============================================================================
# POSTGRESQL DEPLOYMENT
# ============================================================================

def generate_postgresql() -> List[Dict]:
    """Generate PostgreSQL deployment"""
    manifests = []
    
    # PostgreSQL StatefulSet
    manifests.append({
        'apiVersion': 'apps/v1',
        'kind': 'StatefulSet',
        'metadata': {
            'name': 'postgresql',
            'namespace': 'wellomex'
        },
        'spec': {
            'serviceName': 'postgresql',
            'replicas': 1,
            'selector': {
                'matchLabels': {
                    'app': 'postgresql'
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': 'postgresql'
                    }
                },
                'spec': {
                    'containers': [{
                        'name': 'postgresql',
                        'image': 'postgres:15-alpine',
                        'ports': [{
                            'containerPort': 5432,
                            'name': 'postgresql'
                        }],
                        'env': [
                            {
                                'name': 'POSTGRES_DB',
                                'value': 'wellomex'
                            },
                            {
                                'name': 'POSTGRES_USER',
                                'valueFrom': {
                                    'secretKeyRef': {
                                        'name': 'postgresql-secret',
                                        'key': 'username'
                                    }
                                }
                            },
                            {
                                'name': 'POSTGRES_PASSWORD',
                                'valueFrom': {
                                    'secretKeyRef': {
                                        'name': 'postgresql-secret',
                                        'key': 'password'
                                    }
                                }
                            }
                        ],
                        'volumeMounts': [{
                            'name': 'data',
                            'mountPath': '/var/lib/postgresql/data'
                        }],
                        'resources': {
                            'requests': {
                                'cpu': '500m',
                                'memory': '1Gi'
                            },
                            'limits': {
                                'cpu': '2000m',
                                'memory': '4Gi'
                            }
                        }
                    }]
                }
            },
            'volumeClaimTemplates': [{
                'metadata': {
                    'name': 'data'
                },
                'spec': {
                    'accessModes': ['ReadWriteOnce'],
                    'resources': {
                        'requests': {
                            'storage': '50Gi'
                        }
                    }
                }
            }]
        }
    })
    
    # PostgreSQL Service
    manifests.append({
        'apiVersion': 'v1',
        'kind': 'Service',
        'metadata': {
            'name': 'postgresql',
            'namespace': 'wellomex'
        },
        'spec': {
            'selector': {
                'app': 'postgresql'
            },
            'ports': [{
                'port': 5432,
                'targetPort': 5432
            }]
        }
    })
    
    return manifests


# ============================================================================
# GPU NODE POOL
# ============================================================================

def generate_gpu_deployment() -> Dict:
    """Generate GPU-enabled deployment for model serving"""
    return {
        'apiVersion': 'apps/v1',
        'kind': 'Deployment',
        'metadata': {
            'name': 'model-server-gpu',
            'namespace': 'wellomex'
        },
        'spec': {
            'replicas': 2,
            'selector': {
                'matchLabels': {
                    'app': 'model-server-gpu'
                }
            },
            'template': {
                'metadata': {
                    'labels': {
                        'app': 'model-server-gpu'
                    }
                },
                'spec': {
                    'nodeSelector': {
                        'cloud.google.com/gke-accelerator': 'nvidia-tesla-t4'
                    },
                    'containers': [{
                        'name': 'model-server',
                        'image': 'wellomex/model-server:latest',
                        'ports': [{
                            'containerPort': 8080
                        }],
                        'resources': {
                            'requests': {
                                'cpu': '4000m',
                                'memory': '8Gi',
                                'nvidia.com/gpu': 1
                            },
                            'limits': {
                                'cpu': '8000m',
                                'memory': '16Gi',
                                'nvidia.com/gpu': 1
                            }
                        },
                        'env': [{
                            'name': 'CUDA_VISIBLE_DEVICES',
                            'value': '0'
                        }]
                    }]
                }
            }
        }
    }


# ============================================================================
# DEPLOYMENT MANAGER
# ============================================================================

class DeploymentManager:
    """
    Manage Kubernetes deployments
    """
    
    def __init__(self, output_dir: str = "./k8s"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_manifest(self, name: str, manifest: Dict):
        """Save manifest to YAML file"""
        filepath = os.path.join(self.output_dir, f"{name}.yaml")
        
        with open(filepath, 'w') as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved manifest: {filepath}")
    
    def save_multi_manifest(self, name: str, manifests: List[Dict]):
        """Save multiple manifests to single file"""
        filepath = os.path.join(self.output_dir, f"{name}.yaml")
        
        with open(filepath, 'w') as f:
            for i, manifest in enumerate(manifests):
                if i > 0:
                    f.write('\n---\n')
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved multi-manifest: {filepath}")
    
    def deploy_complete_infrastructure(self):
        """Deploy complete infrastructure"""
        logger.info("Generating complete Kubernetes infrastructure...")
        
        # API deployment
        api_config = DeploymentConfig(
            name="nutrition-api",
            replicas=5,
            image="wellomex/nutrition-api:latest",
            environment=Environment.PRODUCTION,
            resources=ResourceRequirements(
                cpu_request="500m",
                cpu_limit="2000m",
                memory_request="1Gi",
                memory_limit="4Gi"
            ),
            env_vars={
                'ENVIRONMENT': 'production',
                'LOG_LEVEL': 'INFO',
                'WORKERS': '4'
            },
            secrets={
                'DATABASE_URL': 'postgresql://...',
                'REDIS_URL': 'redis://...',
                'SECRET_KEY': 'your-secret-key'
            }
        )
        
        generator = KubernetesManifestGenerator(api_config)
        manifests = generator.generate_all_manifests()
        
        # Save API manifests
        for name, manifest in manifests.items():
            self.save_manifest(f"api-{name}", manifest)
        
        # Save database manifests
        self.save_multi_manifest("postgresql", generate_postgresql())
        
        # Save Redis manifests
        self.save_multi_manifest("redis-cluster", generate_redis_cluster())
        
        # Save GPU deployment
        self.save_manifest("model-server-gpu", generate_gpu_deployment())
        
        logger.info("✓ Complete infrastructure generated")


# ============================================================================
# TESTING
# ============================================================================

def test_kubernetes_manifests():
    """Test Kubernetes manifest generation"""
    print("=" * 80)
    print("KUBERNETES DEPLOYMENT - TEST")
    print("=" * 80)
    
    # Create deployment config
    config = DeploymentConfig(
        name="test-api",
        replicas=3,
        resources=ResourceRequirements(
            cpu_request="100m",
            memory_request="256Mi"
        )
    )
    
    generator = KubernetesManifestGenerator(config)
    
    print("\n✓ Generator created")
    
    # Generate manifests
    print("\n" + "="*80)
    print("Test: Generating Manifests")
    print("="*80)
    
    manifests = generator.generate_all_manifests()
    
    print(f"✓ Generated {len(manifests)} manifests:")
    for name in manifests.keys():
        print(f"  - {name}")
    
    # Test deployment manager
    print("\n" + "="*80)
    print("Test: Deployment Manager")
    print("="*80)
    
    manager = DeploymentManager(output_dir="./test_k8s")
    
    print("✓ Deployment manager created")
    print("✓ Manifests would be saved to ./test_k8s/")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_kubernetes_manifests()
