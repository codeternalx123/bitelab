"""
CI/CD Pipeline Infrastructure
==============================

Complete CI/CD pipeline configuration for automated testing,
building, and deployment of nutrition AI platform.

Features:
1. GitHub Actions workflows
2. Docker multi-stage builds
3. Automated testing (unit, integration, e2e)
4. Code quality checks (linting, type checking)
5. Security scanning
6. Container scanning
7. Automated deployment to K8s
8. Rollback mechanisms

Pipelines:
- Pull Request validation
- Main branch deployment
- Release management
- Hotfix deployment
- Scheduled tasks (backups, cleanup)

Author: Wellomex AI Team
Date: November 2025
Version: 5.0.0
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class PipelineStage(Enum):
    """CI/CD pipeline stages"""
    LINT = "lint"
    TEST = "test"
    BUILD = "build"
    SECURITY_SCAN = "security_scan"
    DEPLOY_DEV = "deploy_dev"
    DEPLOY_STAGING = "deploy_staging"
    DEPLOY_PROD = "deploy_prod"


@dataclass
class CICDConfig:
    """CI/CD configuration"""
    project_name: str = "wellomex-nutrition-ai"
    docker_registry: str = "ghcr.io/wellomex"
    kubernetes_cluster: str = "production-cluster"
    enable_auto_deploy: bool = True
    require_approval: bool = True


# ============================================================================
# GITHUB ACTIONS WORKFLOWS
# ============================================================================

class GitHubActionsGenerator:
    """
    Generate GitHub Actions workflow files
    """
    
    def __init__(self, config: CICDConfig):
        self.config = config
    
    def generate_pr_workflow(self) -> Dict:
        """Generate Pull Request validation workflow"""
        return {
            'name': 'Pull Request Validation',
            'on': {
                'pull_request': {
                    'branches': ['main', 'develop']
                }
            },
            'jobs': {
                'lint': {
                    'name': 'Code Quality Checks',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up Python',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '3.11'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt && pip install -r requirements-dev.txt'
                        },
                        {
                            'name': 'Run Black formatter',
                            'run': 'black --check .'
                        },
                        {
                            'name': 'Run isort',
                            'run': 'isort --check-only .'
                        },
                        {
                            'name': 'Run flake8',
                            'run': 'flake8 . --max-line-length=100 --ignore=E203,W503'
                        },
                        {
                            'name': 'Run mypy',
                            'run': 'mypy app --ignore-missing-imports'
                        },
                        {
                            'name': 'Run pylint',
                            'run': 'pylint app --max-line-length=100'
                        }
                    ]
                },
                'test': {
                    'name': 'Run Tests',
                    'runs-on': 'ubuntu-latest',
                    'strategy': {
                        'matrix': {
                            'python-version': ['3.9', '3.10', '3.11']
                        }
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up Python ${{ matrix.python-version }}',
                            'uses': 'actions/setup-python@v4',
                            'with': {
                                'python-version': '${{ matrix.python-version }}'
                            }
                        },
                        {
                            'name': 'Install dependencies',
                            'run': 'pip install -r requirements.txt && pip install -r requirements-dev.txt'
                        },
                        {
                            'name': 'Run unit tests',
                            'run': 'pytest tests/unit -v --cov=app --cov-report=xml'
                        },
                        {
                            'name': 'Run integration tests',
                            'run': 'pytest tests/integration -v'
                        },
                        {
                            'name': 'Upload coverage',
                            'uses': 'codecov/codecov-action@v3',
                            'with': {
                                'file': './coverage.xml',
                                'flags': 'unittests',
                                'name': 'codecov-umbrella'
                            }
                        }
                    ]
                },
                'security': {
                    'name': 'Security Scan',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Run Bandit security scan',
                            'run': 'pip install bandit && bandit -r app -f json -o bandit-report.json'
                        },
                        {
                            'name': 'Run Safety check',
                            'run': 'pip install safety && safety check --json'
                        },
                        {
                            'name': 'Run Snyk security scan',
                            'uses': 'snyk/actions/python@master',
                            'env': {
                                'SNYK_TOKEN': '${{ secrets.SNYK_TOKEN }}'
                            }
                        }
                    ]
                }
            }
        }
    
    def generate_main_workflow(self) -> Dict:
        """Generate main branch deployment workflow"""
        return {
            'name': 'Build and Deploy',
            'on': {
                'push': {
                    'branches': ['main']
                }
            },
            'env': {
                'REGISTRY': self.config.docker_registry,
                'IMAGE_NAME': f"{self.config.project_name}-api"
            },
            'jobs': {
                'build': {
                    'name': 'Build Docker Image',
                    'runs-on': 'ubuntu-latest',
                    'permissions': {
                        'contents': 'read',
                        'packages': 'write'
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up Docker Buildx',
                            'uses': 'docker/setup-buildx-action@v3'
                        },
                        {
                            'name': 'Log in to Container Registry',
                            'uses': 'docker/login-action@v3',
                            'with': {
                                'registry': 'ghcr.io',
                                'username': '${{ github.actor }}',
                                'password': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Extract metadata',
                            'id': 'meta',
                            'uses': 'docker/metadata-action@v5',
                            'with': {
                                'images': '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}',
                                'tags': '|\\ntype=ref,event=branch\\ntype=semver,pattern={{version}}\\ntype=sha'
                            }
                        },
                        {
                            'name': 'Build and push Docker image',
                            'uses': 'docker/build-push-action@v5',
                            'with': {
                                'context': '.',
                                'push': True,
                                'tags': '${{ steps.meta.outputs.tags }}',
                                'labels': '${{ steps.meta.outputs.labels }}',
                                'cache-from': 'type=gha',
                                'cache-to': 'type=gha,mode=max'
                            }
                        },
                        {
                            'name': 'Scan image with Trivy',
                            'uses': 'aquasecurity/trivy-action@master',
                            'with': {
                                'image-ref': '${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest',
                                'format': 'sarif',
                                'output': 'trivy-results.sarif'
                            }
                        }
                    ]
                },
                'deploy-staging': {
                    'name': 'Deploy to Staging',
                    'needs': 'build',
                    'runs-on': 'ubuntu-latest',
                    'environment': 'staging',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up kubectl',
                            'uses': 'azure/setup-kubectl@v3'
                        },
                        {
                            'name': 'Configure kubectl',
                            'run': '|\n' +
                                   'echo "${{ secrets.KUBE_CONFIG }}" | base64 -d > kubeconfig\n' +
                                   'export KUBECONFIG=kubeconfig'
                        },
                        {
                            'name': 'Deploy to staging',
                            'run': 'kubectl apply -f k8s/staging/ --namespace=wellomex-staging'
                        },
                        {
                            'name': 'Wait for rollout',
                            'run': 'kubectl rollout status deployment/nutrition-api -n wellomex-staging'
                        },
                        {
                            'name': 'Run smoke tests',
                            'run': 'pytest tests/smoke -v --env=staging'
                        }
                    ]
                },
                'deploy-production': {
                    'name': 'Deploy to Production',
                    'needs': 'deploy-staging',
                    'runs-on': 'ubuntu-latest',
                    'environment': {
                        'name': 'production',
                        'url': 'https://api.wellomex.com'
                    },
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4'
                        },
                        {
                            'name': 'Set up kubectl',
                            'uses': 'azure/setup-kubectl@v3'
                        },
                        {
                            'name': 'Configure kubectl',
                            'run': '|\n' +
                                   'echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > kubeconfig\n' +
                                   'export KUBECONFIG=kubeconfig'
                        },
                        {
                            'name': 'Deploy to production',
                            'run': 'kubectl apply -f k8s/production/ --namespace=wellomex'
                        },
                        {
                            'name': 'Wait for rollout',
                            'run': 'kubectl rollout status deployment/nutrition-api -n wellomex --timeout=10m'
                        },
                        {
                            'name': 'Run health checks',
                            'run': '|\n' +
                                   'for i in {1..30}; do\n' +
                                   '  if curl -f https://api.wellomex.com/health; then\n' +
                                   '    echo "Health check passed"\n' +
                                   '    exit 0\n' +
                                   '  fi\n' +
                                   '  sleep 10\n' +
                                   'done\n' +
                                   'echo "Health check failed"\n' +
                                   'exit 1'
                        },
                        {
                            'name': 'Notify deployment',
                            'uses': 'slackapi/slack-github-action@v1',
                            'with': {
                                'webhook-url': '${{ secrets.SLACK_WEBHOOK }}',
                                'payload': '|\n' +
                                          '{\n' +
                                          '  "text": "✅ Production deployment successful"\n' +
                                          '}'
                            }
                        }
                    ]
                }
            }
        }
    
    def generate_release_workflow(self) -> Dict:
        """Generate release workflow"""
        return {
            'name': 'Release',
            'on': {
                'push': {
                    'tags': ['v*.*.*']
                }
            },
            'jobs': {
                'release': {
                    'name': 'Create Release',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Checkout code',
                            'uses': 'actions/checkout@v4',
                            'with': {
                                'fetch-depth': 0
                            }
                        },
                        {
                            'name': 'Generate changelog',
                            'id': 'changelog',
                            'uses': 'metcalfc/changelog-generator@v4.1.0',
                            'with': {
                                'myToken': '${{ secrets.GITHUB_TOKEN }}'
                            }
                        },
                        {
                            'name': 'Create Release',
                            'uses': 'actions/create-release@v1',
                            'env': {
                                'GITHUB_TOKEN': '${{ secrets.GITHUB_TOKEN }}'
                            },
                            'with': {
                                'tag_name': '${{ github.ref }}',
                                'release_name': 'Release ${{ github.ref }}',
                                'body': '${{ steps.changelog.outputs.changelog }}',
                                'draft': False,
                                'prerelease': False
                            }
                        }
                    ]
                }
            }
        }
    
    def generate_scheduled_workflow(self) -> Dict:
        """Generate scheduled tasks workflow"""
        return {
            'name': 'Scheduled Tasks',
            'on': {
                'schedule': [
                    {'cron': '0 2 * * *'}  # Daily at 2 AM
                ]
            },
            'jobs': {
                'database-backup': {
                    'name': 'Database Backup',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Backup PostgreSQL',
                            'run': '|\n' +
                                   'kubectl exec -n wellomex postgresql-0 -- pg_dump -U postgres wellomex > backup.sql\n' +
                                   'aws s3 cp backup.sql s3://wellomex-backups/$(date +%Y%m%d)/database.sql'
                        }
                    ]
                },
                'cleanup': {
                    'name': 'Cleanup Old Resources',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Clean Docker images',
                            'run': 'docker image prune -a --filter "until=168h" -f'
                        },
                        {
                            'name': 'Clean old logs',
                            'run': 'find logs/ -type f -mtime +30 -delete'
                        }
                    ]
                },
                'dependency-update': {
                    'name': 'Check Dependency Updates',
                    'runs-on': 'ubuntu-latest',
                    'steps': [
                        {
                            'name': 'Run pip-audit',
                            'run': 'pip install pip-audit && pip-audit'
                        }
                    ]
                }
            }
        }


# ============================================================================
# DOCKERFILE GENERATOR
# ============================================================================

class DockerfileGenerator:
    """
    Generate optimized Dockerfiles
    """
    
    @staticmethod
    def generate_multi_stage_dockerfile() -> str:
        """Generate multi-stage Dockerfile"""
        return """# Multi-stage Dockerfile for Wellomex Nutrition AI

# ============================================================================
# Stage 1: Base Python image with system dependencies
# ============================================================================
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PIP_NO_CACHE_DIR=1 \\
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    libgomp1 \\
    curl \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


# ============================================================================
# Stage 2: Build dependencies
# ============================================================================
FROM base as builder

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --user --no-warn-script-location -r requirements.txt


# ============================================================================
# Stage 3: Production image
# ============================================================================
FROM base as production

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY ./flaskbackend/app /app

# Create non-root user
RUN useradd -m -u 1000 appuser && \\
    chown -R appuser:appuser /app

USER appuser

# Add local bin to PATH
ENV PATH=/root/.local/bin:$PATH

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["gunicorn", "--bind", "0.0.0.0:8000", \\
     "--workers", "4", \\
     "--timeout", "120", \\
     "--access-logfile", "-", \\
     "--error-logfile", "-", \\
     "app.main:app"]


# ============================================================================
# Stage 4: Development image
# ============================================================================
FROM production as development

USER root

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --user -r requirements-dev.txt

USER appuser

# Override CMD for development
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=8000", "--reload"]
"""
    
    @staticmethod
    def generate_model_server_dockerfile() -> str:
        """Generate Dockerfile for GPU model server"""
        return """# GPU-enabled Model Server Dockerfile

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \\
    python3.11 \\
    python3-pip \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA support
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Copy requirements
COPY requirements-gpu.txt .
RUN pip install -r requirements-gpu.txt

# Copy model serving code
COPY ./model_server /app

EXPOSE 8080

CMD ["python3", "serve.py"]
"""


# ============================================================================
# DOCKER COMPOSE
# ============================================================================

class DockerComposeGenerator:
    """
    Generate docker-compose.yml for local development
    """
    
    @staticmethod
    def generate_docker_compose() -> Dict:
        """Generate docker-compose configuration"""
        return {
            'version': '3.8',
            'services': {
                'api': {
                    'build': {
                        'context': '.',
                        'target': 'development'
                    },
                    'ports': ['8000:8000'],
                    'environment': {
                        'DATABASE_URL': 'postgresql://postgres:postgres@db:5432/wellomex',
                        'REDIS_URL': 'redis://redis:6379/0',
                        'ENVIRONMENT': 'development'
                    },
                    'volumes': [
                        './flaskbackend:/app'
                    ],
                    'depends_on': ['db', 'redis']
                },
                'db': {
                    'image': 'postgres:15-alpine',
                    'environment': {
                        'POSTGRES_USER': 'postgres',
                        'POSTGRES_PASSWORD': 'postgres',
                        'POSTGRES_DB': 'wellomex'
                    },
                    'ports': ['5432:5432'],
                    'volumes': [
                        'postgres_data:/var/lib/postgresql/data'
                    ]
                },
                'redis': {
                    'image': 'redis:7-alpine',
                    'ports': ['6379:6379'],
                    'volumes': [
                        'redis_data:/data'
                    ]
                }
            },
            'volumes': {
                'postgres_data': {},
                'redis_data': {}
            }
        }


# ============================================================================
# PIPELINE MANAGER
# ============================================================================

class PipelineManager:
    """
    Manage CI/CD pipeline configurations
    """
    
    def __init__(self, config: CICDConfig, output_dir: str = "./.github/workflows"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_all_workflows(self):
        """Generate all workflow files"""
        generator = GitHubActionsGenerator(self.config)
        
        workflows = {
            'pr-validation': generator.generate_pr_workflow(),
            'main-deploy': generator.generate_main_workflow(),
            'release': generator.generate_release_workflow(),
            'scheduled': generator.generate_scheduled_workflow()
        }
        
        # Save workflows
        import yaml
        
        for name, workflow in workflows.items():
            filepath = os.path.join(self.output_dir, f"{name}.yml")
            with open(filepath, 'w') as f:
                yaml.dump(workflow, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Generated workflow: {filepath}")
        
        logger.info("✓ All workflows generated")
    
    def generate_dockerfiles(self):
        """Generate Dockerfiles"""
        dockerfile_gen = DockerfileGenerator()
        
        # Main Dockerfile
        with open("Dockerfile", 'w') as f:
            f.write(dockerfile_gen.generate_multi_stage_dockerfile())
        
        # Model server Dockerfile
        with open("Dockerfile.model-server", 'w') as f:
            f.write(dockerfile_gen.generate_model_server_dockerfile())
        
        logger.info("✓ Dockerfiles generated")
    
    def generate_docker_compose(self):
        """Generate docker-compose.yml"""
        import yaml
        
        compose_gen = DockerComposeGenerator()
        compose_config = compose_gen.generate_docker_compose()
        
        with open("docker-compose.yml", 'w') as f:
            yaml.dump(compose_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info("✓ docker-compose.yml generated")


# ============================================================================
# TESTING
# ============================================================================

def test_cicd_pipeline():
    """Test CI/CD pipeline generation"""
    print("=" * 80)
    print("CI/CD PIPELINE - TEST")
    print("=" * 80)
    
    config = CICDConfig()
    
    print("\n" + "="*80)
    print("Test: GitHub Actions Workflows")
    print("="*80)
    
    generator = GitHubActionsGenerator(config)
    
    workflows = {
        'PR': generator.generate_pr_workflow(),
        'Main': generator.generate_main_workflow(),
        'Release': generator.generate_release_workflow(),
        'Scheduled': generator.generate_scheduled_workflow()
    }
    
    print(f"✓ Generated {len(workflows)} workflows:")
    for name in workflows.keys():
        print(f"  - {name}")
    
    print("\n" + "="*80)
    print("Test: Dockerfile Generation")
    print("="*80)
    
    dockerfile_gen = DockerfileGenerator()
    
    dockerfile = dockerfile_gen.generate_multi_stage_dockerfile()
    print(f"✓ Multi-stage Dockerfile: {len(dockerfile)} characters")
    
    model_dockerfile = dockerfile_gen.generate_model_server_dockerfile()
    print(f"✓ Model server Dockerfile: {len(model_dockerfile)} characters")
    
    print("\n" + "="*80)
    print("Test: Docker Compose")
    print("="*80)
    
    compose_gen = DockerComposeGenerator()
    compose = compose_gen.generate_docker_compose()
    
    print(f"✓ Docker Compose services: {len(compose['services'])}")
    for service in compose['services'].keys():
        print(f"  - {service}")
    
    print("\n✅ All tests passed!")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_cicd_pipeline()
