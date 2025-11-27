"""
Hybrid LLM Disease Database
============================

Combines LLM-generated profiles with traditional database for best of both worlds:
- LLM generates profiles on-demand or in batches
- Profiles are cached for performance
- Fallback to hardcoded data if LLM unavailable
- Automatic profile updates and versioning
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from llm_disease_profile_generator import LLMDiseaseProfileGenerator, LLMDiseaseProfile
from comprehensive_disease_db import ComprehensiveDiseaseDatabase, DiseaseProfile


class LLMHybridDiseaseDatabase:
    """
    Hybrid disease database that uses LLM for profile generation
    with intelligent caching and fallback mechanisms
    """
    
    def __init__(self, 
                 cache_dir: str = "./disease_profiles_cache",
                 use_llm: bool = True,
                 cache_ttl_days: int = 30,
                 api_key: Optional[str] = None):
        """
        Initialize hybrid database
        
        Args:
            cache_dir: Directory for caching LLM-generated profiles
            use_llm: Whether to use LLM (if False, only uses hardcoded data)
            cache_ttl_days: Cache validity period in days
            api_key: LLM API key
        """
        self.cache_dir = cache_dir
        self.use_llm = use_llm
        self.cache_ttl_days = cache_ttl_days
        
        # Initialize LLM generator
        self.llm_generator = None
        if use_llm:
            self.llm_generator = LLMDiseaseProfileGenerator(api_key=api_key)
        
        # Initialize fallback database
        self.fallback_db = ComprehensiveDiseaseDatabase()
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        print(f"✅ Hybrid Database Initialized")
        print(f"   LLM Mode: {'Enabled' if use_llm and self.llm_generator.client else 'Disabled'}")
        print(f"   Cache Directory: {cache_dir}")
        print(f"   Cache TTL: {cache_ttl_days} days")
        print(f"   Fallback Database: {len(self.fallback_db.diseases)} diseases")
    
    def get_disease(self, disease_id: str, force_regenerate: bool = False) -> Optional[DiseaseProfile]:
        """
        Get disease profile with intelligent sourcing:
        1. Check cache (if valid and not forced)
        2. Generate with LLM (if enabled)
        3. Fallback to hardcoded data
        
        Args:
            disease_id: Disease identifier
            force_regenerate: Force LLM regeneration even if cached
        
        Returns:
            DiseaseProfile or None
        """
        # Try cache first
        if not force_regenerate:
            cached_profile = self._load_from_cache(disease_id)
            if cached_profile:
                return self._convert_llm_to_disease_profile(cached_profile)
        
        # Try LLM generation
        if self.use_llm and self.llm_generator and self.llm_generator.client:
            # Get basic info from fallback DB to help LLM
            fallback_disease = self.fallback_db.get_disease(disease_id)
            
            if fallback_disease:
                llm_profile = self.llm_generator.generate_disease_profile(
                    disease_name=fallback_disease.name,
                    icd10_code=fallback_disease.icd10_codes[0] if fallback_disease.icd10_codes else None,
                    category=fallback_disease.category
                )
                
                if llm_profile:
                    # Cache it
                    self._save_to_cache(disease_id, llm_profile)
                    return self._convert_llm_to_disease_profile(llm_profile)
        
        # Fallback to hardcoded data
        return self.fallback_db.get_disease(disease_id)
    
    def get_all_diseases(self) -> Dict[str, DiseaseProfile]:
        """
        Get all available diseases
        Returns hardcoded diseases by default (for speed)
        Use batch_generate_all() to create LLM versions
        """
        return self.fallback_db.diseases
    
    def batch_generate_all(self, delay: float = 2.0, max_diseases: Optional[int] = None):
        """
        Generate LLM profiles for all diseases in fallback database
        
        Args:
            delay: Delay between API calls
            max_diseases: Limit number of diseases (for testing/cost control)
        """
        if not self.use_llm or not self.llm_generator or not self.llm_generator.client:
            print("❌ LLM not available for batch generation")
            return
        
        all_diseases = self.fallback_db.diseases
        
        # Prepare disease list
        disease_list = []
        for disease_id, profile in list(all_diseases.items())[:max_diseases]:
            disease_list.append({
                "disease_id": disease_id,
                "name": profile.name,
                "icd10": profile.icd10_codes[0] if profile.icd10_codes else None,
                "category": profile.category
            })
        
        print(f"\n🚀 Starting batch generation of {len(disease_list)} diseases...")
        print(f"💰 Estimated cost: ${len(disease_list) * 0.2:.2f} - ${len(disease_list) * 0.5:.2f}")
        print(f"⏱️ Estimated time: {len(disease_list) * delay / 60:.1f} minutes")
        
        # Generate
        self.llm_generator.batch_generate_profiles(
            diseases=disease_list,
            delay=delay,
            cache_dir=self.cache_dir
        )
    
    def _load_from_cache(self, disease_id: str) -> Optional[LLMDiseaseProfile]:
        """Load profile from cache if valid"""
        cache_file = os.path.join(self.cache_dir, f"{disease_id}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # Check if cache is still valid
            file_mtime = os.path.getmtime(cache_file)
            file_age = datetime.now() - datetime.fromtimestamp(file_mtime)
            
            if file_age > timedelta(days=self.cache_ttl_days):
                print(f"⚠️ Cache expired for {disease_id} ({file_age.days} days old)")
                return None
            
            # Load cache
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            return self.llm_generator._dict_to_profile(data)
            
        except Exception as e:
            print(f"⚠️ Failed to load cache for {disease_id}: {e}")
            return None
    
    def _save_to_cache(self, disease_id: str, profile: LLMDiseaseProfile):
        """Save profile to cache"""
        cache_file = os.path.join(self.cache_dir, f"{disease_id}.json")
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.llm_generator._profile_to_dict(profile), f, indent=2)
            print(f"💾 Cached profile: {disease_id}")
        except Exception as e:
            print(f"⚠️ Failed to cache {disease_id}: {e}")
    
    def _convert_llm_to_disease_profile(self, llm_profile: LLMDiseaseProfile) -> DiseaseProfile:
        """Convert LLM profile to standard DiseaseProfile format"""
        from comprehensive_disease_db import NutritionalGuideline, FoodRestriction
        
        # Convert nutritional guidelines
        guidelines = [
            NutritionalGuideline(
                nutrient=g.nutrient,
                target=g.target,
                unit=g.unit,
                priority=g.priority,
                reasoning=g.reasoning
            )
            for g in llm_profile.nutritional_guidelines
        ]
        
        # Convert food restrictions
        restrictions = [
            FoodRestriction(
                food_item=r.food_item,
                severity=r.severity,
                reason=r.reason,
                alternative=', '.join(r.alternatives) if r.alternatives else None
            )
            for r in llm_profile.food_restrictions
        ]
        
        return DiseaseProfile(
            disease_id=llm_profile.disease_id,
            name=llm_profile.name,
            icd10_codes=llm_profile.icd10_codes,
            category=llm_profile.category,
            nutritional_guidelines=guidelines,
            food_restrictions=restrictions,
            recommended_foods=llm_profile.recommended_foods,
            meal_timing=llm_profile.meal_timing,
            portion_control=llm_profile.portion_control
        )
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about cached profiles"""
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
        
        stats = {
            "total_cached": len(cache_files),
            "cache_dir": self.cache_dir,
            "cache_ttl_days": self.cache_ttl_days,
            "profiles": []
        }
        
        for cache_file in cache_files:
            try:
                file_path = os.path.join(self.cache_dir, cache_file)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                file_mtime = os.path.getmtime(file_path)
                age_days = (datetime.now() - datetime.fromtimestamp(file_mtime)).days
                
                stats["profiles"].append({
                    "disease_id": data["disease_id"],
                    "name": data["name"],
                    "age_days": age_days,
                    "is_valid": age_days <= self.cache_ttl_days,
                    "llm_model": data.get("llm_model", "unknown"),
                    "last_updated": data.get("last_updated", "unknown")
                })
            except:
                pass
        
        return stats
    
    def clear_cache(self, disease_id: Optional[str] = None):
        """
        Clear cache
        
        Args:
            disease_id: Clear specific disease (if None, clears all)
        """
        if disease_id:
            cache_file = os.path.join(self.cache_dir, f"{disease_id}.json")
            if os.path.exists(cache_file):
                os.remove(cache_file)
                print(f"🗑️ Cleared cache: {disease_id}")
        else:
            cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.json')]
            for f in cache_files:
                os.remove(os.path.join(self.cache_dir, f))
            print(f"🗑️ Cleared all cache ({len(cache_files)} files)")


def test_hybrid_database():
    """Test the hybrid database"""
    
    print("\n" + "="*80)
    print("TESTING HYBRID LLM DATABASE")
    print("="*80)
    
    # Test 1: Initialize
    print("\nTEST 1: Initialize Hybrid Database")
    print("-" * 80)
    
    db = LLMHybridDiseaseDatabase(use_llm=True)
    
    # Test 2: Get disease (will use fallback initially)
    print("\nTEST 2: Get Disease (Fallback Mode)")
    print("-" * 80)
    
    diabetes = db.get_disease("diabetes_type2")
    if diabetes:
        print(f"✅ Retrieved: {diabetes.name}")
        print(f"   Source: Fallback database")
        print(f"   Guidelines: {len(diabetes.nutritional_guidelines)}")
        print(f"   Restrictions: {len(diabetes.food_restrictions)}")
    
    # Test 3: Generate LLM profile
    if db.llm_generator and db.llm_generator.client:
        print("\nTEST 3: Generate LLM Profile")
        print("-" * 80)
        
        diabetes_llm = db.get_disease("diabetes_type2", force_regenerate=True)
        if diabetes_llm:
            print(f"✅ Retrieved: {diabetes_llm.name}")
            print(f"   Source: LLM-generated")
            print(f"   Guidelines: {len(diabetes_llm.nutritional_guidelines)}")
            print(f"   Restrictions: {len(diabetes_llm.food_restrictions)}")
        
        # Test 4: Get from cache
        print("\nTEST 4: Get from Cache (No API Call)")
        print("-" * 80)
        
        diabetes_cached = db.get_disease("diabetes_type2")
        if diabetes_cached:
            print(f"✅ Retrieved: {diabetes_cached.name}")
            print(f"   Source: Cache")
    else:
        print("\nℹ️ LLM not configured - skipping LLM tests")
        print("   Set OPENAI_API_KEY to enable LLM features")
    
    # Test 5: Cache stats
    print("\nTEST 5: Cache Statistics")
    print("-" * 80)
    
    stats = db.get_cache_stats()
    print(f"Total Cached Profiles: {stats['total_cached']}")
    print(f"Cache Directory: {stats['cache_dir']}")
    
    for profile in stats['profiles']:
        status = "✅ Valid" if profile['is_valid'] else "⚠️ Expired"
        print(f"  {status} {profile['name']} ({profile['age_days']} days old)")
    
    # Test 6: Batch generation (optional, costs money)
    print("\nTEST 6: Batch Generation (Optional)")
    print("-" * 80)
    print("To generate LLM profiles for all diseases:")
    print("  db.batch_generate_all(max_diseases=10)  # Generate 10 diseases")
    print("  db.batch_generate_all()  # Generate all 175 diseases")
    print("\n💰 Cost estimate:")
    print("  10 diseases: $2-5")
    print("  100 diseases: $20-50")
    print("  175 diseases: $35-90")
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)


if __name__ == "__main__":
    test_hybrid_database()
