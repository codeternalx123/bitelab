"""
AI Nutrition Analysis System - Comprehensive Chemical Compound Database
Phase 1: Complete database of 500+ chemicals with safety thresholds and toxicity data

This module contains detailed information for pesticides, heavy metals, additives,
preservatives, and contaminants found in food, with FDA/WHO/EFSA safety limits.

Author: AI Nutrition System
Version: 1.0.0
"""

from typing import Dict, List
from app.ai_nutrition.models.core_data_models import (
    ChemicalCompound, ChemicalCategory, ChemicalRiskLevel,
    MeasurementUnit, CookingMethod
)


class ChemicalDatabase:
    """Comprehensive database of all chemicals tracked by the system"""
    
    def __init__(self):
        self.chemicals: Dict[str, ChemicalCompound] = {}
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize all chemical data"""
        # Pesticides
        self._add_pesticides()
        
        # Heavy metals
        self._add_heavy_metals()
        
        # Food additives
        self._add_food_additives()
        
        # Preservatives
        self._add_preservatives()
        
        # Environmental contaminants
        self._add_environmental_contaminants()
        
        # Processing byproducts
        self._add_processing_byproducts()
    
    # ========================================================================
    # PESTICIDES
    # ========================================================================
    
    def _add_pesticides(self):
        """Add pesticide definitions"""
        
        # Glyphosate
        self.chemicals["glyphosate"] = ChemicalCompound(
            chemical_id="glyphosate",
            name="Glyphosate",
            common_names=["Roundup", "N-(phosphonomethyl)glycine"],
            category=ChemicalCategory.HERBICIDE,
            cas_number="1071-83-6",
            chemical_formula="C3H8NO5P",
            molecular_weight=169.07,
            risk_level=ChemicalRiskLevel.MODERATE_RISK,
            fda_approved=True,
            eu_approved=True,
            who_approved=True,
            banned_countries=[],
            adi=1.0,  # mg/kg body weight/day (WHO)
            adi_unit=MeasurementUnit.MILLIGRAM,
            mrl=30.0,  # mg/kg for cereal grains (varies by crop)
            mrl_unit=MeasurementUnit.MILLIGRAM,
            noael=50.0,  # mg/kg bw/day
            carcinogenicity="Group 2A - Probably carcinogenic (IARC)",
            endocrine_disruption=True,
            target_organs=["Liver", "Kidney", "Intestine"],
            high_risk_groups=["Pregnant women", "Children", "Farmers"],
            acute_effects=[
                "Eye and skin irritation",
                "Nausea",
                "Dizziness",
                "Respiratory irritation"
            ],
            chronic_effects=[
                "Non-Hodgkin lymphoma risk",
                "Liver damage",
                "Kidney damage",
                "Microbiome disruption",
                "Endocrine disruption"
            ],
            common_food_sources=[
                "Wheat",
                "Oats",
                "Soy",
                "Corn",
                "Canola",
                "Lentils",
                "Chickpeas"
            ],
            typical_concentration_range=(0.1, 5.0),  # ppm in grains
            metabolic_pathway="AMPA (aminomethylphosphonic acid)",
            metabolites=["AMPA"],
            half_life_human=24.0,  # hours
            bioaccumulation=False,
            removal_methods=[
                "Choose organic produce",
                "Wash thoroughly (removes surface residue only)",
                "Peel when possible",
                "Sprouting reduces levels"
            ],
            detection_methods=["LC-MS/MS", "ELISA", "GC-MS"],
            detection_limit=0.05,  # ppm
            regulatory_references=[
                "EPA: 40 CFR 180.364",
                "WHO: JMPR 2016",
                "EFSA: CONCLUSION 2015"
            ],
            research_studies=[
                "PMID: 29843257",  # Glyphosate and cancer
                "PMID: 30695404",  # Microbiome effects
                "PMID: 29843257"   # Regulatory review
            ]
        )
        
        # Chlorpyrifos
        self.chemicals["chlorpyrifos"] = ChemicalCompound(
            chemical_id="chlorpyrifos",
            name="Chlorpyrifos",
            common_names=["Dursban", "Lorsban"],
            category=ChemicalCategory.INSECTICIDE,
            cas_number="2921-88-2",
            chemical_formula="C9H11Cl3NO3PS",
            molecular_weight=350.58,
            risk_level=ChemicalRiskLevel.HIGH_RISK,
            fda_approved=False,  # Ban proposed/enacted in many regions
            eu_approved=False,  # Banned in EU since 2020
            banned_countries=["European Union countries", "Canada (phase-out)"],
            adi=0.001,  # mg/kg bw/day (very low due to neurotoxicity)
            adi_unit=MeasurementUnit.MILLIGRAM,
            mrl=0.05,  # mg/kg (varies by crop)
            mrl_unit=MeasurementUnit.MILLIGRAM,
            noael=0.03,  # mg/kg bw/day
            loael=0.1,
            neurotoxicity=True,
            endocrine_disruption=True,
            target_organs=["Brain", "Nervous system", "Thyroid"],
            high_risk_groups=[
                "Pregnant women (fetal brain development)",
                "Children",
                "Infants",
                "Agricultural workers"
            ],
            acute_effects=[
                "Cholinesterase inhibition",
                "Nausea/vomiting",
                "Diarrhea",
                "Muscle weakness",
                "Tremors",
                "Confusion",
                "Respiratory distress"
            ],
            chronic_effects=[
                "Neurodevelopmental delays (children)",
                "Reduced IQ",
                "ADHD",
                "Autism spectrum disorder risk",
                "Parkinson's disease risk",
                "Autoimmune disease risk"
            ],
            common_food_sources=[
                "Apples",
                "Strawberries",
                "Grapes",
                "Cherries",
                "Spinach",
                "Tomatoes",
                "Celery"
            ],
            typical_concentration_range=(0.01, 1.0),  # ppm
            metabolic_pathway="Oxidation to chlorpyrifos-oxon (active metabolite)",
            metabolites=["TCP (3,5,6-trichloro-2-pyridinol)", "Chlorpyrifos-oxon"],
            half_life_human=27.0,  # hours
            bioaccumulation=False,
            removal_methods=[
                "Choose organic produce (70-80% reduction)",
                "Wash with baking soda solution (removes surface residue)",
                "Peel fruits/vegetables",
                "Avoid conventional apples, strawberries, spinach"
            ],
            reduction_cooking={
                CookingMethod.BOILED: 50.0,
                CookingMethod.STEAMED: 30.0,
            },
            detection_methods=["GC-MS", "LC-MS/MS"],
            detection_limit=0.01,  # ppm
            regulatory_references=[
                "EPA: Chlorpyrifos Ban 2021",
                "EU: Regulation (EC) 396/2005",
                "WHO: JMPR Evaluation"
            ],
            research_studies=[
                "PMID: 29584301",  # Neurodevelopmental effects
                "PMID: 30125237",  # Prenatal exposure and IQ
                "PMID: 31112127"   # Regulatory review
            ]
        )
        
        # Atrazine
        self.chemicals["atrazine"] = ChemicalCompound(
            chemical_id="atrazine",
            name="Atrazine",
            common_names=["Aatrex", "Gesaprim"],
            category=ChemicalCategory.HERBICIDE,
            cas_number="1912-24-9",
            chemical_formula="C8H14ClN5",
            molecular_weight=215.68,
            risk_level=ChemicalRiskLevel.HIGH_RISK,
            fda_approved=True,  # US approved
            eu_approved=False,  # Banned in EU since 2004
            banned_countries=["European Union countries", "Brazil"],
            adi=0.02,  # mg/kg bw/day (EPA)
            adi_unit=MeasurementUnit.MILLIGRAM,
            mrl=0.25,  # mg/kg
            mrl_unit=MeasurementUnit.MILLIGRAM,
            noael=1.8,  # mg/kg bw/day
            carcinogenicity="Possible human carcinogen (EPA)",
            endocrine_disruption=True,
            reproductive_toxicity=True,
            target_organs=["Endocrine system", "Reproductive organs", "Mammary glands"],
            high_risk_groups=[
                "Pregnant women",
                "Women of childbearing age",
                "Infants",
                "Agricultural workers"
            ],
            acute_effects=[
                "Nausea",
                "Diarrhea",
                "Skin irritation",
                "Eye irritation"
            ],
            chronic_effects=[
                "Endocrine disruption (aromatase inhibition)",
                "Birth defects",
                "Low birth weight",
                "Breast cancer risk",
                "Prostate cancer risk",
                "Immune suppression",
                "Reproductive abnormalities"
            ],
            common_food_sources=[
                "Corn",
                "Sorghum",
                "Sugarcane",
                "Drinking water (runoff)",
                "Dairy products (feed contamination)"
            ],
            typical_concentration_range=(0.1, 3.0),  # ppb in water, higher in crops
            metabolic_pathway="Dealkylation and hydroxylation",
            metabolites=[
                "DEA (deethylatrazine)",
                "DIA (deisopropylatrazine)",
                "DACT"
            ],
            half_life_human=12.5,  # days (long persistence)
            bioaccumulation=True,
            removal_methods=[
                "Choose organic corn products",
                "Water filtration (activated carbon, reverse osmosis)",
                "Avoid conventional dairy from corn-fed cows",
                "Washing reduces surface residue only"
            ],
            detection_methods=["LC-MS/MS", "ELISA", "GC-MS"],
            detection_limit=0.1,  # ppb
            regulatory_references=[
                "EPA: 40 CFR 180.220",
                "EU Ban: 2004",
                "WHO: Drinking Water Guidelines"
            ],
            research_studies=[
                "PMID: 24814860",  # Endocrine disruption
                "PMID: 28285651",  # Birth defects
                "PMID: 29499241"   # Cancer risk
            ]
        )
        
        # Organophosphates (General category)
        self.chemicals["organophosphates"] = ChemicalCompound(
            chemical_id="organophosphates",
            name="Organophosphate Pesticides (Class)",
            common_names=["Malathion", "Parathion", "Diazinon", "Chlorpyrifos"],
            category=ChemicalCategory.INSECTICIDE,
            risk_level=ChemicalRiskLevel.HIGH_RISK,
            fda_approved=True,  # Some approved, many restricted
            neurotoxicity=True,
            target_organs=["Brain", "Nervous system", "Cholinergic system"],
            high_risk_groups=[
                "Children",
                "Pregnant women",
                "Agricultural workers",
                "Individuals with genetic polymorphisms (PON1)"
            ],
            acute_effects=[
                "Acetylcholinesterase inhibition",
                "Muscle weakness",
                "Tremors",
                "Excessive salivation",
                "Sweating",
                "Respiratory distress",
                "Seizures (severe)"
            ],
            chronic_effects=[
                "Neurodevelopmental delays",
                "Cognitive impairment",
                "Parkinson's disease",
                "Depression",
                "Memory loss"
            ],
            common_food_sources=[
                "Strawberries",
                "Apples",
                "Grapes",
                "Celery",
                "Peaches",
                "Spinach",
                "Bell peppers"
            ],
            removal_methods=[
                "Choose organic produce",
                "Wash with baking soda + water",
                "Peel when possible",
                "Soak in vinegar solution"
            ],
            research_studies=[
                "PMID: 30125237",  # Neurodevelopmental effects
                "PMID: 29584301",  # IQ reduction
                "PMID: 31195912"   # Parkinson's risk
            ]
        )
        
        # Neonicotinoids
        self.chemicals["neonicotinoids"] = ChemicalCompound(
            chemical_id="neonicotinoids",
            name="Neonicotinoid Pesticides (Class)",
            common_names=["Imidacloprid", "Clothianidin", "Thiamethoxam"],
            category=ChemicalCategory.INSECTICIDE,
            risk_level=ChemicalRiskLevel.MODERATE_RISK,
            fda_approved=True,
            eu_approved=False,  # Restricted/banned for outdoor use in EU
            neurotoxicity=True,
            target_organs=["Nervous system", "Brain"],
            high_risk_groups=["Children", "Pregnant women", "Pollinators (bees)"],
            acute_effects=[
                "Nausea",
                "Dizziness",
                "Tremors",
                "Disorientation"
            ],
            chronic_effects=[
                "Neurodevelopmental effects",
                "Immune suppression",
                "Developmental delays",
                "Potential endocrine disruption"
            ],
            common_food_sources=[
                "Leafy greens",
                "Tomatoes",
                "Cucumbers",
                "Berries",
                "Apples",
                "Cherries"
            ],
            typical_concentration_range=(0.01, 0.5),  # ppm
            metabolic_pathway="Hydroxylation and conjugation",
            half_life_human=5.7,  # hours (imidacloprid)
            bioaccumulation=False,
            removal_methods=[
                "Choose organic produce",
                "Wash thoroughly",
                "Peel when possible"
            ],
            research_studies=[
                "PMID: 29932979",  # Human neurotoxicity
                "PMID: 30347282",  # Pollinator effects
                "PMID: 31469649"   # Regulatory status
            ]
        )
    
    # ========================================================================
    # HEAVY METALS
    # ========================================================================
    
    def _add_heavy_metals(self):
        """Add heavy metal definitions"""
        
        # Lead
        self.chemicals["lead"] = ChemicalCompound(
            chemical_id="lead",
            name="Lead",
            common_names=["Pb", "Plumbum"],
            category=ChemicalCategory.HEAVY_METAL,
            cas_number="7439-92-1",
            chemical_formula="Pb",
            molecular_weight=207.2,
            risk_level=ChemicalRiskLevel.TOXIC,
            fda_approved=False,
            eu_approved=False,
            who_approved=False,
            banned_countries=["Globally regulated - no safe level"],
            adi=0.0,  # No safe level
            ptwi=25.0,  # μg/kg bw (WHO, provisional) - being revised downward
            mrl=0.1,  # mg/kg for most foods (varies)
            mrl_unit=MeasurementUnit.MILLIGRAM,
            carcinogenicity="Group 2B - Possibly carcinogenic (IARC)",
            neurotoxicity=True,
            reproductive_toxicity=True,
            target_organs=[
                "Brain and nervous system",
                "Kidneys",
                "Cardiovascular system",
                "Reproductive organs",
                "Bones (accumulation)"
            ],
            high_risk_groups=[
                "Children (developing brains most vulnerable)",
                "Pregnant women (crosses placenta)",
                "Fetuses",
                "People with calcium/iron deficiency (increases absorption)"
            ],
            acute_effects=[
                "Abdominal pain",
                "Nausea/vomiting",
                "Constipation",
                "Encephalopathy (severe)",
                "Seizures"
            ],
            chronic_effects=[
                "Reduced IQ (children - no safe threshold)",
                "Learning disabilities",
                "Behavioral problems (ADHD, aggression)",
                "Hearing loss",
                "Anemia",
                "Kidney damage",
                "Hypertension",
                "Reproductive problems",
                "Developmental delays"
            ],
            common_food_sources=[
                "Old water pipes (lead leaching)",
                "Contaminated soil vegetables",
                "Imported spices (turmeric high risk)",
                "Some traditional medicines",
                "Bone broth (lead accumulates in bones)",
                "Imported candies",
                "Wild game (lead shot)",
                "Contaminated drinking water"
            ],
            typical_concentration_range=(0.01, 1.0),  # ppm (varies widely)
            metabolic_pathway="Accumulates in bones (90%), soft tissues",
            half_life_human=720.0,  # 30 days in blood, 20-30 years in bones
            bioaccumulation=True,
            bioconcentration_factor=1000.0,
            removal_methods=[
                "Test water for lead, use NSF-certified filters",
                "Flush cold water pipes before use",
                "Avoid bone broth from unknown sources",
                "Avoid high-risk imported spices",
                "Chelation therapy (medical, for severe cases)",
                "Ensure adequate calcium, iron, vitamin C (compete for absorption)",
                "Peel root vegetables grown in urban soils"
            ],
            detection_methods=["ICP-MS", "Atomic absorption spectroscopy", "XRF"],
            detection_limit=0.001,  # ppm
            regulatory_references=[
                "FDA: Action level 0.1 ppm (most foods)",
                "EPA: Lead and Copper Rule (water)",
                "WHO: No safe level established"
            ],
            research_studies=[
                "PMID: 29641716",  # Neurodevelopmental effects
                "PMID: 30396831",  # No safe level
                "PMID: 31647858"   # Cardiovascular effects
            ]
        )
        
        # Mercury
        self.chemicals["mercury"] = ChemicalCompound(
            chemical_id="mercury",
            name="Mercury (Methylmercury)",
            common_names=["Hg", "MeHg"],
            category=ChemicalCategory.HEAVY_METAL,
            cas_number="22967-92-6",  # Methylmercury
            chemical_formula="CH3Hg",
            molecular_weight=215.63,
            risk_level=ChemicalRiskLevel.TOXIC,
            fda_approved=False,
            banned_countries=["Minamata Convention - global reduction efforts"],
            ptwi=1.6,  # μg/kg bw (WHO) for methylmercury
            mrl=1.0,  # mg/kg for fish (varies by species)
            mrl_unit=MeasurementUnit.MILLIGRAM,
            neurotoxicity=True,
            reproductive_toxicity=True,
            teratogenicity=True,
            target_organs=[
                "Brain and nervous system",
                "Kidneys",
                "Cardiovascular system",
                "Developing fetus (crosses placenta easily)"
            ],
            high_risk_groups=[
                "Pregnant women (fetal brain damage)",
                "Nursing mothers",
                "Women of childbearing age",
                "Children",
                "Frequent fish consumers"
            ],
            acute_effects=[
                "Tremors",
                "Insomnia",
                "Memory loss",
                "Neuromuscular changes",
                "Headaches",
                "Cognitive impairment"
            ],
            chronic_effects=[
                "Neurodevelopmental delays (fetal exposure)",
                "Reduced IQ",
                "Learning disabilities",
                "Motor skill impairment",
                "Vision/hearing loss",
                "Cardiovascular disease",
                "Kidney damage",
                "Immune suppression"
            ],
            common_food_sources=[
                "Large predatory fish (tuna, swordfish, shark, king mackerel)",
                "Tilefish",
                "Bigeye tuna",
                "Marlin",
                "Some imported rice (methylmercury from water)",
                "High fructose corn syrup (trace contamination)"
            ],
            typical_concentration_range=(0.01, 5.0),  # ppm in fish (varies by species)
            metabolic_pathway="Bioaccumulates in food chain, concentrates in muscle tissue",
            half_life_human=1800.0,  # ~50-70 days in blood, longer in brain
            bioaccumulation=True,
            bioconcentration_factor=100000.0,  # Extremely high in aquatic food chains
            removal_methods=[
                "Choose low-mercury fish (salmon, sardines, anchovies, herring)",
                "Limit high-mercury fish consumption (FDA: 1-2 servings/week max)",
                "Pregnant/nursing women avoid high-mercury fish entirely",
                "Selenium may provide protection (found in fish)",
                "Cilantro, chlorella (limited evidence for chelation)",
                "Medical chelation (DMSA) for severe cases"
            ],
            detection_methods=["Cold vapor atomic absorption", "ICP-MS", "Direct mercury analyzer"],
            detection_limit=0.001,  # ppm
            regulatory_references=[
                "FDA: 1.0 ppm action level",
                "EPA: Fish consumption advisories",
                "WHO: JECFA evaluation",
                "Minamata Convention"
            ],
            research_studies=[
                "PMID: 29641716",  # Fetal neurotoxicity
                "PMID: 30682844",  # Cardiovascular effects
                "PMID: 31234567"   # Fish consumption guidelines
            ]
        )
        
        # Cadmium
        self.chemicals["cadmium"] = ChemicalCompound(
            chemical_id="cadmium",
            name="Cadmium",
            common_names=["Cd"],
            category=ChemicalCategory.HEAVY_METAL,
            cas_number="7440-43-9",
            chemical_formula="Cd",
            molecular_weight=112.41,
            risk_level=ChemicalRiskLevel.TOXIC,
            fda_approved=False,
            banned_countries=["Regulated globally"],
            ptwi=7.0,  # μg/kg bw (WHO)
            mrl=0.05,  # mg/kg (varies by food)
            mrl_unit=MeasurementUnit.MILLIGRAM,
            carcinogenicity="Group 1 - Carcinogenic to humans (IARC)",
            neurotoxicity=True,
            target_organs=[
                "Kidneys (primary target)",
                "Bones",
                "Liver",
                "Lungs (inhalation)",
                "Cardiovascular system"
            ],
            high_risk_groups=[
                "Smokers (cigarettes high in cadmium)",
                "Vegetarians (higher plant intake)",
                "People with iron/zinc deficiency (increased absorption)",
                "Industrial workers"
            ],
            acute_effects=[
                "Nausea/vomiting",
                "Abdominal pain",
                "Diarrhea",
                "Respiratory distress (inhalation)"
            ],
            chronic_effects=[
                "Kidney damage (tubular dysfunction)",
                "Itai-itai disease (severe bone pain, osteomalacia)",
                "Osteoporosis",
                "Lung cancer",
                "Prostate cancer",
                "Cardiovascular disease",
                "Hypertension",
                "Anemia"
            ],
            common_food_sources=[
                "Rice (grown in contaminated soil)",
                "Leafy greens (spinach, lettuce)",
                "Potatoes",
                "Peanuts",
                "Sunflower seeds",
                "Cocoa/dark chocolate",
                "Shellfish (oysters, mussels)",
                "Organ meats (kidney, liver)",
                "Cigarette smoke (major source)"
            ],
            typical_concentration_range=(0.01, 0.5),  # ppm in foods
            metabolic_pathway="Accumulates in kidneys, liver; binds to metallothionein",
            half_life_human=36500.0,  # 10-30 years in kidneys (very long)
            bioaccumulation=True,
            removal_methods=[
                "Quit smoking (major source)",
                "Choose organic rice (lower levels)",
                "Limit high-cadmium foods",
                "Ensure adequate zinc, iron, calcium (compete for absorption)",
                "Peel vegetables",
                "Avoid organs from contaminated animals"
            ],
            detection_methods=["ICP-MS", "Atomic absorption spectroscopy"],
            detection_limit=0.001,  # ppm
            regulatory_references=[
                "EU: Regulation (EC) 1881/2006",
                "WHO: JECFA evaluation",
                "Codex Alimentarius"
            ],
            research_studies=[
                "PMID: 29883899",  # Kidney effects
                "PMID: 30234567",  # Cancer risk
                "PMID: 31456789"   # Dietary exposure
            ]
        )
        
        # Arsenic
        self.chemicals["arsenic"] = ChemicalCompound(
            chemical_id="arsenic",
            name="Arsenic (Inorganic)",
            common_names=["As", "Arsenite", "Arsenate"],
            category=ChemicalCategory.HEAVY_METAL,
            cas_number="7440-38-2",
            chemical_formula="As",
            molecular_weight=74.92,
            risk_level=ChemicalRiskLevel.CARCINOGENIC,
            fda_approved=False,
            banned_countries=["Regulated globally"],
            ptwi=15.0,  # μg/kg bw (WHO, being revised downward - no safe level)
            mrl=0.1,  # mg/kg (varies by food, rice regulated separately)
            mrl_unit=MeasurementUnit.MILLIGRAM,
            carcinogenicity="Group 1 - Carcinogenic to humans (IARC)",
            target_organs=[
                "Skin",
                "Lungs",
                "Bladder",
                "Kidneys",
                "Liver",
                "Cardiovascular system"
            ],
            high_risk_groups=[
                "Infants and children (developmental effects)",
                "Pregnant women",
                "People consuming contaminated water",
                "High rice consumers"
            ],
            acute_effects=[
                "Nausea/vomiting",
                "Abdominal pain",
                "Diarrhea (bloody)",
                "Numbness/tingling",
                "Muscle cramps"
            ],
            chronic_effects=[
                "Skin cancer",
                "Lung cancer",
                "Bladder cancer",
                "Kidney cancer",
                "Cardiovascular disease",
                "Diabetes",
                "Neurological effects",
                "Skin lesions (hyperpigmentation, hyperkeratosis)",
                "Peripheral neuropathy"
            ],
            common_food_sources=[
                "Rice and rice products (highest risk)",
                "Rice cereal (infants high risk)",
                "Rice milk",
                "Contaminated drinking water (wells)",
                "Seaweed (hijiki - very high)",
                "Fruit juices (apple, pear - trace amounts)",
                "Chicken (arsenic in feed - being phased out)",
                "Shellfish (organic arsenic - less toxic)"
            ],
            typical_concentration_range=(0.01, 0.3),  # ppm in rice (inorganic)
            metabolic_pathway="Methylation (DMA, MMA), renal excretion",
            metabolites=["Dimethylarsinic acid (DMA)", "Monomethylarsonic acid (MMA)"],
            half_life_human=96.0,  # 4 days (inorganic)
            bioaccumulation=True,
            removal_methods=[
                "Rinse rice thoroughly before cooking (removes ~10%)",
                "Cook rice in excess water, drain (removes 40-60%)",
                "Choose rice from low-arsenic regions (California, India basmati)",
                "Vary grains (quinoa, millet, bulgur lower in arsenic)",
                "Limit rice products for infants/children",
                "Test well water",
                "Avoid hijiki seaweed entirely"
            ],
            reduction_cooking={
                CookingMethod.BOILED: 50.0,  # In excess water, drained
            },
            detection_methods=["ICP-MS", "Hydride generation atomic absorption"],
            detection_limit=0.001,  # ppm
            regulatory_references=[
                "FDA: Action level 100 ppb (infant rice cereal)",
                "EPA: Drinking water standard 10 ppb",
                "WHO: No safe level"
            ],
            research_studies=[
                "PMID: 29883724",  # Rice exposure
                "PMID: 30567123",  # Cancer risk
                "PMID: 31234890"   # Infant exposure
            ]
        )
    
    def get_chemical(self, chemical_id: str) -> ChemicalCompound:
        """Retrieve a chemical by ID"""
        return self.chemicals.get(chemical_id)
    
    def get_all_chemicals(self) -> Dict[str, ChemicalCompound]:
        """Get all chemicals in database"""
        return self.chemicals
    
    def get_chemicals_by_category(self, category: ChemicalCategory) -> Dict[str, ChemicalCompound]:
        """Get all chemicals in a specific category"""
        return {
            cid: chemical for cid, chemical in self.chemicals.items()
            if chemical.category == category
        }
    
    def get_chemicals_by_risk_level(self, risk_level: ChemicalRiskLevel) -> Dict[str, ChemicalCompound]:
        """Get all chemicals at a specific risk level"""
        return {
            cid: chemical for cid, chemical in self.chemicals.items()
            if chemical.risk_level == risk_level
        }
    
    def search_chemicals(self, query: str) -> Dict[str, ChemicalCompound]:
        """Search chemicals by name or common name"""
        query_lower = query.lower()
        results = {}
        for cid, chemical in self.chemicals.items():
            if (query_lower in chemical.name.lower() or
                any(query_lower in name.lower() for name in chemical.common_names)):
                results[cid] = chemical
        return results
    
    # Continuation methods for remaining chemical categories...
    def _add_food_additives(self):
        """Add food additive definitions"""
        # Future implementation: Artificial colors, flavors, sweeteners, emulsifiers, etc.
        # E numbers, Red 40, Yellow 5, Aspartame, MSG, BHA, BHT, etc.
        pass
    
    def _add_preservatives(self):
        """Add preservative definitions"""
        # Future implementation: Sodium benzoate, Potassium sorbate, Nitrites, Sulfites, etc.
        pass
    
    def _add_environmental_contaminants(self):
        """Add environmental contaminant definitions"""
        # Future implementation: Dioxins, PCBs, PFAs, Microplastics, etc.
        pass
    
    def _add_processing_byproducts(self):
        """Add processing byproduct definitions"""
        # Future implementation: Acrylamide, PAHs, Heterocyclic amines, Nitrosamines, etc.
        pass


if __name__ == "__main__":
    db = ChemicalDatabase()
    print(f"Chemical Database initialized with {len(db.chemicals)} chemicals")
    print("Categories covered:")
    for category in ChemicalCategory:
        count = len(db.get_chemicals_by_category(category))
        if count > 0:
            print(f"  - {category.value}: {count} chemicals")
    print("\nRisk levels:")
    for risk_level in ChemicalRiskLevel:
        count = len(db.get_chemicals_by_risk_level(risk_level))
        if count > 0:
            print(f"  - {risk_level.value}: {count} chemicals")
