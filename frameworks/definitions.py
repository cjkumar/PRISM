"""
Framework Definitions
=====================

Section-level structure and maximum scoring definitions for NCCP and CVD frameworks.
These mirror the validated analytical frameworks from the Delphi consensus processes.
"""

from typing import Dict, List, Tuple

# ===========================================================================
#  NCCP Framework: 12 elements, 76 sub-elements
#  Based on 2008 Atun et al. framework, validated with 67 experts
# ===========================================================================

NCCP_SECTIONS: Dict[str, List[str]] = {
    "Outcomes": [
        "Health",
        "Financial Risk Protection",
        "User Satisfaction",
        "Psychosocial Support Services",
    ],
    "Objectives": [
        "Effectiveness",
        "Efficiency",
        "Equity",
        "Responsiveness",
    ],
    "Outputs": [
        "Individual Health Services",
        "Population Health Services",
        "Community-Based Palliative Care",
        "Cancer Surveillance Systems",
    ],
    "Functions": [
        "Governance and Organisation",
        "Financing",
        "Resource Management",
    ],
    "Threats": [
        "Demographic Threats",
        "Epidemiologic Threats",
        "Political Threats",
        "Legal Threats",
        "Sociocultural Threats",
        "Economic Threats",
        "Ecological Threats",
        "Technological Threats",
    ],
    "Opportunities": [
        "Demographic Opportunities",
        "Epidemiologic Opportunities",
        "Political Opportunities",
        "Legal Opportunities",
        "Sociocultural Opportunities",
        "Economic Opportunities",
        "Ecological Opportunities",
        "Technological Opportunities",
    ],
    "Strategy": [
        "Vision",
        "Mission",
        "Goals",
        "Objectives",
        "Values",
        "Integration with Noncommunicable Disease Programs",
    ],
    "Governance and Organisation": [
        "Macro-organization",
        "Governance",
        "Policy",
        "Regulation",
        "Decentralization",
        "Strategic public private partnerships",
    ],
    "Financing": [
        "Cost measurement systems",
        "Current financing and fiscal space",
        "Proposed funding to implement plan",
        "Sources of funds",
        "Pooling of funds",
        "Channeling of funds",
        "Allocation of funds",
        "Payment mechanisms for providers",
        "Payment mechanisms for capital investments",
        "Resource Mobilization Strategies",
        "Financing for Governance and Oversight",
    ],
    "Resource Management": [
        "Human resources",
        "Capacity Building for Health Professionals",
        "Infrastructure",
        "Pharmaceuticals and medical supplies",
        "Patient Navigation",
        "Information technology and data systems",
        "Supply chain management",
        "Knowledge Generation",
        "Innovation ecosystem",
        "Education Initiatives",
        "Innovation Initiatives",
    ],
    "Health Services": [
        "Public Health Services (Health promotion)",
        "Public Health Services (Health protection)",
        "Public Health Services (Disease Prevention)",
        "Personal Healthcare Services (Diagnosis)",
        "Personal Healthcare Services (Treatment)",
        "Personal Healthcare Services (Palliation and rehabilitative care)",
        "Provider Value Enhancement",
    ],
    "Implementation": [
        "Monitoring and Evaluation framework",
        "Change management",
        "Risk and Mitigation Strategies",
        "Stakeholder engagement",
    ],
}

NCCP_MAX_SCORES: Dict[str, int] = {
    "Health": 5,
    "Financial Risk Protection": 5,
    "User Satisfaction": 5,
    "Psychosocial Support Services": 5,
    "Effectiveness": 5,
    "Efficiency": 5,
    "Equity": 5,
    "Responsiveness": 5,
    "Individual Health Services": 5,
    "Population Health Services": 5,
    "Community-Based Palliative Care": 5,
    "Cancer Surveillance Systems": 5,
    "Governance and Organisation": 3,
    "Financing": 3,
    "Resource Management": 3,
    "Demographic Threats": 2,
    "Epidemiologic Threats": 2,
    "Political Threats": 2,
    "Legal Threats": 2,
    "Sociocultural Threats": 2,
    "Economic Threats": 2,
    "Ecological Threats": 2,
    "Technological Threats": 2,
    "Demographic Opportunities": 2,
    "Epidemiologic Opportunities": 2,
    "Political Opportunities": 2,
    "Legal Opportunities": 2,
    "Sociocultural Opportunities": 2,
    "Economic Opportunities": 2,
    "Ecological Opportunities": 2,
    "Technological Opportunities": 2,
    "Vision": 3,
    "Mission": 3,
    "Goals": 3,
    "Objectives": 3,
    "Values": 3,
    "Integration with Noncommunicable Disease Programs": 3,
    "Macro-organization": 3,
    "Governance": 3,
    "Policy": 3,
    "Regulation": 3,
    "Decentralization": 3,
    "Strategic public private partnerships": 3,
    "Cost measurement systems": 3,
    "Current financing and fiscal space": 3,
    "Proposed funding to implement plan": 3,
    "Sources of funds": 3,
    "Pooling of funds": 3,
    "Channeling of funds": 3,
    "Allocation of funds": 3,
    "Payment mechanisms for providers": 3,
    "Payment mechanisms for capital investments": 3,
    "Resource Mobilization Strategies": 3,
    "Financing for Governance and Oversight": 3,
    "Human resources": 3,
    "Capacity Building for Health Professionals": 3,
    "Infrastructure": 3,
    "Pharmaceuticals and medical supplies": 3,
    "Patient Navigation": 3,
    "Information technology and data systems": 3,
    "Supply chain management": 3,
    "Knowledge Generation": 3,
    "Innovation ecosystem": 3,
    "Education Initiatives": 3,
    "Innovation Initiatives": 3,
    "Public Health Services (Health promotion)": 5,
    "Public Health Services (Health protection)": 5,
    "Public Health Services (Disease Prevention)": 5,
    "Personal Healthcare Services (Diagnosis)": 5,
    "Personal Healthcare Services (Treatment)": 5,
    "Personal Healthcare Services (Palliation and rehabilitative care)": 5,
    "Provider Value Enhancement": 5,
    "Monitoring and Evaluation framework": 5,
    "Change management": 5,
    "Risk and Mitigation Strategies": 5,
    "Stakeholder engagement": 5,
}


# ===========================================================================
#  CVD Framework: 11 elements, 69 sub-elements
#  Adapted from cancer framework, validated with 42 specialists, 28 countries
# ===========================================================================

CVD_SECTIONS: Dict[str, List[str]] = {
    "Outcomes": [
        "Health",
        "Financial Risk Protection",
        "User Satisfaction",
    ],
    "Objectives": [
        "Effectiveness",
        "Efficiency",
        "Equity",
        "Responsiveness",
    ],
    "Outputs": [
        "Individual Health Services",
        "Population Health Services",
        "CVD Surveillance Systems",
    ],
    "Threats": [
        "Demographic Threats",
        "Epidemiologic Threats",
        "Political Threats",
        "Legal Threats",
        "Sociocultural Threats",
        "Economic Threats",
        "Ecological Threats",
        "Technological Threats",
    ],
    "Opportunities": [
        "Demographic Opportunities",
        "Epidemiologic Opportunities",
        "Political Opportunities",
        "Legal Opportunities",
        "Sociocultural Opportunities",
        "Economic Opportunities",
        "Ecological Opportunities",
        "Technological Opportunities",
    ],
    "Strategic Direction": [
        "Vision",
        "Mission",
        "Goals",
        "Objectives",
        "Values",
        "Integration with Noncommunicable Disease Programs",
    ],
    "Governance Arrangements": [
        "Macro-organization",
        "Governance",
        "Policy",
        "Regulation",
        "Decentralization",
        "Strategic public private partnerships",
    ],
    "Financing": [
        "Cost measurement systems",
        "Current financing and fiscal space",
        "Proposed funding to implement plan",
        "Sources of funds",
        "Pooling of funds",
        "Channeling of funds",
        "Allocation of funds",
        "Payment mechanisms for providers",
        "Payment mechanisms for capital investments",
        "Resource Mobilization Strategies",
        "Financing for Governance and Oversight",
    ],
    "Resource Management": [
        "Human resources",
        "Capacity Building for Health Professionals",
        "Infrastructure",
        "Pharmaceuticals and medical supplies",
        "Patient Navigation",
        "Information technology and data systems",
        "Supply chain management",
        "Knowledge Generation",
        "Innovation ecosystem",
    ],
    "Health Services": [
        "Public Health Services (Health promotion)",
        "Public Health Services (Health protection)",
        "Public Health Services (Disease Prevention)",
        "Personal Healthcare Services (Diagnosis)",
        "Personal Healthcare Services (Treatment)",
        "Personal Healthcare Services (Rehabilitation)",
        "Provider Value Enhancement",
    ],
    "Implementation": [
        "Monitoring and Evaluation framework",
        "Change management",
        "Risk and Mitigation Strategies",
        "Stakeholder engagement",
    ],
}

CVD_MAX_SCORES: Dict[str, int] = {
    "Health": 5,
    "Financial Risk Protection": 5,
    "User Satisfaction": 5,
    "Effectiveness": 5,
    "Efficiency": 5,
    "Equity": 5,
    "Responsiveness": 5,
    "Individual Health Services": 5,
    "Population Health Services": 5,
    "CVD Surveillance Systems": 5,
    "Demographic Threats": 2,
    "Epidemiologic Threats": 2,
    "Political Threats": 2,
    "Legal Threats": 2,
    "Sociocultural Threats": 2,
    "Economic Threats": 2,
    "Ecological Threats": 2,
    "Technological Threats": 2,
    "Demographic Opportunities": 2,
    "Epidemiologic Opportunities": 2,
    "Political Opportunities": 2,
    "Legal Opportunities": 2,
    "Sociocultural Opportunities": 2,
    "Economic Opportunities": 2,
    "Ecological Opportunities": 2,
    "Technological Opportunities": 2,
    "Vision": 3,
    "Mission": 3,
    "Goals": 3,
    "Objectives": 3,
    "Values": 3,
    "Integration with Noncommunicable Disease Programs": 3,
    "Macro-organization": 3,
    "Governance": 3,
    "Policy": 3,
    "Regulation": 3,
    "Decentralization": 3,
    "Strategic public private partnerships": 3,
    "Cost measurement systems": 3,
    "Current financing and fiscal space": 3,
    "Proposed funding to implement plan": 3,
    "Sources of funds": 3,
    "Pooling of funds": 3,
    "Channeling of funds": 3,
    "Allocation of funds": 3,
    "Payment mechanisms for providers": 3,
    "Payment mechanisms for capital investments": 3,
    "Resource Mobilization Strategies": 3,
    "Financing for Governance and Oversight": 3,
    "Human resources": 3,
    "Capacity Building for Health Professionals": 3,
    "Infrastructure": 3,
    "Pharmaceuticals and medical supplies": 3,
    "Patient Navigation": 3,
    "Information technology and data systems": 3,
    "Supply chain management": 3,
    "Knowledge Generation": 3,
    "Innovation ecosystem": 3,
    "Public Health Services (Health promotion)": 5,
    "Public Health Services (Health protection)": 5,
    "Public Health Services (Disease Prevention)": 5,
    "Personal Healthcare Services (Diagnosis)": 5,
    "Personal Healthcare Services (Treatment)": 5,
    "Personal Healthcare Services (Rehabilitation)": 5,
    "Provider Value Enhancement": 5,
    "Monitoring and Evaluation framework": 5,
    "Change management": 5,
    "Risk and Mitigation Strategies": 5,
    "Stakeholder engagement": 5,
}


def get_all_sub_elements(domain: str = "cancer") -> List[str]:
    """Return flat list of all sub-element names for a domain."""
    sections = NCCP_SECTIONS if domain == "cancer" else CVD_SECTIONS
    return [se for subs in sections.values() for se in subs]


def get_max_scores(domain: str = "cancer") -> Dict[str, int]:
    """Return max score dictionary for a domain."""
    return NCCP_MAX_SCORES if domain == "cancer" else CVD_MAX_SCORES


def get_sections(domain: str = "cancer") -> Dict[str, List[str]]:
    """Return section structure for a domain."""
    return NCCP_SECTIONS if domain == "cancer" else CVD_SECTIONS


def normalize_score(raw_score: float, sub_element: str,
                    domain: str = "cancer", target_max: float = 5.0) -> float:
    """Normalize a raw score to a 0-target_max scale.

    Formula: normalized = (raw_score / max_score) * target_max
    """
    max_scores = get_max_scores(domain)
    max_score = max_scores.get(sub_element, 5)
    if max_score == 0:
        return 0.0
    return (raw_score / max_score) * target_max
