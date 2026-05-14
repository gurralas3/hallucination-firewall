FIREWALL_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the organization's documents for information relevant to a claim",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query to find relevant document chunks"
                    },
                    "org_id": {
                        "type": "string",
                        "description": "Organization ID whose documents to search"
                    }
                },
                "required": ["query", "org_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "verify_claim",
            "description": "Record verification verdict for a claim after reviewing evidence from documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim": {
                        "type": "string",
                        "description": "The exact claim being verified"
                    },
                    "verdict": {
                        "type": "string",
                        "enum": ["SUPPORTED", "CONTRADICTED", "UNVERIFIABLE"],
                        "description": "Whether the claim is supported or contradicted by the organization's documents"
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score between 0.0 and 1.0"
                    },
                    "citation": {
                        "type": "string",
                        "description": "Exact quote from the document that supports this verdict"
                    }
                },
                "required": ["claim", "verdict", "confidence", "citation"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "flag_risk_level",
            "description": "Assign a risk level to the claim based on verification results",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk": {
                        "type": "string",
                        "enum": ["HIGH", "MEDIUM", "LOW"],
                        "description": "Risk level — HIGH if contradicted, MEDIUM if unverifiable, LOW if supported"
                    },
                    "reason": {
                        "type": "string",
                        "description": "Plain English explanation of why this risk level was assigned"
                    }
                },
                "required": ["risk", "reason"]
            }
        }
    }
]
