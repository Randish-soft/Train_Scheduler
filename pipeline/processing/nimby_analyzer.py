"""
NIMBY (Not In My Backyard) Analysis Module
==========================================

Specialized module for analyzing and mitigating NIMBY factors:
- Community impact assessment
- Stakeholder identification and mapping
- Public consultation strategy development
- Compensation and mitigation planning
- Communication strategy recommendations

Author: Miguel Ibrahim E
"""

import logging
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class StakeholderGroup:
    """Represents a stakeholder group affected by railway development."""
    name: str
    size: int
    influence_level: str  # High, Medium, Low
    concern_level: str    # High, Medium, Low
    primary_concerns: List[str]
    engagement_strategy: str
    compensation_needed: bool


@dataclass
class CommunityImpact:
    """Represents impact on a specific community area."""
    location: str
    impact_type: str
    severity: str
    affected_population: int
    mitigation_measures: List[str]
    estimated_cost: float


class NIMBYAnalyzer:
    """Analyzes NIMBY factors and develops mitigation strategies."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def analyze_nimby_factors(self, route_options: List[Dict], 
                            constraint_analysis: Dict) -> Dict[str, Any]:
        """
        Comprehensive NIMBY analysis for all routes.
        
        Args:
            route_options: List of route dictionaries
            constraint_analysis: Constraint analysis from optimization module
            
        Returns:
            Dictionary containing NIMBY analysis and mitigation strategies
        """
        self.logger.info("🏘️ Analyzing NIMBY factors and community impacts...")
        
        nimby_analysis = {
            'route_nimby_assessments': {},
            'stakeholder_mapping': {},
            'community_engagement_plan': {},
            'mitigation_strategies': {},
            'compensation_framework': {},
            'communication_strategy': {}
        }
        
        # Analyze each route for NIMBY factors
        for route in route_options:
            route_id = route['route_id']
            constraints = constraint_analysis.get(route_id, {}).get('constraints', {})
            
            route_nimby = self._analyze_route_nimby(route, constraints)
            nimby_analysis['route_nimby_assessments'][route_id] = route_nimby
        
        # Generate stakeholder mapping
        nimby_analysis['stakeholder_mapping'] = self._create_stakeholder_mapping(
            route_options, nimby_analysis['route_nimby_assessments']
        )
        
        # Develop community engagement plan
        nimby_analysis['community_engagement_plan'] = self._develop_engagement_plan(
            nimby_analysis['route_nimby_assessments'],
            nimby_analysis['stakeholder_mapping']
        )
        
        # Create mitigation strategies
        nimby_analysis['mitigation_strategies'] = self._develop_mitigation_strategies(
            nimby_analysis['route_nimby_assessments']
        )
        
        # Design compensation framework
        nimby_analysis['compensation_framework'] = self._design_compensation_framework(
            nimby_analysis['route_nimby_assessments']
        )
        
        # Create communication strategy
        nimby_analysis['communication_strategy'] = self._create_communication_strategy(
            nimby_analysis['stakeholder_mapping']
        )
        
        # Save NIMBY analysis
        self._save_nimby_analysis(nimby_analysis)
        
        self.logger.info("✅ NIMBY analysis completed")
        return nimby_analysis
    
    def _analyze_route_nimby(self, route: Dict, constraints: Dict) -> Dict[str, Any]:
        """Analyze NIMBY factors for a specific route."""
        nimby_factors = constraints.get('nimby_factors', {})
        
        route_nimby = {
            'nimby_risk_level': self._assess_nimby_risk_level(nimby_factors),
            'affected_communities': self._identify_affected_communities(route, nimby_factors),
            'key_concerns': self._identify_key_concerns(nimby_factors),
            'opposition_probability': self._estimate_opposition_probability(nimby_factors),
            'critical_stakeholders': self._identify_critical_stakeholders(route, nimby_factors),
            'mitigation_priority': self._assess_mitigation_priority(nimby_factors),
            'engagement_timeline': self._recommend_engagement_timeline(nimby_factors)
        }
        
        return route_nimby
    
    def _assess_nimby_risk_level(self, nimby_factors: Dict) -> str:
        """Assess overall NIMBY risk level for a route."""
        risk_factors = 0
        
        # Count high-severity impacts
        residential_impacts = nimby_factors.get('residential_impacts', [])
        high_severity_impacts = len([r for r in residential_impacts if r.get('severity') == 'high'])
        risk_factors += high_severity_impacts * 2
        
        # Count medium-severity impacts
        medium_severity_impacts = len([r for r in residential_impacts if r.get('severity') == 'medium'])
        risk_factors += medium_severity_impacts
        
        # Property value impacts
        if nimby_factors.get('property_value_impacts'):
            risk_factors += 2
        
        # Community severance
        if nimby_factors.get('community_severance'):
            risk_factors += 3
        
        # Noise concerns
        noise_concerns = nimby_factors.get('noise_concerns', [])
        risk_factors += len(noise_concerns)
        
        # Classify risk level
        if risk_factors >= 8:
            return "Very High - Expect organized opposition"
        elif risk_factors >= 5:
            return "High - Significant opposition likely"
        elif risk_factors >= 3:
            return "Medium - Moderate opposition possible"
        elif risk_factors >= 1:
            return "Low - Minor opposition expected"
        else:
            return "Very Low - Minimal opposition anticipated"
    
    def _identify_affected_communities(self, route: Dict, nimby_factors: Dict) -> List[Dict[str, Any]]:
        """Identify specific communities affected by the route."""
        communities = []
        
        residential_impacts = nimby_factors.get('residential_impacts', [])
        
        for impact in residential_impacts:
            community = {
                'name': impact.get('location', 'Unknown Area'),
                'affected_population': impact.get('affected_population', 0),
                'severity': impact.get('severity', 'unknown'),
                'distance_to_route': impact.get('distance_to_city_center_km', 5),
                'primary_concerns': self._infer_community_concerns(impact),
                'socioeconomic_profile': self._estimate_socioeconomic_profile(impact),
                'political_influence': self._estimate_political_influence(impact),
                'organization_potential': self._estimate_organization_potential(impact)
            }
            communities.append(community)
        
        return communities
    
    def _infer_community_concerns(self, impact: Dict) -> List[str]:
        """Infer primary concerns for a community based on impact data."""
        concerns = []
        
        affected_pop = impact.get('affected_population', 0)
        severity = impact.get('severity', 'unknown')
        
        if severity == 'high':
            concerns.extend([
                "Property value decline",
                "Noise and vibration",
                "Traffic disruption during construction",
                "Loss of community character",
                "Safety concerns"
            ])
        elif severity == 'medium':
            concerns.extend([
                "Noise levels",
                "Construction disruption",
                "Property access",
                "Visual impact"
            ])
        else:  # low severity
            concerns.extend([
                "Construction noise",
                "Temporary access issues"
            ])
        
        # Add population-specific concerns
        if affected_pop > 50000:
            concerns.append("Large-scale displacement risk")
        elif affected_pop > 10000:
            concerns.append("Community division")
        
        return concerns[:5]  # Return top 5 concerns
    
    def _estimate_socioeconomic_profile(self, impact: Dict) -> str:
        """Estimate socioeconomic profile of affected community."""
        affected_pop = impact.get('affected_population', 0)
        severity = impact.get('severity', 'unknown')
        
        # Simple heuristic based on population and impact severity
        if severity == 'high' and affected_pop > 100000:
            return "Mixed urban - diverse income levels, high organization potential"
        elif severity == 'high' and affected_pop > 20000:
            return "Suburban middle-class - high property ownership, strong advocacy potential"
        elif severity == 'medium':
            return "Mixed residential - moderate organization capacity"
        else:
            return "Lower density residential - limited organization resources"
    
    def _estimate_political_influence(self, impact: Dict) -> str:
        """Estimate political influence level of affected community."""
        affected_pop = impact.get('affected_population', 0)
        severity = impact.get('severity', 'unknown')
        
        if severity == 'high' and affected_pop > 50000:
            return "High - Significant voting bloc, likely political attention"
        elif severity == 'high' or affected_pop > 20000:
            return "Medium - Moderate political influence, local government attention"
        else:
            return "Low - Limited political influence"
    
    def _estimate_organization_potential(self, impact: Dict) -> str:
        """Estimate potential for community organization and opposition."""
        affected_pop = impact.get('affected_population', 0)
        severity = impact.get('severity', 'unknown')
        
        organization_score = 0
        
        if severity == 'high':
            organization_score += 3
        elif severity == 'medium':
            organization_score += 2
        else:
            organization_score += 1
        
        if affected_pop > 100000:
            organization_score += 3
        elif affected_pop > 20000:
            organization_score += 2
        elif affected_pop > 5000:
            organization_score += 1
        
        if organization_score >= 5:
            return "Very High - Expect organized campaigns, legal challenges"
        elif organization_score >= 4:
            return "High - Likely organized opposition groups"
        elif organization_score >= 3:
            return "Medium - Possible community organization"
        else:
            return "Low - Individual complaints, limited organization"
    
    def _identify_key_concerns(self, nimby_factors: Dict) -> List[str]:
        """Identify key concerns across all affected areas."""
        all_concerns = set()
        
        # Aggregate concerns from all impact types
        if nimby_factors.get('noise_concerns'):
            all_concerns.update([
                "Railway noise and vibration",
                "24/7 operational noise",
                "Horn/whistle noise in residential areas"
            ])
        
        if nimby_factors.get('property_value_impacts'):
            all_concerns.update([
                "Property value decline (5-15%)",
                "Difficulty selling properties near tracks",
                "Insurance rate increases"
            ])
        
        if nimby_factors.get('community_severance'):
            all_concerns.update([
                "Community division by railway barrier",
                "Reduced access between neighborhoods",
                "Impact on schools and emergency services"
            ])
        
        if nimby_factors.get('visual_impacts'):
            all_concerns.update([
                "Visual intrusion of railway infrastructure",
                "Overhead wires and poles",
                "Loss of scenic views"
            ])
        
        # Safety and security concerns (always present)
        all_concerns.update([
            "Pedestrian and vehicle safety at crossings",
            "Trespassing and security issues",
            "Emergency access concerns"
        ])
        
        return list(all_concerns)
    
    def _estimate_opposition_probability(self, nimby_factors: Dict) -> Dict[str, Any]:
        """Estimate probability and intensity of opposition."""
        opposition_factors = []
        probability_score = 0
        
        residential_impacts = nimby_factors.get('residential_impacts', [])
        for impact in residential_impacts:
            severity = impact.get('severity', 'unknown')
            affected_pop = impact.get('affected_population', 0)
            
            if severity == 'high':
                probability_score += 3
                opposition_factors.append(f"High-severity impact on {affected_pop:,} residents")
            elif severity == 'medium':
                probability_score += 2
                opposition_factors.append(f"Medium-severity impact on {affected_pop:,} residents")
            else:
                probability_score += 1
        
        if nimby_factors.get('property_value_impacts'):
            probability_score += 2
            opposition_factors.append("Significant property value impacts")
        
        if nimby_factors.get('community_severance'):
            probability_score += 3
            opposition_factors.append("Community severance concerns")
        
        # Determine probability level
        if probability_score >= 8:
            probability = "Very High (>80%)"
            intensity = "Intense organized opposition with legal challenges"
        elif probability_score >= 6:
            probability = "High (60-80%)"
            intensity = "Organized opposition groups, media attention"
        elif probability_score >= 4:
            probability = "Medium (40-60%)"
            intensity = "Community meetings, petitions, local opposition"
        elif probability_score >= 2:
            probability = "Low (20-40%)"
            intensity = "Individual complaints, limited organization"
        else:
            probability = "Very Low (<20%)"
            intensity = "Minimal opposition, mostly individual concerns"
        
        return {
            'probability': probability,
            'expected_intensity': intensity,
            'key_factors': opposition_factors,
            'timeline_to_mobilization': "2-6 months" if probability_score >= 6 else "6-12 months"
        }
    
    def _identify_critical_stakeholders(self, route: Dict, nimby_factors: Dict) -> List[StakeholderGroup]:
        """Identify critical stakeholders for the route."""
        stakeholders = []
        
        # Affected residents
        total_affected = sum(
            impact.get('affected_population', 0) 
            for impact in nimby_factors.get('residential_impacts', [])
        )
        
        if total_affected > 0:
            stakeholders.append(StakeholderGroup(
                name="Directly Affected Residents",
                size=total_affected,
                influence_level="High",
                concern_level="High",
                primary_concerns=["Property values", "Noise", "Quality of life"],
                engagement_strategy="Direct consultation, compensation discussions",
                compensation_needed=True
            ))
        
        # Local government
        origin_pop = route['origin']['population']
        dest_pop = route['destination']['population']
        
        stakeholders.append(StakeholderGroup(
            name="Local Government Officials",
            size=10,  # Approximate number of key officials
            influence_level="Very High",
            concern_level="Medium",
            primary_concerns=["Economic impact", "Public support", "Environmental compliance"],
            engagement_strategy="Formal presentations, regular briefings",
            compensation_needed=False
        ))
        
        # Business community
        if origin_pop > 500000 or dest_pop > 500000:
            stakeholders.append(StakeholderGroup(
                name="Local Business Association",
                size=200,  # Estimated membership
                influence_level="High",
                concern_level="Medium",
                primary_concerns=["Construction disruption", "Long-term benefits", "Access"],
                engagement_strategy="Business forums, economic impact presentations",
                compensation_needed=False
            ))
        
        # Environmental groups
        stakeholders.append(StakeholderGroup(
            name="Environmental Organizations",
            size=50,
            influence_level="Medium",
            concern_level="High",
            primary_concerns=["Habitat disruption", "Pollution", "Sustainable transport"],
            engagement_strategy="Technical briefings, mitigation discussions",
            compensation_needed=False
        ))
        
        # Property owners association
        if nimby_factors.get('property_value_impacts'):
            stakeholders.append(StakeholderGroup(
                name="Property Owners Association",
                size=500,  # Estimated
                influence_level="High",
                concern_level="Very High",
                primary_concerns=["Property values", "Compensation", "Future development"],
                engagement_strategy="Property value guarantee programs, direct negotiations",
                compensation_needed=True
            ))
        
        return stakeholders
    
    def _assess_mitigation_priority(self, nimby_factors: Dict) -> str:
        """Assess priority level for NIMBY mitigation."""
        high_severity_count = len([
            impact for impact in nimby_factors.get('residential_impacts', [])
            if impact.get('severity') == 'high'
        ])
        
        has_property_impacts = bool(nimby_factors.get('property_value_impacts'))
        has_severance = bool(nimby_factors.get('community_severance'))
        
        if high_severity_count >= 2 or has_property_impacts or has_severance:
            return "Critical - Immediate and comprehensive mitigation required"
        elif high_severity_count >= 1:
            return "High - Proactive mitigation strongly recommended"
        elif nimby_factors.get('residential_impacts'):
            return "Medium - Standard mitigation measures sufficient"
        else:
            return "Low - Minimal mitigation required"
    
    def _recommend_engagement_timeline(self, nimby_factors: Dict) -> Dict[str, List[str]]:
        """Recommend timeline for community engagement."""
        timeline = {
            'pre_planning': [],
            'planning_phase': [],
            'approval_phase': [],
            'construction_phase': [],
            'operational_phase': []
        }
        
        # Pre-planning phase (12-18 months before formal planning)
        timeline['pre_planning'] = [
            "Informal community meetings to introduce concept",
            "Stakeholder identification and initial outreach",
            "Community liaison appointment",
            "Establish communication channels"
        ]
        
        # Planning phase (during formal planning)
        timeline['planning_phase'] = [
            "Formal public consultation sessions",
            "Technical briefings for key stakeholders",
            "Property impact assessments and notifications",
            "Community feedback integration workshops",
            "Regular progress updates and newsletters"
        ]
        
        # Approval phase (during regulatory approval)
        timeline['approval_phase'] = [
            "Public hearings participation",
            "Response to formal objections",
            "Compensation negotiations",
            "Mitigation measure finalization"
        ]
        
        # Construction phase
        timeline['construction_phase'] = [
            "Construction liaison committees",
            "Regular disruption notifications",
            "Complaint resolution procedures",
            "Progress celebrations and community events"
        ]
        
        # Operational phase
        timeline['operational_phase'] = [
            "Community benefit programs",
            "Ongoing noise and impact monitoring",
            "Annual community meetings",
            "Local economic development partnerships"
        ]
        
        return timeline
    
    def _create_stakeholder_mapping(self, route_options: List[Dict], 
                                  route_nimby_assessments: Dict) -> Dict[str, Any]:
        """Create comprehensive stakeholder mapping across all routes."""
        all_stakeholders = {}
        influence_network = {}
        
        # Aggregate stakeholders from all routes
        for route_id, nimby_assessment in route_nimby_assessments.items():
            critical_stakeholders = nimby_assessment.get('critical_stakeholders', [])
            
            for stakeholder in critical_stakeholders:
                stakeholder_key = stakeholder.name
                if stakeholder_key not in all_stakeholders:
                    all_stakeholders[stakeholder_key] = {
                        'stakeholder_group': stakeholder,
                        'affected_routes': [],
                        'total_influence_score': 0
                    }
                
                all_stakeholders[stakeholder_key]['affected_routes'].append(route_id)
                
                # Calculate influence score
                influence_score = self._calculate_influence_score(stakeholder)
                all_stakeholders[stakeholder_key]['total_influence_score'] += influence_score
        
        # Create influence network analysis
        high_influence_stakeholders = [
            name for name, data in all_stakeholders.items()
            if data['total_influence_score'] >= 8
        ]
        
        return {
            'all_stakeholders': all_stakeholders,
            'high_influence_stakeholders': high_influence_stakeholders,
            'stakeholder_engagement_priority': self._prioritize_stakeholder_engagement(all_stakeholders),
            'coalition_potential': self._assess_coalition_potential(all_stakeholders)
        }
    
    def _calculate_influence_score(self, stakeholder: StakeholderGroup) -> int:
        """Calculate influence score for a stakeholder group."""
        score = 0
        
        # Size factor
        if stakeholder.size > 1000:
            score += 3
        elif stakeholder.size > 100:
            score += 2
        elif stakeholder.size > 10:
            score += 1
        
        # Influence level factor
        influence_scores = {"Very High": 4, "High": 3, "Medium": 2, "Low": 1}
        score += influence_scores.get(stakeholder.influence_level, 1)
        
        # Concern level factor
        concern_scores = {"Very High": 3, "High": 2, "Medium": 1, "Low": 0}
        score += concern_scores.get(stakeholder.concern_level, 0)
        
        return score
    
    def _prioritize_stakeholder_engagement(self, all_stakeholders: Dict) -> List[Dict[str, Any]]:
        """Prioritize stakeholders for engagement based on influence and concern."""
        prioritized = []
        
        for name, data in all_stakeholders.items():
            stakeholder = data['stakeholder_group']
            priority_score = data['total_influence_score']
            
            # Adjustment factors
            if stakeholder.compensation_needed:
                priority_score += 2
            
            if len(data['affected_routes']) > 1:
                priority_score += 1  # Affects multiple routes
            
            prioritized.append({
                'stakeholder_name': name,
                'priority_score': priority_score,
                'engagement_urgency': "Immediate" if priority_score >= 10 else "High" if priority_score >= 7 else "Medium",
                'affected_routes_count': len(data['affected_routes']),
                'recommended_approach': stakeholder.engagement_strategy
            })
        
        return sorted(prioritized, key=lambda x: x['priority_score'], reverse=True)
    
    def _assess_coalition_potential(self, all_stakeholders: Dict) -> Dict[str, Any]:
        """Assess potential for stakeholder coalitions."""
        opposition_coalition_risk = "Low"
        support_coalition_potential = "Medium"
        
        # Check for high-concern, high-influence groups
        high_risk_groups = []
        for name, data in all_stakeholders.items():
            stakeholder = data['stakeholder_group']
            if (stakeholder.concern_level in ["High", "Very High"] and 
                stakeholder.influence_level in ["High", "Very High"]):
                high_risk_groups.append(name)
        
        if len(high_risk_groups) >= 3:
            opposition_coalition_risk = "High"
        elif len(high_risk_groups) >= 2:
            opposition_coalition_risk = "Medium"
        
        return {
            'opposition_coalition_risk': opposition_coalition_risk,
            'support_coalition_potential': support_coalition_potential,
            'key_coalition_leaders': high_risk_groups[:3],
            'coalition_prevention_strategy': [
                "Early individual engagement with high-risk groups",
                "Address concerns before they can organize collectively",
                "Create competing positive narratives about project benefits"
            ]
        }
    
    def _develop_engagement_plan(self, route_nimby_assessments: Dict, 
                               stakeholder_mapping: Dict) -> Dict[str, Any]:
        """Develop comprehensive community engagement plan."""
        engagement_plan = {
            'overall_strategy': self._create_overall_engagement_strategy(stakeholder_mapping),
            'engagement_methods': self._recommend_engagement_methods(route_nimby_assessments),
            'timeline_and_milestones': self._create_engagement_timeline(),
            'resource_requirements': self._estimate_engagement_resources(stakeholder_mapping),
            'success_metrics': self._define_engagement_success_metrics(),
            'risk_mitigation': self._create_engagement_risk_mitigation()
        }
        
        return engagement_plan
    
    def _create_overall_engagement_strategy(self, stakeholder_mapping: Dict) -> List[str]:
        """Create overall engagement strategy principles."""
        return [
            "Proactive and transparent communication from project inception",
            "Two-way dialogue with genuine consideration of community input",
            "Culturally appropriate engagement methods for diverse communities",
            "Regular updates and feedback loops throughout project lifecycle",
            "Independent facilitation for contentious issues",
            "Compensation framework developed through community consultation",
            "Local economic benefit emphasis and job creation opportunities",
            "Environmental benefit communication (reduced traffic, cleaner transport)"
        ]
    
    def _recommend_engagement_methods(self, route_nimby_assessments: Dict) -> Dict[str, List[str]]:
        """Recommend specific engagement methods based on NIMBY assessment."""
        methods = {
            'public_forums': [
                "Large community meetings for general information sharing",
                "Q&A sessions with technical experts",
                "Public exhibitions with project displays and models"
            ],
            'targeted_consultation': [
                "Small group sessions with directly affected residents",
                "One-on-one meetings with property owners",
                "Focus groups with specific demographic segments"
            ],
            'digital_engagement': [
                "Project website with regular updates and feedback forms",
                "Social media channels for informal communication",
                "Virtual reality route demonstrations",
                "Online surveys and feedback platforms"
            ],
            'collaborative_planning': [
                "Community liaison committees with voting members",
                "Co-design workshops for station areas and public spaces",
                "Joint problem-solving sessions for specific concerns"
            ],
            'ongoing_communication': [
                "Regular newsletters and progress reports",
                "Community hotline for questions and complaints",
                "Local media interviews and press releases",
                "Door-to-door information campaigns in high-impact areas"
            ]
        }
        
        return methods
    
    def _create_engagement_timeline(self) -> Dict[str, Dict[str, Any]]:
        """Create detailed engagement timeline with milestones."""
        return {
            'months_1_6': {
                'phase': 'Early Engagement',
                'key_activities': [
                    "Stakeholder identification and mapping",
                    "Community liaison team establishment",
                    "Initial informal community meetings",
                    "Baseline community sentiment assessment"
                ],
                'milestones': [
                    "Stakeholder database complete",
                    "Community liaison team operational",
                    "Initial feedback collected from 500+ residents"
                ]
            },
            'months_7_18': {
                'phase': 'Formal Consultation',
                'key_activities': [
                    "Public exhibition and information sessions",
                    "Technical briefings for key stakeholders",
                    "Property impact assessments and notifications",
                    "Formal submission and feedback period"
                ],
                'milestones': [
                    "Public exhibitions reach 2000+ attendees",
                    "All directly affected properties notified",
                    "Formal feedback period completed"
                ]
            },
            'months_19_30': {
                'phase': 'Response and Refinement',
                'key_activities': [
                    "Community feedback analysis and response",
                    "Design modifications based on input",
                    "Compensation framework development",
                    "Mitigation measure finalization"
                ],
                'milestones': [
                    "Community feedback report published",
                    "Design modifications communicated",
                    "Compensation agreements reached with 80%+ of affected parties"
                ]
            }
        }
    
    def _estimate_engagement_resources(self, stakeholder_mapping: Dict) -> Dict[str, Any]:
        """Estimate resource requirements for community engagement."""
        high_priority_stakeholders = len(stakeholder_mapping.get('high_influence_stakeholders', []))
        total_stakeholders = len(stakeholder_mapping.get('all_stakeholders', {}))
        
        return {
            'personnel_requirements': {
                'community_engagement_manager': 1,
                'community_liaison_officers': max(2, high_priority_stakeholders // 3),
                'technical_communication_specialists': 2,
                'administrative_support': 2
            },
            'estimated_annual_budget': {
                'personnel_costs': 800000,  # $800k for team
                'event_and_meeting_costs': 200000,  # $200k for venues, materials
                'communication_materials': 150000,  # $150k for brochures, website, etc.
                'independent_facilitation': 100000,  # $100k for external facilitators
                'compensation_administration': 50000,  # $50k for compensation process
                'total_annual_budget': 1300000
            },
            'timeline': '30 months intensive engagement, then ongoing maintenance',
            'success_factors': [
                "Early start before opposition organizes",
                "Adequate budget for meaningful engagement",
                "Senior management commitment to process",
                "Local hire preference for engagement team"
            ]
        }
    
    def _define_engagement_success_metrics(self) -> Dict[str, List[str]]:
        """Define metrics for measuring engagement success."""
        return {
            'quantitative_metrics': [
                "80%+ of directly affected residents contacted individually",
                "70%+ positive or neutral sentiment in post-engagement surveys",
                "50%+ reduction in formal objections vs. similar projects",
                "90%+ of compensation agreements reached without litigation"
            ],
            'qualitative_metrics': [
                "Community leaders publicly support or neutral on project",
                "Local media coverage majority neutral or positive",
                "No organized opposition campaigns lasting >6 months",
                "Community benefits package agreed through consultation"
            ],
            'process_metrics': [
                "All engagement commitments met on schedule",
                "Community feedback incorporation rate >30%",
                "Stakeholder satisfaction with process >70%",
                "Complaint resolution within 14 days average"
            ]
        }
    
    def _create_engagement_risk_mitigation(self) -> List[Dict[str, Any]]:
        """Create risk mitigation strategies for engagement process."""
        return [
            {
                'risk': 'Organized opposition campaign formation',
                'probability': 'Medium',
                'impact': 'High',
                'mitigation': [
                    "Early engagement before opposition can organize",
                    "Address concerns proactively",
                    "Develop community champions and supporters",
                    "Counter-narrative emphasizing project benefits"
                ]
            },
            {
                'risk': 'Legal challenges to consultation process',
                'probability': 'Low',
                'impact': 'Very High',
                'mitigation': [
                    "Exceed minimum legal requirements for consultation",
                    "Document all engagement activities thoroughly",
                    "Use independent facilitation for contentious issues",
                    "Legal review of engagement process"
                ]
            },
            {
                'risk': 'Media campaigns against project',
                'probability': 'Medium',
                'impact': 'Medium',
                'mitigation': [
                    "Proactive media strategy and relationships",
                    "Rapid response capability for negative coverage",
                    "Third-party endorsements from credible sources",
                    "Social media monitoring and engagement"
                ]
            },
            {
                'risk': 'Political interference or opposition',
                'probability': 'Medium',
                'impact': 'High',
                'mitigation': [
                    "Early briefings for all political parties",
                    "Emphasize economic and environmental benefits",
                    "Cross-party support development",
                    "Avoid engagement during election periods"
                ]
            }
        ]
    
    def _develop_mitigation_strategies(self, route_nimby_assessments: Dict) -> Dict[str, Any]:
        """Develop comprehensive mitigation strategies."""
        # Implementation details for mitigation strategies
        # This would include detailed technical and procedural mitigation measures
        
        return {
            'physical_mitigation': [
                "Sound barriers: 3-5m high acoustic barriers along residential sections",
                "Vibration isolation: Special track construction to reduce vibration",
                "Grade separation: Eliminate at-grade crossings in urban areas",
                "Landscaping: Native vegetation screening for visual impact reduction"
            ],
            'compensation_measures': [
                "Property value guarantee programs for directly affected properties",
                "Temporary accommodation during peak construction periods",
                "Business disruption compensation for affected commercial areas",
                "Community infrastructure improvements as offset benefits"
            ],
            'operational_measures': [
                "Quiet zones with reduced horn use in residential areas",
                "Speed restrictions through sensitive areas",
                "Night-time operational limitations (11 PM - 6 AM)",
                "Regular noise and vibration monitoring with public reporting"
            ]
        }
    
    def _design_compensation_framework(self, route_nimby_assessments: Dict) -> Dict[str, Any]:
        """Design comprehensive compensation framework."""
        return {
            'compensation_principles': [
                "Fair market value compensation for acquired properties",
                "Property value protection for adjacent properties",
                "Business loss compensation during construction",
                "Community benefit sharing from project revenues"
            ],
            'compensation_categories': {
                'direct_acquisition': "Full market value plus 10% premium plus relocation costs",
                'partial_acquisition': "Proportional compensation plus impact assessment",
                'temporary_impact': "Business loss compensation plus inconvenience payments",
                'residual_impact': "Property value guarantee for 5 years post-completion"
            },
            'dispute_resolution': [
                "Independent property valuation process",
                "Community mediation services",
                "Fast-track arbitration for unresolved disputes",
                "Appeals process with community representation"
            ]
        }
    
    def _create_communication_strategy(self, stakeholder_mapping: Dict) -> Dict[str, Any]:
        """Create comprehensive communication strategy."""
        return {
            'key_messages': [
                "Sustainable transport solution for growing population",
                "Economic development catalyst for local communities",
                "Environmental benefits through reduced road traffic",
                "Job creation during construction and operation",
                "Improved connectivity and accessibility"
            ],
            'communication_channels': {
                'traditional_media': "Local newspapers, radio, television interviews",
                'digital_media': "Project website, social media, email newsletters",
                'community_channels': "Community meetings, local events, door-to-door",
                'stakeholder_briefings': "Technical reports, formal presentations"
            },
            'crisis_communication': [
                "24-hour response capability for major issues",
                "Pre-approved messaging for common concerns",
                "Senior spokesperson availability for media",
                "Community hotline for immediate concerns"
            ]
        }
    
    def _save_nimby_analysis(self, nimby_analysis: Dict) -> None:
        """Save NIMBY analysis to file."""
        output_path = self.output_dir / "nimby_analysis.json"
        with open(output_path, 'w') as f:
            json.dump(nimby_analysis, f, indent=2, default=str)
        
        self.logger.info(f"✅ NIMBY analysis saved to {output_path}")