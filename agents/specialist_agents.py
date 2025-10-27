"""
Specialized collaborative agents for multi-agent workflows.

Provides ready-to-use agents with collaboration capabilities:
- ResearchAgent: Gathers and synthesizes information
- AnalysisAgent: Analyzes data and identifies patterns
- SynthesisAgent: Combines information into coherent output
- CriticAgent: Reviews and provides feedback on work
"""

from typing import Dict, Any, List, Optional
import logging

from .base_agent import BaseCollaborativeAgent
from .core import AgentRole
from .state import StateManager
from .communication import MessageBus, MessageType


class ResearchAgent(BaseCollaborativeAgent):
    """
    Specialized agent for research and information gathering.

    Capabilities:
    - Information gathering from multiple sources
    - Fact verification and source tracking
    - Collaborative research with other agents
    - Knowledge synthesis

    Example:
        >>> role = AgentRole(
        ...     name="researcher",
        ...     description="Conducts research and gathers information",
        ...     allowed_tools=["web_search", "database_query"]
        ... )
        >>> agent = ResearchAgent(role, state_manager, message_bus)
        >>> await agent.initialize()
        >>> result = await agent.execute({"topic": "AI safety"})
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conduct research on given topic.

        Args:
            input_data: Must contain 'topic' key

        Returns:
            Research findings with sources and confidence scores
        """
        topic = input_data.get("topic", "")
        if not topic:
            topic = await self.read_state("research_topic", "")

        if not topic:
            self.logger.warning("No research topic provided")
            return {"status": "error", "message": "No topic provided"}

        self.logger.info(f"Researching: {topic[:100]}...")

        # Phase 1: Initial research
        findings = await self._gather_information(topic)

        # Phase 2: Verify with other agents if available
        verified_findings = await self._verify_with_peers(findings)

        # Phase 3: Synthesize results
        research_report = {
            "topic": topic,
            "findings": verified_findings,
            "sources": findings.get("sources", []),
            "confidence": self._calculate_confidence(verified_findings),
            "summary": self._create_summary(verified_findings)
        }

        # Store in state
        await self.write_state("research_findings", research_report)

        self.logger.info(f"Research complete: {len(verified_findings)} findings")

        return {
            "status": "success",
            "findings_count": len(verified_findings),
            "confidence": research_report["confidence"]
        }

    async def _gather_information(self, topic: str) -> Dict[str, Any]:
        """
        Gather information on topic.

        In production, would use tools like web_search, database_query, etc.
        """
        # Simplified research - would use actual tools in production
        findings = {
            "facts": [
                f"Key finding about {topic} (fact 1)",
                f"Important insight regarding {topic} (fact 2)",
                f"Critical information on {topic} (fact 3)"
            ],
            "sources": [
                {"url": "source1.com", "credibility": 0.9},
                {"url": "source2.com", "credibility": 0.85},
                {"url": "source3.com", "credibility": 0.8}
            ]
        }

        return findings

    async def _verify_with_peers(self, findings: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Verify findings with peer research agents.

        Args:
            findings: Initial findings to verify

        Returns:
            Verified findings with confidence scores
        """
        # Get active agents
        active_agents = self.message_bus.get_active_agents()
        peer_researchers = [
            agent_id for agent_id in active_agents
            if agent_id != self.agent_id and "research" in agent_id.lower()
        ]

        if not peer_researchers:
            # No peers to verify with, return original findings
            return [
                {"fact": fact, "confidence": 0.7, "verified": False}
                for fact in findings.get("facts", [])
            ]

        verified = []
        for fact in findings.get("facts", []):
            # Ask peer to verify
            try:
                response = await self.send_query(
                    recipient=peer_researchers[0],
                    query={"action": "verify", "fact": fact},
                    timeout=10.0
                )

                verified.append({
                    "fact": fact,
                    "confidence": response.content.get("confidence", 0.7),
                    "verified": True
                })
            except Exception as e:
                self.logger.warning(f"Failed to verify with peer: {e}")
                verified.append({
                    "fact": fact,
                    "confidence": 0.7,
                    "verified": False
                })

        return verified

    def _calculate_confidence(self, findings: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence score."""
        if not findings:
            return 0.0

        total_confidence = sum(f.get("confidence", 0.5) for f in findings)
        return total_confidence / len(findings)

    def _create_summary(self, findings: List[Dict[str, Any]]) -> str:
        """Create summary of research findings."""
        if not findings:
            return "No findings to summarize."

        summary_parts = [f"- {f['fact']}" for f in findings[:5]]
        return "\n".join(summary_parts)


class AnalysisAgent(BaseCollaborativeAgent):
    """
    Specialized agent for data analysis and pattern identification.

    Capabilities:
    - Statistical analysis
    - Pattern recognition
    - Trend identification
    - Collaborative analysis with other agents

    Example:
        >>> role = AgentRole(
        ...     name="analyzer",
        ...     description="Analyzes data and identifies patterns",
        ...     allowed_tools=["data_processor", "statistical_tools"]
        ... )
        >>> agent = AnalysisAgent(role, state_manager, message_bus)
        >>> await agent.initialize()
        >>> result = await agent.execute({"data": [1, 2, 3, 4, 5]})
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze provided data.

        Args:
            input_data: Must contain 'data' key or reference to data in state

        Returns:
            Analysis results with patterns and insights
        """
        # Get data to analyze
        data = input_data.get("data")
        if not data:
            data = await self.read_state("data_to_analyze")

        if not data:
            self.logger.warning("No data provided for analysis")
            return {"status": "error", "message": "No data provided"}

        self.logger.info(f"Analyzing data...")

        # Phase 1: Perform analysis
        analysis_results = await self._perform_analysis(data)

        # Phase 2: Get second opinion from peers
        peer_insights = await self._get_peer_insights(data, analysis_results)

        # Phase 3: Synthesize final analysis
        final_analysis = {
            "primary_analysis": analysis_results,
            "peer_insights": peer_insights,
            "patterns": analysis_results.get("patterns", []),
            "confidence": self._calculate_analysis_confidence(analysis_results, peer_insights),
            "recommendations": self._generate_recommendations(analysis_results)
        }

        # Store in state
        await self.write_state("analysis_results", final_analysis)

        self.logger.info(f"Analysis complete: {len(final_analysis['patterns'])} patterns found")

        return {
            "status": "success",
            "patterns_found": len(final_analysis["patterns"]),
            "confidence": final_analysis["confidence"]
        }

    async def _perform_analysis(self, data: Any) -> Dict[str, Any]:
        """
        Perform core analysis on data.

        In production, would use statistical tools and ML models.
        """
        # Simplified analysis
        patterns = []

        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            # Numeric data analysis
            avg = sum(data) / len(data) if data else 0
            patterns.append({
                "type": "average",
                "value": avg,
                "description": f"Average value is {avg:.2f}"
            })

            if len(data) > 1:
                trend = "increasing" if data[-1] > data[0] else "decreasing"
                patterns.append({
                    "type": "trend",
                    "value": trend,
                    "description": f"Overall trend is {trend}"
                })

        return {
            "patterns": patterns,
            "data_type": type(data).__name__,
            "data_size": len(data) if hasattr(data, "__len__") else 1
        }

    async def _get_peer_insights(self, data: Any,
                                primary_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get insights from peer analysis agents.

        Args:
            data: Original data
            primary_analysis: Results from primary analysis

        Returns:
            List of peer insights
        """
        active_agents = self.message_bus.get_active_agents()
        peer_analyzers = [
            agent_id for agent_id in active_agents
            if agent_id != self.agent_id and "analys" in agent_id.lower()
        ]

        insights = []
        for peer_id in peer_analyzers[:2]:  # Limit to 2 peers
            try:
                response = await self.send_query(
                    recipient=peer_id,
                    query={
                        "action": "review_analysis",
                        "analysis": primary_analysis
                    },
                    timeout=10.0
                )

                insights.append({
                    "source": peer_id,
                    "insight": response.content.get("insight", "No additional insights"),
                    "agreement": response.content.get("agreement", 0.8)
                })
            except Exception as e:
                self.logger.warning(f"Failed to get insight from {peer_id}: {e}")

        return insights

    def _calculate_analysis_confidence(self, primary: Dict[str, Any],
                                      peer_insights: List[Dict[str, Any]]) -> float:
        """Calculate confidence in analysis results."""
        if not peer_insights:
            return 0.75  # Base confidence

        # Increase confidence based on peer agreement
        avg_agreement = sum(i.get("agreement", 0.5) for i in peer_insights) / len(peer_insights)
        return min(0.95, 0.75 + (avg_agreement * 0.2))

    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        patterns = analysis.get("patterns", [])
        if not patterns:
            recommendations.append("Gather more data for meaningful analysis")
        else:
            for pattern in patterns:
                if pattern["type"] == "trend":
                    recommendations.append(f"Monitor {pattern['value']} trend closely")

        return recommendations


class SynthesisAgent(BaseCollaborativeAgent):
    """
    Specialized agent for synthesizing information from multiple sources.

    Capabilities:
    - Combining diverse information
    - Resolving conflicting information
    - Creating coherent narratives
    - Collaborative synthesis

    Example:
        >>> role = AgentRole(
        ...     name="synthesizer",
        ...     description="Synthesizes information into coherent output",
        ...     allowed_tools=["text_generator", "summarizer"]
        ... )
        >>> agent = SynthesisAgent(role, state_manager, message_bus)
        >>> await agent.initialize()
        >>> result = await agent.execute({"sources": ["data1", "data2"]})
    """

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize information from multiple sources.

        Args:
            input_data: May contain 'sources' or references to state data

        Returns:
            Synthesized output with coherent narrative
        """
        self.logger.info("Synthesizing information...")

        # Gather all information from state
        research_findings = await self.read_state("research_findings", {})
        analysis_results = await self.read_state("analysis_results", {})
        additional_data = input_data.get("additional_data", {})

        # Collect all sources
        sources = [
            {"type": "research", "data": research_findings},
            {"type": "analysis", "data": analysis_results},
            {"type": "additional", "data": additional_data}
        ]

        # Phase 1: Synthesize information
        synthesis = await self._synthesize_sources(sources)

        # Phase 2: Get feedback from critics
        refined_synthesis = await self._refine_with_feedback(synthesis)

        # Store in state
        await self.write_state("synthesis_output", refined_synthesis)

        self.logger.info("Synthesis complete")

        return {
            "status": "success",
            "synthesis": refined_synthesis,
            "sources_used": len([s for s in sources if s["data"]])
        }

    async def _synthesize_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Synthesize information from all sources.

        In production, would use advanced NLG and synthesis models.
        """
        # Collect all key points
        key_points = []

        for source in sources:
            source_data = source["data"]
            if not source_data:
                continue

            source_type = source["type"]

            if source_type == "research" and "findings" in source_data:
                findings = source_data.get("findings", [])
                for finding in findings:
                    if isinstance(finding, dict):
                        key_points.append(finding.get("fact", str(finding)))
                    else:
                        key_points.append(str(finding))

            elif source_type == "analysis" and "patterns" in source_data:
                patterns = source_data.get("patterns", [])
                for pattern in patterns:
                    key_points.append(pattern.get("description", str(pattern)))

        # Create synthesis
        synthesis = {
            "summary": self._create_synthesis_summary(key_points),
            "key_points": key_points,
            "sources_count": len([s for s in sources if s["data"]]),
            "completeness": len(key_points) / max(len(sources), 1)
        }

        return synthesis

    def _create_synthesis_summary(self, key_points: List[str]) -> str:
        """Create summary from key points."""
        if not key_points:
            return "No information available to synthesize."

        summary = "Synthesis of findings:\n\n"
        for i, point in enumerate(key_points[:10], 1):
            summary += f"{i}. {point}\n"

        if len(key_points) > 10:
            summary += f"\n... and {len(key_points) - 10} more points."

        return summary

    async def _refine_with_feedback(self, synthesis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get feedback from critic agents and refine synthesis.

        Args:
            synthesis: Initial synthesis

        Returns:
            Refined synthesis incorporating feedback
        """
        active_agents = self.message_bus.get_active_agents()
        critics = [
            agent_id for agent_id in active_agents
            if "critic" in agent_id.lower()
        ]

        if not critics:
            return synthesis

        # Get feedback from first critic
        try:
            response = await self.send_query(
                recipient=critics[0],
                query={
                    "action": "review",
                    "content": synthesis
                },
                timeout=10.0
            )

            feedback = response.content.get("feedback", {})

            # Apply feedback
            if feedback.get("suggestions"):
                synthesis["refinements"] = feedback["suggestions"]
                synthesis["refined"] = True

        except Exception as e:
            self.logger.warning(f"Failed to get critic feedback: {e}")

        return synthesis


class CriticAgent(BaseCollaborativeAgent):
    """
    Specialized agent for reviewing and critiquing work.

    Capabilities:
    - Quality assessment
    - Constructive feedback
    - Error detection
    - Collaborative review

    Example:
        >>> role = AgentRole(
        ...     name="critic",
        ...     description="Reviews work and provides feedback",
        ...     metadata={"criteria": ["accuracy", "completeness", "clarity"]}
        ... )
        >>> agent = CriticAgent(role, state_manager, message_bus)
        >>> await agent.initialize()
        >>> result = await agent.execute({"artifact": "content to review"})
    """

    def __init__(self, role: AgentRole, state_manager: StateManager,
                 message_bus: Optional[MessageBus] = None,
                 criteria: Optional[List[str]] = None):
        """
        Initialize critic agent.

        Args:
            role: Agent role
            state_manager: State manager
            message_bus: Message bus for communication
            criteria: Review criteria (overrides role.metadata)
        """
        super().__init__(role, state_manager, message_bus)
        self.criteria = criteria or role.metadata.get("criteria", [
            "accuracy", "completeness", "clarity", "coherence"
        ])

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Review artifact against criteria.

        Args:
            input_data: Must contain 'artifact' or reference to state

        Returns:
            Review results with feedback and approval status
        """
        # Get artifact to review
        artifact = input_data.get("artifact")
        artifact_key = input_data.get("artifact_key", "synthesis_output")

        if not artifact:
            artifact = await self.read_state(artifact_key)

        if not artifact:
            self.logger.warning(f"No artifact found for review")
            return {"status": "error", "message": "No artifact to review"}

        self.logger.info(f"Reviewing artifact against {len(self.criteria)} criteria")

        # Phase 1: Initial review
        issues = await self._review_artifact(artifact, self.criteria)

        # Phase 2: Collaborate with other critics
        consensus_review = await self._build_consensus(artifact, issues)

        # Determine approval
        approved = len(consensus_review["issues"]) == 0

        # Store review results
        await self.write_state("critique", consensus_review)
        await self.write_state("approved", approved)

        issues_count = len(consensus_review["issues"])
        status_msg = "APPROVED" if approved else f"{issues_count} issues found"
        self.logger.info(f"Review complete: {status_msg}")

        return {
            "status": "success",
            "approved": approved,
            "issues_found": len(consensus_review["issues"]),
            "feedback": consensus_review.get("feedback", "")
        }

    async def _review_artifact(self, artifact: Any,
                              criteria: List[str]) -> List[Dict[str, Any]]:
        """
        Review artifact against criteria.

        In production, would use advanced analysis tools.
        """
        issues = []

        # Check each criterion
        for criterion in criteria:
            issue = self._check_criterion(artifact, criterion)
            if issue:
                issues.append(issue)

        return issues

    def _check_criterion(self, artifact: Any, criterion: str) -> Optional[Dict[str, Any]]:
        """Check artifact against specific criterion."""
        if isinstance(artifact, dict):
            if criterion == "completeness":
                if not artifact.get("key_points"):
                    return {
                        "criterion": criterion,
                        "issue": "Missing key points",
                        "severity": "high"
                    }

            elif criterion == "accuracy":
                confidence = artifact.get("confidence", 1.0)
                if confidence < 0.7:
                    return {
                        "criterion": criterion,
                        "issue": f"Low confidence score: {confidence}",
                        "severity": "medium"
                    }

        return None

    async def _build_consensus(self, artifact: Any,
                              initial_issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build consensus with other critic agents.

        Args:
            artifact: Artifact being reviewed
            initial_issues: Issues found in initial review

        Returns:
            Consensus review results
        """
        active_agents = self.message_bus.get_active_agents()
        other_critics = [
            agent_id for agent_id in active_agents
            if agent_id != self.agent_id and "critic" in agent_id.lower()
        ]

        all_issues = initial_issues.copy()

        # Get reviews from other critics
        for critic_id in other_critics[:2]:  # Limit to 2 other critics
            try:
                response = await self.send_query(
                    recipient=critic_id,
                    query={
                        "action": "review",
                        "artifact": artifact
                    },
                    timeout=10.0
                )

                peer_issues = response.content.get("issues", [])
                all_issues.extend(peer_issues)

            except Exception as e:
                self.logger.warning(f"Failed to get review from {critic_id}: {e}")

        # Deduplicate and prioritize issues
        unique_issues = self._deduplicate_issues(all_issues)

        return {
            "issues": unique_issues,
            "feedback": self._generate_feedback(unique_issues),
            "reviewers_count": 1 + len([c for c in other_critics[:2]])
        }

    def _deduplicate_issues(self, issues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate issues."""
        seen = set()
        unique = []

        for issue in issues:
            key = (issue.get("criterion", ""), issue.get("issue", ""))
            if key not in seen:
                seen.add(key)
                unique.append(issue)

        return unique

    def _generate_feedback(self, issues: List[Dict[str, Any]]) -> str:
        """Generate constructive feedback from issues."""
        if not issues:
            return "Work meets all criteria. Approved."

        feedback_parts = ["The following issues were identified:\n"]

        for i, issue in enumerate(issues, 1):
            feedback_parts.append(
                f"{i}. [{issue.get('severity', 'medium').upper()}] "
                f"{issue.get('criterion', 'General')}: {issue.get('issue', 'Unknown issue')}"
            )

        return "\n".join(feedback_parts)

    async def _handle_query(self, message):
        """Handle review queries from other agents."""
        query = message.content
        action = query.get("action")

        if action == "verify":
            # Verify a fact
            fact = query.get("fact", "")
            confidence = 0.8  # Would use actual verification in production

            await self.send_message(
                recipient=message.sender,
                message_type=MessageType.RESPONSE,
                content={"confidence": confidence, "verified": True},
                reply_to=message.message_id
            )

        elif action == "review_analysis":
            # Review analysis results
            analysis = query.get("analysis", {})
            insight = "Analysis appears sound with good methodology"
            agreement = 0.85

            await self.send_message(
                recipient=message.sender,
                message_type=MessageType.RESPONSE,
                content={"insight": insight, "agreement": agreement},
                reply_to=message.message_id
            )

        elif action == "review":
            # Review artifact
            artifact = query.get("artifact") or query.get("content")
            issues = await self._review_artifact(artifact, self.criteria)

            await self.send_message(
                recipient=message.sender,
                message_type=MessageType.RESPONSE,
                content={
                    "issues": issues,
                    "feedback": self._generate_feedback(issues)
                },
                reply_to=message.message_id
            )

        else:
            # Default query handling from parent
            await super()._handle_query(message)
