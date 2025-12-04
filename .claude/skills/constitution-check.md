# Constitution Compliance Checker Skill

## Purpose
Validate any artifact (spec, plan, implementation, content) against the Physical AI Textbook Constitution principles to ensure quality and compliance.

## When to Use
- Before finalizing feature specifications
- After creating implementation plans
- Before committing chapter content
- During code review
- When validating architectural decisions

## Input
- **artifact_type**: spec | plan | tasks | chapter | code | architecture
- **artifact_path**: Path to the file or content to validate
- **principles_to_check**: Optional - specific principles to focus on (default: all)

## Constitution Principles Reference

### Principle I: Educational-First Content Design
- Content structured as progression from fundamentals to advanced topics
- Clear learning objectives stated at the beginning
- Complex concepts introduced incrementally with concrete examples
- Technical content includes both theoretical explanation and practical application
- Code examples are runnable, tested, and include expected outputs
- Visual aids accompany complex explanations

### Principle II: Modular Content Architecture
- Content organized into independent, reusable modules
- Each module completable independently
- Explicit prerequisites and learning dependencies declared
- No circular dependencies between modules
- Chapters are self-contained
- Cross-module references use explicit links and context

### Principle III: Interactive Learning Through RAG
- RAG system can answer questions on any section
- Content structured to support text-selection-based queries
- RAG responses can cite specific sections
- Handles both beginner and advanced queries
- Non-intrusive integration
- Maintains conversation context

### Principle IV: Code-First Technical Validation
- Code snippets syntactically correct and tested
- ROS 2 examples tested with specified distribution
- Simulation examples include setup, execution, expected results
- URDF models validated for syntax
- Python agent-ROS bridges demonstrate working message passing
- Commands verified on target platform

### Principle V: Accessibility and Personalization
- Assumes only basic programming knowledge
- Hardware/robotics background not assumed
- Technical jargon defined on first use
- User background profiles inform content adaptation
- Personalization adjusts content depth
- Translation preserves technical accuracy
- Accessibility standards supported

### Principle VI: Deployment and Performance Standards
- Build completes without errors/warnings
- Automated deployment via CI/CD
- Page load time under 3 seconds
- RAG API responds within 2 seconds
- Database handles concurrent queries
- Vector search under 500ms
- Secure authentication
- Rate limiting and input validation

### Principle VII: Spec-Driven Development with Claude Code
- Features start with specification
- Specs reviewed and approved before implementation
- Implementation references specification
- Subagents and skills created for reusable workflows
- Architectural decisions documented as ADRs
- Prompt History Records for significant work
- Test-First discipline where applicable

### Principle VIII: Version Control and Collaboration Hygiene
- Atomic, well-described commits
- Branch naming: `feature/<###-feature-name>`
- Pull requests include description, testing, demo
- No secrets committed
- Setup instructions in README
- Dependencies pinned
- Documentation updated with code

## Validation Process

### For Specifications (spec.md)
1. Check user stories are prioritized and independently testable
2. Verify functional requirements are specific and measurable
3. Ensure success criteria are technology-agnostic
4. Validate edge cases are documented
5. Check for constitution principle alignment

### For Plans (plan.md)
1. Verify technical context completeness
2. Check constitution check section exists and passes
3. Validate project structure matches constitution standards
4. Ensure complexity justifications for violations
5. Verify ADR references for significant decisions

### For Tasks (tasks.md)
1. Check tasks include exact file paths
2. Verify parallel tasks marked with [P]
3. Ensure user story alignment [US#]
4. Validate test-first ordering (tests before implementation)
5. Check foundational phase blocks user stories appropriately

### For Chapter Content
1. Verify learning objectives stated clearly
2. Check code examples are complete and tested
3. Ensure progressive difficulty
4. Validate minimum 2 visual aids for complex topics
5. Check 1500-2000 word length
6. Verify 40% code / 60% text ratio
7. Ensure exercises at 3 difficulty levels

### For Code
1. Validate syntax correctness
2. Check for security vulnerabilities (no hardcoded secrets)
3. Verify error handling
4. Ensure comments where logic isn't self-evident
5. Check imports and dependencies exist

### For Architecture Decisions
1. Verify options considered documented
2. Check trade-offs analyzed
3. Ensure decision rationale clear
4. Validate ADR created for significant decisions
5. Check alignment with tech stack requirements

## Output Format

```markdown
# Constitution Compliance Report

**Artifact**: [path/to/artifact]
**Type**: [artifact_type]
**Date**: [ISO date]
**Overall Status**: ✅ PASS | ⚠️ PASS WITH WARNINGS | ❌ FAIL

## Principle Compliance

### ✅ Principle I: Educational-First Content Design
- [✅] Learning objectives stated clearly
- [✅] Progressive difficulty maintained
- [✅] Code examples are runnable
- [⚠️] Only 1 diagram present (minimum 2 required)

**Status**: PASS WITH WARNINGS
**Recommendation**: Add one more diagram to illustrate ROS 2 node communication

### [Continue for all relevant principles]

## Violations Found

### Critical (Must Fix Before Proceeding)
1. **Principle IV**: Code example in section 3.2 has syntax error (line 45)
   - **Fix**: Change `rclpy.init()` to `rclpy.init(args=args)`

### Warnings (Should Address)
1. **Principle I**: Chapter length is 1200 words (target 1500-2000)
   - **Recommendation**: Expand "Core Concepts" section with more examples

### Suggestions (Nice to Have)
1. **Principle V**: Consider adding glossary links for "URDF", "TF2"

## Summary

**Total Checks**: 28
**Passed**: 24
**Warnings**: 3
**Failed**: 1

**Action Required**: Fix critical violations before proceeding to implementation.

## Next Steps
1. Fix syntax error in code example (section 3.2)
2. Add second diagram illustrating concept
3. Expand content to meet word count target
```

## Usage Example

```bash
# Via Skill tool
skill: "constitution-check"
  artifact_type: "chapter"
  artifact_path: "docs/module1/ros2-nodes.md"

# Check specific principles only
skill: "constitution-check"
  artifact_type: "spec"
  artifact_path: "specs/001-rag-chatbot/spec.md"
  principles_to_check: ["I", "III", "VII"]
```

## Quality Gates

This skill enforces quality gates:
- **FAIL**: Critical violations exist - do not proceed
- **PASS WITH WARNINGS**: Can proceed but should address warnings
- **PASS**: Fully compliant - proceed confidently

## Integration Points

- Run before `/sp.plan` to validate spec compliance
- Run after content generation to validate chapters
- Run in CI/CD pipeline for automated quality checks
- Run before pull request submission

## Success Metrics

- Catches 100% of critical constitution violations
- Reduces manual review time by 70%
- Ensures consistent quality across all artifacts
- Provides actionable feedback for improvements
