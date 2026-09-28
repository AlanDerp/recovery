import type { KnowledgeNodeId, KnowledgePredicate } from "./common.ts";

export interface SkillPort {
	name: string;
	type: string;
	required: boolean;
	description?: string;
}

export interface SkillEffect {
	predicate: KnowledgePredicate;
	confidence: "guaranteed" | "expected" | "observed";
}

export interface InstructionSkillOperation {
	kind: "instruction";
	/** Path or resource identifier for progressively disclosed instructions. */
	resource: string;
}

export interface ToolSkillOperation {
	kind: "tool";
	toolName: string;
	/** Tool argument name to input/state expression. */
	argumentBindings: Readonly<Record<string, string>>;
}

export interface CompositeSkillStep {
	id: string;
	nodeId: KnowledgeNodeId;
	dependsOn: readonly string[];
	when?: readonly KnowledgePredicate[];
	onFailure: "abort" | "replan" | "recover";
}

export interface CompositeSkillOperation {
	kind: "composite";
	steps: readonly CompositeSkillStep[];
}

export type SkillOperation = InstructionSkillOperation | ToolSkillOperation | CompositeSkillOperation;

export interface SkillValidator {
	id: string;
	stage: "pre" | "post" | "continuous";
	kind: "predicate" | "tool" | "human";
	ref: string;
	severity: "info" | "error" | "fatal";
}

export interface SkillFailureMode {
	code: string;
	description: string;
	signature: readonly KnowledgePredicate[];
	severity: "recoverable" | "task-fatal" | "safety-critical";
	retryable: boolean;
	recoveryNodeIds: readonly KnowledgeNodeId[];
}

export interface SkillCompatibility {
	platforms: readonly string[];
	embodiments: readonly string[];
	apiVersions: Readonly<Record<string, string>>;
	requiredTools: readonly string[];
}

export interface SkillCost {
	expectedLatencyMs?: number;
	expectedTokens?: number;
	expectedActions?: number;
	monetaryCost?: number;
	riskExposure?: number;
}

/**
 * Machine-checkable P/O/A/V/F contract for an operational skill.
 * Evidence and learned utility are references, not embedded mutable counters.
 */
export interface SkillContract {
	goal: string;
	inputs: readonly SkillPort[];
	outputs: readonly SkillPort[];
	preconditions: readonly KnowledgePredicate[];
	contraindications: readonly KnowledgePredicate[];
	operation: SkillOperation;
	effects: readonly SkillEffect[];
	validators: readonly SkillValidator[];
	failureModes: readonly SkillFailureMode[];
	hardInvariants: readonly KnowledgePredicate[];
	requiresApproval: boolean;
	compatibility: SkillCompatibility;
	cost: SkillCost;
}
