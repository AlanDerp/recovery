import type { KnowledgeNodeId, KnowledgePredicate } from "./common.ts";
import type { SkillContract } from "./skill-contract.ts";

/** Semantic levels. Numeric L0-L4 labels are deliberately excluded because their direction is ambiguous. */
export type KnowledgeNodeKind = "principle" | "method" | "strategy" | "skill" | "atomic-op";

export type KnowledgeNodeStatus = "candidate" | "validated" | "stable" | "deprecated" | "blocked";

export type KnowledgeRelationKind =
	| "is-a"
	| "decomposes-to"
	| "requires"
	| "can-follow"
	| "compatible-with"
	| "alternative-to"
	| "supports"
	| "recovers-with"
	| "contradicts";

export interface KnowledgeRelation {
	kind: KnowledgeRelationKind;
	targetId: KnowledgeNodeId;
	when?: readonly KnowledgePredicate[];
	note?: string;
}

export interface KnowledgeProvenance {
	source: "human" | "skill" | "trajectory" | "memory" | "distilled" | "imported";
	sourceIds: readonly string[];
	createdAt: number;
	updatedAt: number;
	createdBy?: string;
}

export interface KnowledgeNodeBase<TKind extends KnowledgeNodeKind> {
	id: KnowledgeNodeId;
	/** Version is pinned by portfolios so a running task cannot silently switch revisions. */
	version: string;
	kind: TKind;
	status: KnowledgeNodeStatus;
	title: string;
	summary: string;
	tags: readonly string[];
	relations: readonly KnowledgeRelation[];
	provenance: KnowledgeProvenance;
}

export interface PrincipleKnowledgeNode extends KnowledgeNodeBase<"principle"> {
	statement: string;
	enforcement: "hard" | "soft";
	applicability: readonly KnowledgePredicate[];
	invariants: readonly KnowledgePredicate[];
	counterexamples: readonly string[];
}

export interface MethodKnowledgeNode extends KnowledgeNodeBase<"method"> {
	objective: string;
	procedure: readonly string[];
	applicability: readonly KnowledgePredicate[];
}

export interface StrategyStage {
	id: string;
	objective: string;
	nodeIds: readonly KnowledgeNodeId[];
	dependsOn: readonly string[];
}

export interface StrategyKnowledgeNode extends KnowledgeNodeBase<"strategy"> {
	goalPattern: string;
	stages: readonly StrategyStage[];
}

export interface SkillKnowledgeNode extends KnowledgeNodeBase<"skill"> {
	contract: SkillContract;
}

export interface AtomicOpKnowledgeNode extends KnowledgeNodeBase<"atomic-op"> {
	toolName: string;
	executionMode: "sequential" | "parallel";
	inputType: string;
	resultType: string;
}

export type KnowledgeNode =
	| PrincipleKnowledgeNode
	| MethodKnowledgeNode
	| StrategyKnowledgeNode
	| SkillKnowledgeNode
	| AtomicOpKnowledgeNode;

export type { KnowledgeNodeId, KnowledgePredicate } from "./common.ts";
