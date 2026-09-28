import type { KnowledgeNodeId, KnowledgePredicate } from "./common.ts";

export type PortfolioRole = "anchor" | "prerequisite" | "support" | "validator" | "recovery" | "avoid";

export type PortfolioScoreComponent =
	| "relevance"
	| "applicability"
	| "utility"
	| "transfer"
	| "freshness"
	| "compatibility"
	| "cost"
	| "risk"
	| "redundancy";

export interface PortfolioScoreTerm {
	component: PortfolioScoreComponent;
	value: number;
	weight: number;
	contribution: number;
}

export interface PortfolioScore {
	total: number;
	terms: readonly PortfolioScoreTerm[];
}

export interface PortfolioGateDecision {
	outcome: "pass" | "exclude" | "block";
	reasons: readonly string[];
	checkedPredicates: readonly KnowledgePredicate[];
}

export interface PortfolioNodeRef {
	id: KnowledgeNodeId;
	version: string;
}

export interface PortfolioEntry {
	id: string;
	node: PortfolioNodeRef;
	role: PortfolioRole;
	gate: PortfolioGateDecision;
	score: PortfolioScore;
	dependsOn: readonly string[];
	selectionReason: string;
}

export interface PortfolioEdge {
	fromEntryId: string;
	toEntryId: string;
	kind: "depends-on" | "before" | "fallback" | "verifies";
}

export interface PortfolioBudget {
	maxContextTokens?: number;
	maxLatencyMs?: number;
	maxActions?: number;
	maxRisk?: number;
}

/** Immutable, context-specific selection snapshot. Re-ranking creates a new revision. */
export interface Portfolio {
	id: string;
	revision: number;
	taskId: string;
	contextFingerprint: string;
	status: "proposed" | "active" | "recovering" | "completed" | "aborted";
	entries: readonly PortfolioEntry[];
	rejected: readonly PortfolioEntry[];
	edges: readonly PortfolioEdge[];
	budget: PortfolioBudget;
	createdAt: number;
}
