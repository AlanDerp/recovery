import type { KnowledgeNodeId, KnowledgePredicate, KnowledgeValue } from "./common.ts";

export type MemoryRecordKind = "working" | "episodic" | "semantic";

export interface MemoryProvenance {
	source: "runtime" | "user" | "imported" | "consolidator";
	sourceIds: readonly string[];
	modelId?: string;
	codeVersion?: string;
}

export interface TaskMemoryScope {
	kind: "task";
	sessionId: string;
	taskId: string;
}

export interface SharedMemoryScope {
	kind: "project" | "global";
	projectId?: string;
}

export interface MemoryRecordBase<TKind extends MemoryRecordKind> {
	id: string;
	kind: TKind;
	createdAt: number;
	tags: readonly string[];
	/** Expected range is [0, 1]; runtime schemas enforce the numeric bounds. */
	salience: number;
	provenance: MemoryProvenance;
}

export interface WorkingMemoryRecord extends MemoryRecordBase<"working"> {
	scope: TaskMemoryScope;
	revision: number;
	goal: string;
	facts: readonly KnowledgePredicate[];
	activePortfolioId?: string;
	expiresAt?: number;
}

export interface MemoryObservation {
	id: string;
	at: number;
	kind: "state" | "tool-call" | "tool-result" | "validation" | "user-feedback";
	toolCallId?: string;
	toolName?: string;
	data: Readonly<Record<string, KnowledgeValue>>;
}

export interface MemoryOutcome {
	status: "success" | "failure" | "partial" | "aborted" | "blocked";
	summary: string;
	reward?: number;
	failureSeverity?: "recoverable" | "task-fatal" | "safety-critical";
	validatorResults: Readonly<Record<string, "pass" | "fail" | "inconclusive" | "not-applicable">>;
}

export interface EpisodicMemoryRecord extends MemoryRecordBase<"episodic"> {
	scope: TaskMemoryScope;
	episodeId: string;
	startedAt: number;
	endedAt: number;
	contextFingerprint: string;
	portfolioIds: readonly string[];
	knowledgeNodeIds: readonly KnowledgeNodeId[];
	observations: readonly MemoryObservation[];
	outcome: MemoryOutcome;
}

export interface SemanticMemoryRecord extends MemoryRecordBase<"semantic"> {
	scope: TaskMemoryScope | SharedMemoryScope;
	claim: string;
	applicability: readonly KnowledgePredicate[];
	supportingMemoryIds: readonly string[];
	contradictingMemoryIds: readonly string[];
	confidence: number;
	promotion: "candidate" | "validated" | "rejected" | "promoted";
	promotedNodeId?: KnowledgeNodeId;
}

export type MemoryRecord = WorkingMemoryRecord | EpisodicMemoryRecord | SemanticMemoryRecord;
