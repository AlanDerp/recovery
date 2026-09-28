import { describe, expect, it } from "vitest";
import {
	assertValidKnowledgeEventSequence,
	type EpisodicMemoryRecord,
	type KnowledgeRuntimeEvent,
	type Portfolio,
	validateKnowledgeEventSequence,
} from "../../src/knowledge/index.ts";

const basePortfolio = {
	id: "portfolio-1",
	revision: 1,
	taskId: "task-1",
	contextFingerprint: "ctx-1",
	status: "active",
	entries: [
		{
			id: "entry-anchor",
			node: { id: "skill.read-file", version: "1.0.0" },
			role: "anchor",
			gate: { outcome: "pass", reasons: [], checkedPredicates: [] },
			score: {
				total: 0.9,
				terms: [{ component: "relevance", value: 0.9, weight: 1, contribution: 0.9 }],
			},
			dependsOn: [],
			selectionReason: "Matches the task goal",
		},
	],
	rejected: [],
	edges: [],
	budget: { maxActions: 4 },
	createdAt: 1,
} satisfies Portfolio;

const episodeMemory = {
	id: "memory-1",
	kind: "episodic",
	scope: { kind: "task", sessionId: "session-1", taskId: "task-1" },
	createdAt: 20,
	tags: ["test"],
	salience: 0.8,
	provenance: { source: "runtime", sourceIds: ["run-1"] },
	episodeId: "episode-1",
	startedAt: 1,
	endedAt: 20,
	contextFingerprint: "ctx-1",
	portfolioIds: ["portfolio-1"],
	knowledgeNodeIds: ["skill.read-file"],
	observations: [],
	outcome: {
		status: "success",
		summary: "File was read and verified",
		validatorResults: { "validator.read": "pass" },
	},
} satisfies EpisodicMemoryRecord;

type EventInput = KnowledgeRuntimeEvent extends infer TEvent
	? TEvent extends KnowledgeRuntimeEvent
		? Omit<TEvent, "sequence" | "timestamp" | "runId" | "taskId">
		: never
	: never;

function event(sequence: number, value: EventInput): KnowledgeRuntimeEvent {
	return {
		...value,
		sequence,
		timestamp: sequence + 1,
		runId: "run-1",
		taskId: "task-1",
	} as KnowledgeRuntimeEvent;
}

describe("knowledge runtime event sequence", () => {
	it("accepts a successful tool call through validation and evidence commit", () => {
		const events: KnowledgeRuntimeEvent[] = [
			event(0, { type: "knowledge_task_started", goal: "Read the file" }),
			event(1, { type: "knowledge_portfolio_compiled", turn: 0, portfolio: basePortfolio, reason: "initial" }),
			event(2, {
				type: "knowledge_context_projected",
				turn: 0,
				portfolioId: "portfolio-1",
				nodeIds: ["skill.read-file"],
				memoryIds: [],
			}),
			event(3, {
				type: "knowledge_tool_call_evaluated",
				turn: 0,
				portfolioId: "portfolio-1",
				toolCallId: "call-1",
				toolName: "read",
				decision: "allow",
				matchedEntryIds: ["entry-anchor"],
				reasons: [],
			}),
			event(4, {
				type: "knowledge_tool_result_observed",
				turn: 0,
				toolCallId: "call-1",
				toolName: "read",
				isError: false,
			}),
			event(5, {
				type: "knowledge_validation_completed",
				turn: 0,
				toolCallId: "call-1",
				validatorIds: ["validator.read"],
				outcome: "pass",
			}),
			event(6, { type: "knowledge_turn_completed", turn: 0, outcome: "stop" }),
			event(7, { type: "knowledge_run_settled", outcome: "success" }),
			event(8, { type: "knowledge_memory_committed", memory: episodeMemory }),
		];

		expect(validateKnowledgeEventSequence(events)).toEqual([]);
		expect(() => assertValidKnowledgeEventSequence(events)).not.toThrow();
	});

	it("accepts a hard-gated call without tool result or validation events", () => {
		const events: KnowledgeRuntimeEvent[] = [
			event(0, { type: "knowledge_task_started", goal: "Delete protected data" }),
			event(1, { type: "knowledge_portfolio_compiled", turn: 0, portfolio: basePortfolio, reason: "initial" }),
			event(2, {
				type: "knowledge_context_projected",
				turn: 0,
				portfolioId: "portfolio-1",
				nodeIds: ["principle.protect-data"],
				memoryIds: [],
			}),
			event(3, {
				type: "knowledge_tool_call_evaluated",
				turn: 0,
				portfolioId: "portfolio-1",
				toolCallId: "call-blocked",
				toolName: "bash",
				decision: "block",
				matchedEntryIds: [],
				reasons: ["Protected target requires explicit approval"],
			}),
			event(4, { type: "knowledge_turn_completed", turn: 0, outcome: "stop" }),
			event(5, { type: "knowledge_run_settled", outcome: "blocked" }),
		];

		expect(validateKnowledgeEventSequence(events)).toEqual([]);
	});

	it("accepts failed validation followed by recovery and a new portfolio revision", () => {
		const recoveryPortfolio = {
			...basePortfolio,
			id: "portfolio-2",
			revision: 2,
			status: "recovering",
			contextFingerprint: "ctx-2",
		} satisfies Portfolio;
		const events: KnowledgeRuntimeEvent[] = [
			event(0, { type: "knowledge_task_started", goal: "Read and verify the file" }),
			event(1, { type: "knowledge_portfolio_compiled", turn: 0, portfolio: basePortfolio, reason: "initial" }),
			event(2, {
				type: "knowledge_context_projected",
				turn: 0,
				portfolioId: "portfolio-1",
				nodeIds: ["skill.read-file"],
				memoryIds: [],
			}),
			event(3, {
				type: "knowledge_tool_call_evaluated",
				turn: 0,
				portfolioId: "portfolio-1",
				toolCallId: "call-1",
				toolName: "read",
				decision: "allow",
				matchedEntryIds: ["entry-anchor"],
				reasons: [],
			}),
			event(4, {
				type: "knowledge_tool_result_observed",
				turn: 0,
				toolCallId: "call-1",
				toolName: "read",
				isError: true,
			}),
			event(5, {
				type: "knowledge_validation_completed",
				turn: 0,
				toolCallId: "call-1",
				validatorIds: ["validator.read"],
				outcome: "fail",
			}),
			event(6, {
				type: "knowledge_recovery_selected",
				turn: 0,
				failedToolCallId: "call-1",
				recoveryNodeId: "skill.relocate-file",
				reason: "Original path was stale",
			}),
			event(7, { type: "knowledge_turn_completed", turn: 0, outcome: "recover" }),
			event(8, {
				type: "knowledge_portfolio_compiled",
				turn: 1,
				portfolio: recoveryPortfolio,
				reason: "recovery",
			}),
			event(9, {
				type: "knowledge_context_projected",
				turn: 1,
				portfolioId: "portfolio-2",
				nodeIds: ["skill.relocate-file", "skill.read-file"],
				memoryIds: [],
			}),
			event(10, { type: "knowledge_turn_completed", turn: 1, outcome: "stop" }),
			event(11, { type: "knowledge_run_settled", outcome: "success" }),
		];

		expect(validateKnowledgeEventSequence(events)).toEqual([]);
	});

	it("allows multiple memory records only after settlement", () => {
		const semanticMemory = {
			id: "memory-2",
			kind: "semantic",
			scope: { kind: "project", projectId: "project-1" },
			createdAt: 21,
			tags: ["candidate"],
			salience: 0.6,
			provenance: { source: "consolidator", sourceIds: ["memory-1"] },
			claim: "Verify a read result before depending on its content",
			applicability: [],
			supportingMemoryIds: ["memory-1"],
			contradictingMemoryIds: [],
			confidence: 0.7,
			promotion: "candidate",
		} as const;
		const events: KnowledgeRuntimeEvent[] = [
			event(0, { type: "knowledge_task_started", goal: "Answer without tools" }),
			event(1, { type: "knowledge_portfolio_compiled", turn: 0, portfolio: basePortfolio, reason: "initial" }),
			event(2, {
				type: "knowledge_context_projected",
				turn: 0,
				portfolioId: "portfolio-1",
				nodeIds: [],
				memoryIds: [],
			}),
			event(3, { type: "knowledge_turn_completed", turn: 0, outcome: "stop" }),
			event(4, { type: "knowledge_run_settled", outcome: "success" }),
			event(5, { type: "knowledge_memory_committed", memory: episodeMemory }),
			event(6, { type: "knowledge_memory_committed", memory: semanticMemory }),
		];

		expect(validateKnowledgeEventSequence(events)).toEqual([]);
	});

	it("rejects result observation for a blocked tool call", () => {
		const events: KnowledgeRuntimeEvent[] = [
			event(0, { type: "knowledge_task_started", goal: "Unsafe action" }),
			event(1, { type: "knowledge_portfolio_compiled", turn: 0, portfolio: basePortfolio, reason: "initial" }),
			event(2, {
				type: "knowledge_context_projected",
				turn: 0,
				portfolioId: "portfolio-1",
				nodeIds: [],
				memoryIds: [],
			}),
			event(3, {
				type: "knowledge_tool_call_evaluated",
				turn: 0,
				portfolioId: "portfolio-1",
				toolCallId: "call-1",
				toolName: "bash",
				decision: "block",
				matchedEntryIds: [],
				reasons: ["Denied"],
			}),
			event(4, {
				type: "knowledge_tool_result_observed",
				turn: 0,
				toolCallId: "call-1",
				toolName: "bash",
				isError: true,
			}),
		];

		expect(validateKnowledgeEventSequence(events).map((issue) => issue.code)).toContain("blocked_tool_has_result");
	});

	it("rejects evidence commit before settlement and incomplete allowed calls", () => {
		const events: KnowledgeRuntimeEvent[] = [
			event(0, { type: "knowledge_task_started", goal: "Read the file" }),
			event(1, { type: "knowledge_portfolio_compiled", turn: 0, portfolio: basePortfolio, reason: "initial" }),
			event(2, {
				type: "knowledge_context_projected",
				turn: 0,
				portfolioId: "portfolio-1",
				nodeIds: [],
				memoryIds: [],
			}),
			event(3, {
				type: "knowledge_tool_call_evaluated",
				turn: 0,
				portfolioId: "portfolio-1",
				toolCallId: "call-1",
				toolName: "read",
				decision: "allow",
				matchedEntryIds: [],
				reasons: [],
			}),
			event(4, { type: "knowledge_memory_committed", memory: episodeMemory }),
			event(5, { type: "knowledge_turn_completed", turn: 0, outcome: "stop" }),
		];

		const codes = validateKnowledgeEventSequence(events).map((issue) => issue.code);
		expect(codes).toContain("memory_before_settlement");
		expect(codes).toContain("incomplete_tool_call");
	});
});
