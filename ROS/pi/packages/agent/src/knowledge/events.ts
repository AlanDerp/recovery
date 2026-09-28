import type { KnowledgeNodeId, MemoryRecord, Portfolio } from "./types.ts";

interface KnowledgeRuntimeEventBase<TType extends string> {
	type: TType;
	/** Contiguous sequence number within one run. The first value may be non-zero for a sliced log. */
	sequence: number;
	timestamp: number;
	runId: string;
	taskId: string;
}

export interface KnowledgeTaskStartedEvent extends KnowledgeRuntimeEventBase<"knowledge_task_started"> {
	goal: string;
}

export interface KnowledgePortfolioCompiledEvent extends KnowledgeRuntimeEventBase<"knowledge_portfolio_compiled"> {
	turn: number;
	portfolio: Portfolio;
	reason: "initial" | "turn" | "recovery";
}

export interface KnowledgeContextProjectedEvent extends KnowledgeRuntimeEventBase<"knowledge_context_projected"> {
	turn: number;
	portfolioId: string;
	nodeIds: readonly KnowledgeNodeId[];
	memoryIds: readonly string[];
}

export interface KnowledgeToolCallEvaluatedEvent extends KnowledgeRuntimeEventBase<"knowledge_tool_call_evaluated"> {
	turn: number;
	portfolioId: string;
	toolCallId: string;
	toolName: string;
	decision: "allow" | "block";
	matchedEntryIds: readonly string[];
	reasons: readonly string[];
}

export interface KnowledgeToolResultObservedEvent extends KnowledgeRuntimeEventBase<"knowledge_tool_result_observed"> {
	turn: number;
	toolCallId: string;
	toolName: string;
	isError: boolean;
}

export interface KnowledgeValidationCompletedEvent extends KnowledgeRuntimeEventBase<"knowledge_validation_completed"> {
	turn: number;
	toolCallId: string;
	validatorIds: readonly string[];
	outcome: "pass" | "fail" | "inconclusive" | "not-applicable";
}

export interface KnowledgeRecoverySelectedEvent extends KnowledgeRuntimeEventBase<"knowledge_recovery_selected"> {
	turn: number;
	failedToolCallId: string;
	recoveryNodeId: KnowledgeNodeId;
	reason: string;
}

export interface KnowledgeTurnCompletedEvent extends KnowledgeRuntimeEventBase<"knowledge_turn_completed"> {
	turn: number;
	outcome: "continue" | "stop" | "recover";
}

export interface KnowledgeRunSettledEvent extends KnowledgeRuntimeEventBase<"knowledge_run_settled"> {
	outcome: "success" | "failure" | "partial" | "aborted" | "blocked";
}

export interface KnowledgeMemoryCommittedEvent extends KnowledgeRuntimeEventBase<"knowledge_memory_committed"> {
	memory: MemoryRecord;
}

export type KnowledgeRuntimeEvent =
	| KnowledgeTaskStartedEvent
	| KnowledgePortfolioCompiledEvent
	| KnowledgeContextProjectedEvent
	| KnowledgeToolCallEvaluatedEvent
	| KnowledgeToolResultObservedEvent
	| KnowledgeValidationCompletedEvent
	| KnowledgeRecoverySelectedEvent
	| KnowledgeTurnCompletedEvent
	| KnowledgeRunSettledEvent
	| KnowledgeMemoryCommittedEvent;

export type KnowledgeEventSequenceIssueCode =
	| "empty_sequence"
	| "invalid_first_event"
	| "non_contiguous_sequence"
	| "mixed_run"
	| "mixed_task"
	| "event_after_settlement"
	| "unknown_portfolio"
	| "context_not_projected"
	| "duplicate_tool_call"
	| "unknown_tool_call"
	| "blocked_tool_has_result"
	| "result_before_allow"
	| "duplicate_tool_result"
	| "validation_before_result"
	| "duplicate_validation"
	| "recovery_without_failure"
	| "incomplete_tool_call"
	| "recovery_not_selected"
	| "settlement_before_turn"
	| "memory_before_settlement";

export interface KnowledgeEventSequenceIssue {
	code: KnowledgeEventSequenceIssueCode;
	index: number;
	message: string;
}

interface ToolCallState {
	turn: number;
	decision: "allow" | "block";
	resultObserved: boolean;
	validation?: KnowledgeValidationCompletedEvent["outcome"];
}

/**
 * Validate the event contract shared by the Pi extension adapter and durable evidence writer.
 * The validator is intentionally pure so recovery and replay code can use it before mutating storage.
 */
export function validateKnowledgeEventSequence(
	events: readonly KnowledgeRuntimeEvent[],
): readonly KnowledgeEventSequenceIssue[] {
	const issues: KnowledgeEventSequenceIssue[] = [];
	if (events.length === 0) {
		return [{ code: "empty_sequence", index: 0, message: "Knowledge event sequence is empty" }];
	}

	const first = events[0];
	if (first.type !== "knowledge_task_started") {
		issues.push({
			code: "invalid_first_event",
			index: 0,
			message: "The first knowledge event must be knowledge_task_started",
		});
	}

	const runId = first.runId;
	const taskId = first.taskId;
	const portfolios = new Set<string>();
	const projectedPortfolioByTurn = new Map<number, string>();
	const toolCalls = new Map<string, ToolCallState>();
	const failedToolCalls = new Set<string>();
	const recoveryTurns = new Set<number>();
	const completedTurns = new Set<number>();
	let settled = false;

	for (let index = 0; index < events.length; index++) {
		const event = events[index];
		if (index > 0 && event.sequence !== events[index - 1].sequence + 1) {
			issues.push({
				code: "non_contiguous_sequence",
				index,
				message: `Event sequence ${event.sequence} does not follow ${events[index - 1].sequence}`,
			});
		}
		if (event.runId !== runId) {
			issues.push({ code: "mixed_run", index, message: `Event belongs to run ${event.runId}, expected ${runId}` });
		}
		if (event.taskId !== taskId) {
			issues.push({
				code: "mixed_task",
				index,
				message: `Event belongs to task ${event.taskId}, expected ${taskId}`,
			});
		}

		if (settled && event.type !== "knowledge_memory_committed") {
			issues.push({
				code: "event_after_settlement",
				index,
				message: `${event.type} cannot occur after knowledge_run_settled`,
			});
		}

		switch (event.type) {
			case "knowledge_task_started":
				break;
			case "knowledge_portfolio_compiled":
				portfolios.add(event.portfolio.id);
				break;
			case "knowledge_context_projected":
				if (!portfolios.has(event.portfolioId)) {
					issues.push({
						code: "unknown_portfolio",
						index,
						message: `Context references unknown portfolio ${event.portfolioId}`,
					});
				}
				projectedPortfolioByTurn.set(event.turn, event.portfolioId);
				break;
			case "knowledge_tool_call_evaluated":
				if (projectedPortfolioByTurn.get(event.turn) !== event.portfolioId) {
					issues.push({
						code: "context_not_projected",
						index,
						message: `Tool call ${event.toolCallId} was evaluated without its portfolio being projected`,
					});
				}
				if (toolCalls.has(event.toolCallId)) {
					issues.push({
						code: "duplicate_tool_call",
						index,
						message: `Tool call ${event.toolCallId} was evaluated more than once`,
					});
				} else {
					toolCalls.set(event.toolCallId, {
						turn: event.turn,
						decision: event.decision,
						resultObserved: false,
					});
				}
				break;
			case "knowledge_tool_result_observed": {
				const state = toolCalls.get(event.toolCallId);
				if (!state) {
					issues.push({
						code: "unknown_tool_call",
						index,
						message: `Result references unknown tool call ${event.toolCallId}`,
					});
				} else if (state.decision === "block") {
					issues.push({
						code: "blocked_tool_has_result",
						index,
						message: `Blocked tool call ${event.toolCallId} cannot produce a result observation`,
					});
				} else if (state.resultObserved) {
					issues.push({
						code: "duplicate_tool_result",
						index,
						message: `Tool call ${event.toolCallId} produced more than one result observation`,
					});
				} else {
					state.resultObserved = true;
				}
				break;
			}
			case "knowledge_validation_completed": {
				const state = toolCalls.get(event.toolCallId);
				if (!state) {
					issues.push({
						code: "unknown_tool_call",
						index,
						message: `Validation references unknown tool call ${event.toolCallId}`,
					});
				} else if (!state.resultObserved) {
					issues.push({
						code: "validation_before_result",
						index,
						message: `Tool call ${event.toolCallId} was validated before its result was observed`,
					});
				} else if (state.validation !== undefined) {
					issues.push({
						code: "duplicate_validation",
						index,
						message: `Tool call ${event.toolCallId} was validated more than once`,
					});
				} else {
					state.validation = event.outcome;
					if (event.outcome === "fail") failedToolCalls.add(event.toolCallId);
				}
				break;
			}
			case "knowledge_recovery_selected":
				if (!failedToolCalls.has(event.failedToolCallId)) {
					issues.push({
						code: "recovery_without_failure",
						index,
						message: `Recovery references tool call ${event.failedToolCallId} without a failed validation`,
					});
				}
				recoveryTurns.add(event.turn);
				break;
			case "knowledge_turn_completed":
				for (const [toolCallId, state] of toolCalls) {
					if (state.turn !== event.turn || state.decision === "block") continue;
					if (!state.resultObserved || state.validation === undefined) {
						issues.push({
							code: "incomplete_tool_call",
							index,
							message: `Allowed tool call ${toolCallId} did not finish observation and validation before turn completion`,
						});
					}
				}
				if (event.outcome === "recover" && !recoveryTurns.has(event.turn)) {
					issues.push({
						code: "recovery_not_selected",
						index,
						message: `Turn ${event.turn} completed as recover without a recovery selection`,
					});
				}
				completedTurns.add(event.turn);
				break;
			case "knowledge_run_settled":
				if (completedTurns.size === 0) {
					issues.push({
						code: "settlement_before_turn",
						index,
						message: "Run settled before any turn completed",
					});
				}
				settled = true;
				break;
			case "knowledge_memory_committed":
				if (!settled) {
					issues.push({
						code: "memory_before_settlement",
						index,
						message: `Memory ${event.memory.id} was committed before the run settled`,
					});
				}
				break;
		}
	}

	return issues;
}

export class KnowledgeEventSequenceError extends Error {
	readonly issues: readonly KnowledgeEventSequenceIssue[];

	constructor(issues: readonly KnowledgeEventSequenceIssue[]) {
		super(issues.map((issue) => `${issue.index}:${issue.code}: ${issue.message}`).join("\n"));
		this.name = "KnowledgeEventSequenceError";
		this.issues = issues;
	}
}

export function assertValidKnowledgeEventSequence(events: readonly KnowledgeRuntimeEvent[]): void {
	const issues = validateKnowledgeEventSequence(events);
	if (issues.length > 0) throw new KnowledgeEventSequenceError(issues);
}
