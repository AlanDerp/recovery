import { describe, expect, it } from "vitest";
import type { KnowledgeNode, MemoryRecord, Portfolio, SkillContract } from "../../src/knowledge/index.ts";

const contract = {
	goal: "Read a file and return verified content",
	inputs: [{ name: "path", type: "absolute-path", required: true }],
	outputs: [{ name: "content", type: "text", required: true }],
	preconditions: [{ name: "file_exists", source: "state" }],
	contraindications: [{ name: "path_is_secret", expected: true, source: "policy" }],
	operation: { kind: "tool", toolName: "read", argumentBindings: { path: "inputs.path" } },
	effects: [{ predicate: { name: "content_available" }, confidence: "expected" }],
	validators: [
		{ id: "validator.non-empty", stage: "post", kind: "predicate", ref: "content_non_empty", severity: "error" },
	],
	failureModes: [
		{
			code: "FILE_NOT_FOUND",
			description: "The file moved or the path is stale",
			signature: [{ name: "tool_error", arguments: { code: "ENOENT" } }],
			severity: "recoverable",
			retryable: true,
			recoveryNodeIds: ["skill.relocate-file"],
		},
	],
	hardInvariants: [{ name: "path_authorized", expected: true, source: "policy" }],
	requiresApproval: false,
	compatibility: {
		platforms: ["node"],
		embodiments: [],
		apiVersions: { read: "1" },
		requiredTools: ["read"],
	},
	cost: { expectedActions: 1, riskExposure: 0.1 },
} satisfies SkillContract;

const skillNode = {
	id: "skill.read-file",
	version: "1.0.0",
	kind: "skill",
	status: "validated",
	title: "Read file",
	summary: "Read and validate file content",
	tags: ["filesystem"],
	relations: [{ kind: "requires", targetId: "atomic-op.read" }],
	provenance: { source: "human", sourceIds: ["design-1"], createdAt: 1, updatedAt: 1 },
	contract,
} satisfies KnowledgeNode;

const workingMemory = {
	id: "memory.working-1",
	kind: "working",
	scope: { kind: "task", sessionId: "session-1", taskId: "task-1" },
	createdAt: 1,
	tags: ["active"],
	salience: 1,
	provenance: { source: "runtime", sourceIds: ["run-1"] },
	revision: 1,
	goal: "Read the requested file",
	facts: [{ name: "file_exists", expected: true, source: "state" }],
	activePortfolioId: "portfolio-1",
} satisfies MemoryRecord;

const portfolio = {
	id: "portfolio-1",
	revision: 1,
	taskId: "task-1",
	contextFingerprint: "context-1",
	status: "active",
	entries: [
		{
			id: "entry-1",
			node: { id: skillNode.id, version: skillNode.version },
			role: "anchor",
			gate: { outcome: "pass", reasons: [], checkedPredicates: contract.preconditions },
			score: {
				total: 0.8,
				terms: [
					{ component: "relevance", value: 0.9, weight: 1, contribution: 0.9 },
					{ component: "risk", value: 0.1, weight: -1, contribution: -0.1 },
				],
			},
			dependsOn: [],
			selectionReason: "Best applicable filesystem skill",
		},
	],
	rejected: [],
	edges: [],
	budget: { maxActions: 2, maxRisk: 0.2 },
	createdAt: 1,
} satisfies Portfolio;

describe("knowledge contract types", () => {
	it("keeps the four contracts linked by stable ids and pinned versions", () => {
		expect(skillNode.kind).toBe("skill");
		expect(skillNode.contract.operation.kind).toBe("tool");
		expect(workingMemory.activePortfolioId).toBe(portfolio.id);
		expect(portfolio.entries[0].node).toEqual({ id: skillNode.id, version: skillNode.version });
	});
});
