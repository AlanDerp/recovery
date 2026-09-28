/** JSON-compatible value stored in knowledge contracts, memories, and event metadata. */
export type KnowledgeValue =
	| string
	| number
	| boolean
	| null
	| readonly KnowledgeValue[]
	| { readonly [key: string]: KnowledgeValue };

/** Stable identifier for a versioned knowledge node. */
export type KnowledgeNodeId = string;

export interface KnowledgePredicate {
	/** Predicate name, for example `workspace_clean` or `tool_available`. */
	name: string;
	arguments?: Readonly<Record<string, KnowledgeValue>>;
	expected?: KnowledgeValue;
	source?: "state" | "tool" | "memory" | "user" | "policy";
}
