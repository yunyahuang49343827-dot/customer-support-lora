export type MetricSnapshot = {
  base: number;
  lora: number;
  absolute_delta: number;
  unit: "percentage_points" | "milliseconds";
};

export type BenchmarkSnapshot = {
  source: string;
  evaluated_rows: number;
  metrics: Record<string, MetricSnapshot>;
  locked_test_rerun_performed: false;
};

export type ExpectedResult = {
  intent: string;
  category: string;
  needs_human: boolean;
};

export type ModelSnapshot = {
  intent: string | null;
  category: string | null;
  needs_human: boolean | null;
  json_valid: boolean;
  schema_compliant: boolean;
  response: string;
  raw_output: string;
  generation_truncated: boolean;
  latency_ms: number | null;
  generation_error: string | null;
};

export type DemoCase = {
  id: string;
  label_zh: string;
  message: string;
  expected: ExpectedResult;
  base: ModelSnapshot;
  lora: ModelSnapshot;
  provenance: {
    source: "frozen_local_inference";
    candidate: "candidate_01";
    response_modified: false;
  };
};

export type ProjectStatus = {
  candidate: string;
  decision: "PROMOTE";
  approved_scope: string[];
  known_limitations: string[];
  deployment_constraints: {
    not_approved_as: string[];
    requires_external_grounding_for: string[];
    requires_backend_tools_for: string[];
    human_review_recommended_for: string[];
  };
  unrestricted_production_approval: false;
  sources: string[];
};
