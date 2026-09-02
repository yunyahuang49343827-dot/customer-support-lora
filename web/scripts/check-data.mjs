import { readFile } from "node:fs/promises";

const readJson = async (name) => JSON.parse(await readFile(new URL(`../data/${name}`, import.meta.url), "utf8"));
const [cases, benchmark, status] = await Promise.all([
  readJson("demo_cases.json"),
  readJson("benchmark.json"),
  readJson("project_status.json"),
]);

if (!Array.isArray(cases) || cases.length !== 8) throw new Error("demo_cases.json must contain exactly 8 cases");
for (const demoCase of cases) {
  for (const key of ["id", "label_zh", "message", "expected", "base", "lora", "provenance"]) {
    if (!(key in demoCase)) throw new Error(`Missing ${key} in case ${demoCase.id ?? "unknown"}`);
  }
}
for (const key of ["intent_accuracy", "category_accuracy", "json_valid_rate", "schema_compliance", "escalation_accuracy", "escalation_f1"]) {
  if (!benchmark.metrics?.[key]) throw new Error(`Missing benchmark metric: ${key}`);
}
if (status.candidate !== "candidate_01" || status.decision !== "PROMOTE") throw new Error("Invalid project promotion status");
if (status.unrestricted_production_approval !== false) throw new Error("Unrestricted production approval must remain false");

console.log("Static data validation passed: 8 cases, 6 metrics, Candidate 01 PROMOTE with constraints.");
