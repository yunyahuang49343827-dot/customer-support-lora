"use client";

import { useState } from "react";

import { KpiCard } from "@/components/kpi-card";
import { ResponseCard } from "@/components/response-card";
import { ResultCard } from "@/components/result-card";
import type { BenchmarkSnapshot, DemoCase, ProjectStatus } from "@/lib/types";

type PortfolioDemoProps = {
  benchmark: BenchmarkSnapshot;
  cases: DemoCase[];
  projectStatus: ProjectStatus;
};

const benchmarkLabels: Record<string, string> = {
  intent_accuracy: "Intent Accuracy",
  category_accuracy: "Category Accuracy",
  json_valid_rate: "JSON Valid",
  schema_compliance: "Schema Compliance",
  escalation_accuracy: "Escalation Accuracy",
  escalation_f1: "Escalation F1",
};

const benchmarkOrder = [
  "intent_accuracy",
  "category_accuracy",
  "json_valid_rate",
  "schema_compliance",
  "escalation_accuracy",
  "escalation_f1",
];

export function PortfolioDemo({ benchmark, cases, projectStatus }: PortfolioDemoProps) {
  const [selectedId, setSelectedId] = useState(cases[0]?.id ?? "");
  const selected = cases.find((item) => item.id === selectedId) ?? cases[0];

  if (!selected) {
    return <main className="mx-auto max-w-7xl px-5 py-16 text-slate-700">靜態 demo 資料無法載入。</main>;
  }

  const metrics = benchmark.metrics;
  const constraints = projectStatus.deployment_constraints;

  return (
    <main className="mx-auto w-full max-w-7xl px-4 py-10 sm:px-6 lg:px-8 lg:py-16">
      <header className="max-w-4xl">
        <p className="mb-4 inline-flex rounded-full border border-blue-200 bg-blue-50 px-3 py-1 text-xs font-bold tracking-wide text-blue-700">
          FROZEN SNAPSHOT PORTFOLIO DEMO
        </p>
        <h1 className="text-3xl font-extrabold tracking-tight text-slate-950 sm:text-4xl lg:text-5xl">
          客服分類 LoRA 微調效果比較
        </h1>
        <p className="mt-4 text-base leading-7 text-slate-600 sm:text-lg">
          在固定的 Locked Test 上，微調顯著提升結構化分類與轉介決策能力。
        </p>
      </header>

      <section aria-label="核心成效" className="mt-9 grid gap-4 md:grid-cols-3">
        <KpiCard title="意圖分類準確率" metric={metrics.intent_accuracy} accent="blue" baseDigits={0} loraDigits={0} />
        <KpiCard title="Schema 合規率" metric={metrics.schema_compliance} accent="green" />
        <KpiCard title="轉介判斷準確率" metric={metrics.escalation_accuracy} accent="purple" baseDigits={0} />
      </section>

      <section className="mt-14" aria-labelledby="case-heading">
        <div className="flex flex-col gap-6 lg:flex-row lg:items-end lg:justify-between">
          <div>
            <h2 id="case-heading" className="text-2xl font-bold tracking-tight text-slate-900">測試案例</h2>
            <p className="mt-2 text-sm text-slate-500">選擇一筆由 Frozen 模型預先產生的真實 inference snapshot。</p>
          </div>
          <label className="block w-full max-w-sm text-sm font-semibold text-slate-700">
            案例
            <select
              className="mt-2 w-full rounded-xl border border-slate-300 bg-white px-4 py-3 text-slate-900 shadow-sm outline-none transition focus:border-blue-500 focus:ring-4 focus:ring-blue-100"
              value={selected.id}
              onChange={(event) => setSelectedId(event.target.value)}
            >
              {cases.map((item) => <option key={item.id} value={item.id}>{item.label_zh}</option>)}
            </select>
          </label>
        </div>

        <div className="mt-6 grid gap-5 lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
          <article className="min-w-0 rounded-2xl border border-slate-200 bg-white p-5 shadow-[0_8px_30px_rgba(30,41,59,0.04)]">
            <p className="text-xs font-bold uppercase tracking-wider text-slate-500">Customer Message</p>
            <p className="mt-3 break-words text-lg font-semibold leading-8 text-slate-900">{selected.message}</p>
          </article>
          <article className="min-w-0 rounded-2xl border border-violet-200 border-l-4 border-l-violet-400 bg-white p-5 shadow-[0_8px_30px_rgba(30,41,59,0.04)]">
            <h3 className="font-bold text-slate-900">預期結果（Expected）</h3>
            <dl className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
              {[
                ["意圖", selected.expected.intent],
                ["類別", selected.expected.category],
                ["需要人工處理", String(selected.expected.needs_human)],
              ].map(([label, value]) => (
                <div className="min-w-0" key={label}>
                  <dt className="text-xs text-slate-500">{label}</dt>
                  <dd className="mt-1 break-words text-sm font-bold text-slate-900">{value}</dd>
                </div>
              ))}
            </dl>
          </article>
        </div>
      </section>

      <section aria-label="模型結構化輸出比較" className="mt-7 grid gap-5 lg:grid-cols-2">
        <ResultCard title="Base Model" result={selected.base} expected={selected.expected} tone="base" />
        <ResultCard title="LoRA Candidate 01" result={selected.lora} expected={selected.expected} tone="lora" />
      </section>

      <section className="mt-10" aria-labelledby="response-heading">
        <h2 id="response-heading" className="text-xl font-bold text-slate-900">模型回覆</h2>
        <div className="mt-5 grid gap-5 lg:grid-cols-2">
          <ResponseCard title="Base Model 回覆" result={selected.base} tone="base" />
          <ResponseCard title="LoRA Candidate 01 回覆" result={selected.lora} tone="lora" />
        </div>
      </section>

      <p className="mt-8 rounded-xl border border-slate-200 bg-white/75 px-4 py-3 text-xs leading-5 text-slate-500">
        本公開展示使用 Frozen Base / Candidate 01 預先產生的真實 inference snapshots；完整自由輸入 Live Demo 於本機 MLX 環境執行。
      </p>

      <details className="mt-6 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_8px_30px_rgba(30,41,59,0.035)]">
        <summary className="flex cursor-pointer list-none items-center px-5 py-4 font-bold text-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600">
          查看 Locked Test 完整指標結果
        </summary>
        <div className="overflow-x-auto border-t border-slate-200 p-5">
          <table className="w-full min-w-[560px] border-collapse text-left text-sm">
            <thead><tr className="border-b border-slate-200 text-slate-500"><th className="px-2 py-2">指標</th><th className="px-2 py-2">Base</th><th className="px-2 py-2">LoRA</th><th className="px-2 py-2">改善</th></tr></thead>
            <tbody>
              {benchmarkOrder.map((key) => {
                const metric = metrics[key];
                return (
                  <tr className="border-b border-slate-100 last:border-0" key={key}>
                    <td className="px-2 py-3 font-medium text-slate-700">{benchmarkLabels[key]}</td>
                    <td className="px-2 py-3 text-slate-600">{metric.base.toFixed(1)}%</td>
                    <td className="px-2 py-3 font-semibold text-slate-900">{metric.lora.toFixed(1)}%</td>
                    <td className="px-2 py-3 font-semibold text-emerald-700">{metric.absolute_delta >= 0 ? "+" : ""}{metric.absolute_delta.toFixed(1)} pp</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </details>

      <section className="mt-10 rounded-2xl border border-emerald-200 border-l-4 border-l-emerald-500 bg-emerald-50/45 p-6 shadow-[0_8px_30px_rgba(30,41,59,0.035)]">
        <h2 className="text-xl font-bold text-emerald-900">✅ Candidate 01 已通過 Promotion Gate</h2>
        <p className="mt-4 text-sm font-semibold text-slate-700">適用範圍</p>
        <ul className="mt-2 grid gap-2 text-sm text-slate-700 sm:grid-cols-3">
          {projectStatus.approved_scope.slice(0, 3).map((item) => <li key={item}>• {item}</li>)}
        </ul>
        <p className="mt-5 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
          <strong>Promotion 不代表 unrestricted production approval。</strong> 此模型不得被視為不受限制的正式客服系統。
        </p>
      </section>

      <details className="mt-6 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-[0_8px_30px_rgba(30,41,59,0.035)]">
        <summary className="flex cursor-pointer list-none items-center px-5 py-4 font-bold text-slate-800 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600">
          查看模型限制與部署邊界
        </summary>
        <div className="grid gap-7 border-t border-slate-200 p-5 md:grid-cols-2">
          <div>
            <h3 className="font-bold text-slate-900">回覆限制</h3>
            <ul className="mt-3 space-y-2 text-sm leading-6 text-slate-600">
              {projectStatus.known_limitations.map((item) => <li className="break-words" key={item}>• {item}</li>)}
            </ul>
          </div>
          <div className="space-y-6">
            <div>
              <h3 className="font-bold text-slate-900">需要 external grounding</h3>
              <ul className="mt-3 space-y-2 text-sm text-slate-600">
                {constraints.requires_external_grounding_for.map((item) => <li key={item}>• {item}</li>)}
              </ul>
            </div>
            <div>
              <h3 className="font-bold text-slate-900">需要真實 tools / APIs</h3>
              <ul className="mt-3 space-y-2 text-sm text-slate-600">
                {constraints.requires_backend_tools_for.map((item) => <li key={item}>• {item}</li>)}
              </ul>
            </div>
          </div>
          <p className="rounded-xl border border-blue-100 bg-blue-50 p-4 text-sm leading-6 text-blue-950 md:col-span-2">
            QLoRA 顯著改善 structured classification behavior，但生成式 response 仍可能產生 unsupported policy/capability claims，因此模型不應直接被視為企業 factual authority。
          </p>
        </div>
      </details>
    </main>
  );
}
