import type { ExpectedResult, ModelSnapshot } from "@/lib/types";

type ResultCardProps = {
  title: string;
  result: ModelSnapshot;
  expected: ExpectedResult;
  tone: "base" | "lora";
};

const tones = {
  base: "border-t-rose-300 bg-rose-50/30",
  lora: "border-t-sky-400 bg-sky-50/30",
};

function match(actual: string | boolean | null, expected: string | boolean) {
  return actual === expected ? "✅ 符合" : "❌ 不符合";
}

function value(actual: string | boolean | null) {
  return actual === null ? "無法取得" : String(actual);
}

export function ResultCard({ title, result, expected, tone }: ResultCardProps) {
  const rows = [
    ["意圖", value(result.intent), match(result.intent, expected.intent)],
    ["類別", value(result.category), match(result.category, expected.category)],
    ["需要人工處理", value(result.needs_human), match(result.needs_human, expected.needs_human)],
    ["JSON 有效", result.json_valid ? "有效" : "無效", result.json_valid ? "✅ 有效" : "❌ 無效"],
    ["Schema 合規", result.schema_compliant ? "合規" : "不合規", result.schema_compliant ? "✅ 合規" : "❌ 不合規"],
  ];

  return (
    <article className={`min-w-0 overflow-hidden rounded-2xl border border-slate-200 border-t-4 p-5 shadow-[0_8px_30px_rgba(30,41,59,0.05)] ${tones[tone]}`}>
      <h3 className="text-lg font-bold text-slate-900">{title}</h3>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[420px] border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-slate-200 text-slate-500">
              <th className="px-2 py-2 font-semibold">項目</th>
              <th className="px-2 py-2 font-semibold">結果</th>
              <th className="px-2 py-2 font-semibold">狀態</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, actual, status]) => (
              <tr className="border-b border-slate-200/80 last:border-0" key={String(label)}>
                <td className="px-2 py-3 text-slate-600">{label}</td>
                <td className="break-words px-2 py-3 font-semibold text-slate-900">{actual}</td>
                <td className="whitespace-nowrap px-2 py-3 text-slate-700">{status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {result.generation_error ? (
        <p className="mt-4 rounded-lg bg-red-50 px-3 py-2 text-sm text-red-700">此 snapshot 發生 generation failure。</p>
      ) : null}
    </article>
  );
}
