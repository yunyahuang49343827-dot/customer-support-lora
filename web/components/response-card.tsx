import type { ModelSnapshot } from "@/lib/types";

type ResponseCardProps = {
  title: string;
  result: ModelSnapshot;
  tone: "base" | "lora";
};

const tones = {
  base: "border-l-rose-300",
  lora: "border-l-sky-400",
};

export function ResponseCard({ title, result, tone }: ResponseCardProps) {
  const response = result.response || "No response was generated.";
  return (
    <article className={`min-w-0 rounded-2xl border border-slate-200 border-l-4 bg-white p-5 shadow-[0_8px_30px_rgba(30,41,59,0.04)] ${tones[tone]}`}>
      <h3 className="text-sm font-bold text-slate-800">{title}</h3>
      <p className="mt-3 line-clamp-4 min-h-[5.75rem] break-words text-sm leading-6 text-slate-600">{response}</p>
      <details className="mt-4 border-t border-slate-200 pt-3">
        <summary className="flex cursor-pointer list-none items-center gap-2 text-sm font-semibold text-blue-700 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-blue-600">
          查看完整回覆與 JSON
        </summary>
        <div className="mt-4 space-y-4">
          <div>
            <p className="text-xs font-bold uppercase tracking-wide text-slate-500">完整 Response</p>
            <p className="mt-2 break-words text-sm leading-6 text-slate-700">{response}</p>
          </div>
          <div>
            <p className="text-xs font-bold uppercase tracking-wide text-slate-500">Raw JSON</p>
            <pre className="mt-2 max-h-80 overflow-auto rounded-xl bg-slate-950 p-4 text-xs leading-5 text-slate-100">{result.raw_output || "(empty output)"}</pre>
          </div>
          <div className="flex flex-wrap gap-2 text-xs font-semibold">
            <span className={`rounded-full px-2.5 py-1 ${result.json_valid ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"}`}>
              JSON：{result.json_valid ? "有效" : "無效"}
            </span>
            <span className={`rounded-full px-2.5 py-1 ${result.schema_compliant ? "bg-emerald-50 text-emerald-700" : "bg-red-50 text-red-700"}`}>
              Schema：{result.schema_compliant ? "合規" : "不合規"}
            </span>
          </div>
          {result.generation_truncated ? (
            <p className="rounded-lg bg-amber-50 px-3 py-2 text-sm text-amber-800">此輸出達到 frozen token 上限，可能遭到截斷。</p>
          ) : null}
        </div>
      </details>
    </article>
  );
}
