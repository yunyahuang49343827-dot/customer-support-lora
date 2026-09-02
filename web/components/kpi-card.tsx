import type { MetricSnapshot } from "@/lib/types";

type KpiCardProps = {
  title: string;
  metric: MetricSnapshot;
  accent: "blue" | "green" | "purple";
  baseDigits?: number;
  loraDigits?: number;
};

const accentClasses = {
  blue: "border-t-blue-500 text-blue-700",
  green: "border-t-emerald-500 text-emerald-700",
  purple: "border-t-violet-500 text-violet-700",
};

export function KpiCard({ title, metric, accent, baseDigits = 1, loraDigits = 1 }: KpiCardProps) {
  return (
    <article className={`min-w-0 rounded-2xl border border-slate-200 border-t-4 bg-white p-5 shadow-[0_8px_30px_rgba(30,41,59,0.05)] ${accentClasses[accent]}`}>
      <p className="text-sm font-semibold text-slate-600">{title}</p>
      <p className="mt-3 text-2xl font-bold tracking-tight text-slate-900 sm:text-[1.7rem]">
        {metric.base.toFixed(baseDigits)}% <span className="text-slate-400">→</span> {metric.lora.toFixed(loraDigits)}%
      </p>
      <p className="mt-1.5 text-sm font-bold">+{metric.absolute_delta.toFixed(1)} 個百分點</p>
    </article>
  );
}
